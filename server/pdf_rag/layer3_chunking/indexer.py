"""
Layer 3 — PDF 专用索引器

将所有 PDFChunk 持久化到两个独立存储：

1. ChromaDB Collection: "pdf_rag_collection"
   - 与原有 emei_collection 完全隔离
   - 存储向量 + metadata（序列化为 JSON 兼容格式）

2. 本地 JSON Docstore: storage/pdf_docstore/
   - 按 doc_id 存储所有 chunk（包含完整 metadata 和文本）
   - 供 read_page / extract_table / quote_source 精确查询
   - 额外维护 doc_registry.json（已入库文档注册表，用于去重）

向量计算复用项目现有的 embed_model（Dashscope text-embedding-v3）。
"""
from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import chromadb

from server.pdf_rag.layer2_structure.models import PDFChunk

logger = logging.getLogger(__name__)

# =====================================================================
# 存储路径常量
# =====================================================================
from configs.settings import settings

CHROMA_COLLECTION_NAME = "pdf_rag_collection"
PDF_DOCSTORE_DIR = settings.BASE_DIR / "storage" / "pdf_docstore"
DOC_REGISTRY_PATH = PDF_DOCSTORE_DIR / "doc_registry.json"


# =====================================================================
# 主入库函数
# =====================================================================

def index_chunks(
    chunks: List[PDFChunk],
    doc_id: str,
    file_path: str,
    force_reindex: bool = False,
) -> int:
    """
    将 chunks 写入 ChromaDB 和本地 docstore。

    Args:
        chunks:        已生成的 PDFChunk 列表
        doc_id:        文档唯一 ID
        file_path:     原始 PDF 文件路径（用于注册表）
        force_reindex: 为 True 时强制重新入库（忽略去重）

    Returns:
        成功入库的 chunk 数量
    """
    if not chunks:
        logger.warning(f"[Indexer] doc_id={doc_id}: 无 chunk，跳过入库")
        return 0

    # ---- 确保目录存在 ------------------------------------------------
    PDF_DOCSTORE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 去重检查 ----------------------------------------------------
    registry = _load_registry()
    file_hash = _compute_file_hash(file_path)

    if not force_reindex and file_hash in registry.get("hashes", {}):
        existing_doc_id = registry["hashes"][file_hash]
        logger.info(
            f"[Indexer] {Path(file_path).name} 已入库 (doc_id={existing_doc_id})，跳过。"
            f"如需重新入库请传 force_reindex=True"
        )
        return 0

    # ---- 补全 total_chunks 字段 -------------------------------------
    total = len(chunks)
    for chunk in chunks:
        chunk.metadata.total_chunks = total

    # ---- 计算向量 ----------------------------------------------------
    logger.info(f"[Indexer] 计算 {total} 个 chunk 的向量...")
    texts = [c.text for c in chunks]
    try:
        embeddings = _compute_embeddings(texts)
    except Exception as e:
        logger.error(f"[Indexer] 向量计算失败: {e}")
        raise

    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb

    # ---- 写入 ChromaDB -----------------------------------------------
    _write_to_chroma(chunks, doc_id)

    # ---- 写入本地 Docstore -------------------------------------------
    _write_to_docstore(chunks, doc_id)

    # ---- 更新注册表 --------------------------------------------------
    _update_registry(registry, file_hash, doc_id, file_path, total)

    logger.info(
        f"[Indexer] {Path(file_path).name} 入库完成: {total} 个 chunk，"
        f"doc_id={doc_id}"
    )
    return total


def delete_doc(doc_id: str):
    """从 ChromaDB 和 docstore 中删除指定文档的所有 chunk"""
    # 删除 ChromaDB 中的记录
    try:
        db = chromadb.PersistentClient(
            path=str(settings.BASE_DIR / "storage" / "pdf_chroma_db")
        )
        collection = db.get_or_create_collection(CHROMA_COLLECTION_NAME)
        collection.delete(where={"doc_id": doc_id})
        logger.info(f"[Indexer] 已从 ChromaDB 删除 doc_id={doc_id}")
    except Exception as e:
        logger.warning(f"[Indexer] ChromaDB 删除失败: {e}")

    # 删除 docstore 文件
    docstore_file = PDF_DOCSTORE_DIR / f"{doc_id}.json"
    if docstore_file.exists():
        docstore_file.unlink()
        logger.info(f"[Indexer] 已删除 docstore: {docstore_file}")


# =====================================================================
# 查询函数（供 pdf_handler.py 调用）
# =====================================================================

def get_chroma_collection():
    """返回 PDF 专用的 ChromaDB Collection"""
    db = chromadb.PersistentClient(
        path=str(settings.BASE_DIR / "storage" / "pdf_chroma_db")
    )
    return db.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_doc_chunks(doc_id: str) -> List[dict]:
    """从 docstore 加载指定文档的所有 chunk"""
    docstore_file = PDF_DOCSTORE_DIR / f"{doc_id}.json"
    if not docstore_file.exists():
        return []
    with open(docstore_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_doc_ids() -> List[str]:
    """返回所有已入库的 doc_id 列表"""
    registry = _load_registry()
    return list(registry.get("docs", {}).keys())


# =====================================================================
# 私有辅助函数
# =====================================================================

def _compute_embeddings(texts: List[str]) -> List[List[float]]:
    """批量计算文本向量，复用项目 embed_model"""
    embed_model = settings.embed_model
    # embed_model 是 LlamaIndex 的 OpenAIEmbedding，支持同步批量
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)
    return embeddings


def _write_to_chroma(chunks: List[PDFChunk], doc_id: str):
    """将 chunk 写入 ChromaDB"""
    collection = get_chroma_collection()

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for chunk in chunks:
        if chunk.embedding is None:
            continue
        ids.append(chunk.metadata.chunk_id)
        embeddings.append(chunk.embedding)
        documents.append(chunk.text)
        metadatas.append(_serialize_metadata(chunk.metadata))

    if not ids:
        return

    # ChromaDB upsert（幂等，已存在则覆盖）
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.debug(f"[Indexer] ChromaDB upsert: {len(ids)} 条记录")


def _write_to_docstore(chunks: List[PDFChunk], doc_id: str):
    """将 chunk 序列化写入本地 JSON docstore（按 doc_id 一个文件）"""
    doc_data = []
    for chunk in chunks:
        doc_data.append({
            "chunk_id": chunk.metadata.chunk_id,
            "text": chunk.text,
            "metadata": chunk.metadata.model_dump(),
        })

    docstore_file = PDF_DOCSTORE_DIR / f"{doc_id}.json"
    with open(docstore_file, "w", encoding="utf-8") as f:
        json.dump(doc_data, f, ensure_ascii=False, indent=2)

    logger.debug(f"[Indexer] Docstore 写入: {docstore_file}")


def _serialize_metadata(metadata) -> dict:
    """
    将 ChunkMetadata 序列化为 ChromaDB 兼容格式。
    ChromaDB metadata value 只支持 str/int/float/bool，不支持 list/None。
    """
    raw = metadata.model_dump()
    serialized = {}
    for k, v in raw.items():
        if v is None:
            serialized[k] = ""
        elif isinstance(v, list):
            serialized[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, (str, int, float, bool)):
            serialized[k] = v
        else:
            serialized[k] = str(v)
    return serialized


def _compute_file_hash(file_path: str) -> str:
    """计算文件 MD5，用于去重检查"""
    h = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except Exception:
        return file_path  # fallback：用路径
    return h.hexdigest()


def _load_registry() -> dict:
    """加载已入库文档注册表"""
    if not DOC_REGISTRY_PATH.exists():
        return {"docs": {}, "hashes": {}}
    with open(DOC_REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _update_registry(
    registry: dict,
    file_hash: str,
    doc_id: str,
    file_path: str,
    chunk_count: int,
):
    """更新并持久化注册表"""
    registry.setdefault("docs", {})[doc_id] = {
        "file_path": file_path,
        "file_name": Path(file_path).name,
        "chunk_count": chunk_count,
        "indexed_at": datetime.now().isoformat(),
    }
    registry.setdefault("hashes", {})[file_hash] = doc_id

    with open(DOC_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
