"""
PDF RAG 知识库查询中心

供 Layer 4 工具层调用的查询接口，封装向量检索和精确检索逻辑。
与原有的 EmeiKnowledgeBase 完全隔离，使用独立的 pdf_rag_collection。
"""
from __future__ import annotations

import json
import logging
from typing import List, Dict, Optional, Any

from configs.settings import settings
from server.pdf_rag.layer3_chunking.indexer import (
    get_chroma_collection,
    load_doc_chunks,
    load_all_doc_ids,
    PDF_DOCSTORE_DIR,
    DOC_REGISTRY_PATH,
)

logger = logging.getLogger(__name__)


class PDFKnowledgeBase:
    """PDF 专用知识库查询中心（单例）"""

    def __init__(self):
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = get_chroma_collection()
        return self._collection

    # =====================================================================
    # 向量检索
    # =====================================================================

    def search(
        self,
        query: str,
        top_k: int = 8,
        doc_id: Optional[str] = None,
        chunk_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        向量语义检索。

        Args:
            query:       检索问题
            top_k:       返回结果数
            doc_id:      限定检索文档（None 表示全库）
            chunk_types: 限定 chunk 类型，如 ["text", "table", "image_summary"]

        Returns:
            List of {chunk_id, text, metadata, distance}
        """
        # 计算 query 向量
        query_embedding = settings.embed_model.get_text_embedding(query)

        # 构建过滤条件
        where = {}
        if doc_id:
            where["doc_id"] = doc_id
        if chunk_types and len(chunk_types) == 1:
            where["chunk_type"] = chunk_types[0]
        elif chunk_types and len(chunk_types) > 1:
            where = {"$and": [where, {"chunk_type": {"$in": chunk_types}}]} if where else {
                "chunk_type": {"$in": chunk_types}
            }

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._get_collection_count()),
                where=where if where else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"[PDFHandler] 向量检索失败: {e}")
            return []

        return self._parse_query_results(results)

    # =====================================================================
    # 精确查询
    # =====================================================================

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """按 chunk_id 精确读取"""
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )
            if result["ids"]:
                return {
                    "chunk_id": chunk_id,
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
        except Exception as e:
            logger.error(f"[PDFHandler] get_chunk_by_id 失败: {e}")
        return None

    def get_page_chunks(
        self,
        doc_id: str,
        page_num: int,
    ) -> List[Dict[str, Any]]:
        """读取指定文档指定页的所有 chunk"""
        try:
            result = self.collection.get(
                where={"$and": [{"doc_id": doc_id}, {"page_num": page_num}]},
                include=["documents", "metadatas"],
            )
            chunks = []
            for cid, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            ):
                chunks.append({"chunk_id": cid, "text": doc, "metadata": meta})
            # 按 chunk_index 排序
            chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
            return chunks
        except Exception as e:
            logger.error(f"[PDFHandler] get_page_chunks 失败: {e}")
            return []

    def get_table_chunk(
        self,
        doc_id: str,
        table_id: str,
    ) -> Optional[Dict[str, Any]]:
        """按 table_id 精确读取表格 chunk"""
        try:
            result = self.collection.get(
                where={"$and": [{"doc_id": doc_id}, {"table_id": table_id}]},
                include=["documents", "metadatas"],
            )
            if result["ids"]:
                return {
                    "chunk_id": result["ids"][0],
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
        except Exception as e:
            logger.error(f"[PDFHandler] get_table_chunk 失败: {e}")
        return None

    def get_image_chunk(
        self,
        doc_id: str,
        image_id: str,
    ) -> Optional[Dict[str, Any]]:
        """按 image_id 精确读取图片摘要 chunk"""
        try:
            result = self.collection.get(
                where={"$and": [{"doc_id": doc_id}, {"image_id": image_id}]},
                include=["documents", "metadatas"],
            )
            if result["ids"]:
                return {
                    "chunk_id": result["ids"][0],
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
        except Exception as e:
            logger.error(f"[PDFHandler] get_image_chunk 失败: {e}")
        return None

    def list_indexed_docs(self) -> List[Dict[str, Any]]:
        """返回已入库的文档列表（从注册表读取）"""
        if not DOC_REGISTRY_PATH.exists():
            return []
        with open(DOC_REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)
        return [
            {"doc_id": doc_id, **info}
            for doc_id, info in registry.get("docs", {}).items()
        ]

    # =====================================================================
    # 私有辅助
    # =====================================================================

    def _get_collection_count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 100

    def _parse_query_results(self, results: dict) -> List[Dict[str, Any]]:
        """解析 ChromaDB query 返回结果"""
        chunks = []
        if not results or not results.get("ids"):
            return chunks

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[]])[0]

        for i, (cid, doc, meta) in enumerate(zip(ids, documents, metadatas)):
            distance = distances[i] if i < len(distances) else 1.0
            # 余弦距离 → 相似度分数（0~1，越高越相关）
            score = max(0.0, 1.0 - distance)

            # 反序列化 section_path（存储时 JSON 字符串）
            if "section_path" in meta and isinstance(meta["section_path"], str):
                try:
                    meta["section_path"] = json.loads(meta["section_path"])
                except Exception:
                    meta["section_path"] = []

            chunks.append({
                "chunk_id": cid,
                "text": doc,
                "metadata": meta,
                "score": round(score, 4),
            })

        return chunks


# 全局单例
pdf_kb = PDFKnowledgeBase()
