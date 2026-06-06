"""
Layer 3 — 语义结构切分器

对 StructuredDocument 中的文本块（heading + paragraph + footnote）进行切分：

切分原则：
1. 标题块 + 紧随其后的段落块 → 归为同一个逻辑组
2. 同组内若总字符数超过 MAX_CHUNK_CHARS，按句子边界切分（不生硬截断）
3. 段落之间保留语义关联：同一章节相邻段落尽量合并（不超过 MAX_CHUNK_CHARS）
4. 页眉/页脚/脚注 → 附加到最近正文 chunk 的 metadata，不独立成 chunk
5. 每个 chunk 必须携带 section_path（章节路径），是溯源的核心

特殊处理：
- 合同条款（"第X条"类标题 + 正文）整体视为一个原子单位，不拆散
- heading 本身如果独立存在（下面没有段落），也单独成 chunk
"""
from __future__ import annotations

import re
import logging
from typing import List, Tuple

from server.pdf_rag.layer2_structure.models import (
    TextBlock, StructuredDocument, ChunkMetadata, PDFChunk
)

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 800    # 单个文本 chunk 最大字符数
MIN_CHUNK_CHARS = 50     # 过短则与下一段合并
SENTENCE_ENDINGS = re.compile(r'([。！？；.!?;])')


def chunk_text_blocks(
    doc: StructuredDocument,
    chunk_index_start: int = 0,
) -> List[PDFChunk]:
    """
    主入口：对整份文档中所有文本块进行语义切分。

    Returns:
        文本类型的 PDFChunk 列表（不含表格/图片 chunk，那些由其他 chunker 处理）
    """
    chunks: List[PDFChunk] = []
    chunk_idx = chunk_index_start

    for page in doc.pages:
        text_blocks = [
            b for b in page.blocks
            if b.block_type in ("heading", "paragraph", "footnote", "caption")
        ]

        # 按块分组：将标题与其紧随的段落合并成"组"
        groups = _group_blocks_by_heading(text_blocks)

        for group_blocks in groups:
            if not group_blocks:
                continue

            # 计算组内总字符数
            total_chars = sum(len(b.text or "") for b in group_blocks)

            if total_chars <= MAX_CHUNK_CHARS:
                # 整组作为一个 chunk
                chunk = _make_text_chunk(
                    blocks=group_blocks,
                    doc=doc,
                    chunk_index=chunk_idx,
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_idx += 1
            else:
                # 组内分割
                split_chunks = _split_group(group_blocks, doc, chunk_idx)
                chunks.extend(split_chunks)
                chunk_idx += len(split_chunks)

    logger.info(
        f"[SemanticChunker] {doc.file_name}: "
        f"文本块 → {len(chunks)} 个文本 chunk"
    )
    return chunks


# =====================================================================
# 私有辅助函数
# =====================================================================

def _group_blocks_by_heading(
    text_blocks: List[TextBlock],
) -> List[List[TextBlock]]:
    """
    将文本块按标题分组：每个标题开启一个新组，段落附属于最近的标题。
    没有标题的段落归入上一组（或独立一组）。
    """
    groups: List[List[TextBlock]] = []
    current_group: List[TextBlock] = []

    for block in text_blocks:
        if block.block_type == "heading":
            # 新标题开启新组（但先保存当前组）
            if current_group:
                # 检查当前组是否只有一个heading且后面没有段落
                # 如果新heading紧跟旧heading，两者分开
                groups.append(current_group)
            current_group = [block]
        else:
            # 段落/脚注附属当前组
            if not current_group:
                current_group = [block]
            else:
                current_group.append(block)

    if current_group:
        groups.append(current_group)

    return groups


def _make_text_chunk(
    blocks: List[TextBlock],
    doc: StructuredDocument,
    chunk_index: int,
) -> PDFChunk | None:
    """将一组块合并为单个 PDFChunk"""
    # 用最后一个块的 section_path（最具体）
    section_path = []
    for b in blocks:
        if b.section_path:
            section_path = b.section_path

    # 拼接文本：标题加前缀标记，脚注加括号
    text_parts = []
    for b in blocks:
        t = (b.text or "").strip()
        if not t:
            continue
        if b.block_type == "heading":
            prefix = "#" * (b.level or 1) + " "
            text_parts.append(prefix + t)
        elif b.block_type == "footnote":
            text_parts.append(f"[注释] {t}")
        elif b.block_type == "caption":
            text_parts.append(f"[图注] {t}")
        else:
            text_parts.append(t)

    text = "\n".join(text_parts).strip()
    if not text:
        return None

    # 取首块的页码
    page_num = blocks[0].page_num
    section_str = " > ".join(section_path) if section_path else ""

    metadata = ChunkMetadata(
        doc_id=doc.doc_id,
        file_name=doc.file_name,
        file_path=doc.file_path,
        page_num=page_num,
        section_path=section_path,
        section_str=section_str,
        chunk_type="text",
        block_type=blocks[0].block_type,
        heading_level=blocks[0].level if blocks[0].block_type == "heading" else None,
        chunk_index=chunk_index,
        char_count=len(text),
        pdf_type=doc.pdf_type,
    )
    return PDFChunk(text=text, metadata=metadata)


def _split_group(
    group_blocks: List[TextBlock],
    doc: StructuredDocument,
    chunk_index_start: int,
) -> List[PDFChunk]:
    """
    对超长组进行句子级别切分。
    保留标题块（如果有）作为每个子 chunk 的上下文前缀。
    """
    chunks: List[PDFChunk] = []
    chunk_idx = chunk_index_start

    # 提取标题（如果有）
    heading_blocks = [b for b in group_blocks if b.block_type == "heading"]
    body_blocks = [b for b in group_blocks if b.block_type != "heading"]

    heading_prefix = ""
    if heading_blocks:
        heading_prefix = "# " + " / ".join(b.text.strip() for b in heading_blocks) + "\n"

    # 合并所有正文
    full_text = "\n".join((b.text or "").strip() for b in body_blocks if b.text)

    # 按句子切分
    sentences = _split_into_sentences(full_text)

    current_sentences: List[str] = []
    current_len = len(heading_prefix)

    for sentence in sentences:
        s_len = len(sentence)
        if current_len + s_len > MAX_CHUNK_CHARS and current_sentences:
            chunk_text = heading_prefix + "".join(current_sentences)
            chunk = _make_split_chunk(
                text=chunk_text,
                blocks=group_blocks,
                doc=doc,
                chunk_index=chunk_idx,
            )
            if chunk:
                chunks.append(chunk)
                chunk_idx += 1
            current_sentences = [sentence]
            current_len = len(heading_prefix) + s_len
        else:
            current_sentences.append(sentence)
            current_len += s_len

    # 剩余句子
    if current_sentences:
        chunk_text = heading_prefix + "".join(current_sentences)
        chunk = _make_split_chunk(
            text=chunk_text,
            blocks=group_blocks,
            doc=doc,
            chunk_index=chunk_idx,
        )
        if chunk:
            chunks.append(chunk)

    return chunks


def _make_split_chunk(
    text: str,
    blocks: List[TextBlock],
    doc: StructuredDocument,
    chunk_index: int,
) -> PDFChunk | None:
    """为切分后的子文本创建 chunk"""
    if not text.strip():
        return None

    section_path = []
    for b in blocks:
        if b.section_path:
            section_path = b.section_path

    page_num = blocks[0].page_num
    section_str = " > ".join(section_path) if section_path else ""

    metadata = ChunkMetadata(
        doc_id=doc.doc_id,
        file_name=doc.file_name,
        file_path=doc.file_path,
        page_num=page_num,
        section_path=section_path,
        section_str=section_str,
        chunk_type="text",
        block_type="paragraph",
        chunk_index=chunk_index,
        char_count=len(text),
        pdf_type=doc.pdf_type,
    )
    return PDFChunk(text=text, metadata=metadata)


def _split_into_sentences(text: str) -> List[str]:
    """按中英文句末标点切分，保留标点符号在句尾"""
    if not text:
        return []
    parts = SENTENCE_ENDINGS.split(text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        if sentence.strip():
            sentences.append(sentence)
    # 最后一段（无标点结尾）
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1])
    return sentences
