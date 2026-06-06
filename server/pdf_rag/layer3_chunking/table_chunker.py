"""
Layer 3 — 表格切分器

将 StructuredDocument 中的表格块转换为向量友好的文本 Chunk。

策略：
1. 优先转换为 Markdown 表格格式（可读性强）
2. 额外生成"逐行字段描述"文本（提升向量检索覆盖率）
3. 长表格（行数 > MAX_ROWS_PER_CHUNK）按 MAX_ROWS_PER_CHUNK 行切分，
   每个子 chunk 重复表头，避免断章取义
"""
from __future__ import annotations

import logging
from typing import List, Optional

from server.pdf_rag.layer2_structure.models import TextBlock, ChunkMetadata, PDFChunk

logger = logging.getLogger(__name__)

MAX_ROWS_PER_CHUNK = 20   # 单个表格 chunk 最多包含的数据行数


def chunk_table(
    block: TextBlock,
    doc_id: str,
    file_name: str,
    file_path: str,
    pdf_type: str,
    chunk_index_start: int = 0,
) -> List[PDFChunk]:
    """
    将单个表格块拆分为一或多个 PDFChunk。

    Args:
        block:             表格 TextBlock
        doc_id:            所属文档 ID
        file_name:         文件名
        file_path:         文件路径
        pdf_type:          PDF 类型
        chunk_index_start: 起始 chunk 序号

    Returns:
        PDFChunk 列表
    """
    table_data = block.table_data or []
    if not table_data:
        return []

    # 清洗：过滤完全空行，cell 空值替换为空字符串
    cleaned = []
    for row in table_data:
        if row is None:
            continue
        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
        if any(cell for cell in cleaned_row):  # 至少有一个非空 cell
            cleaned.append(cleaned_row)

    if not cleaned:
        return []

    # 推断表头（第一行）
    header = cleaned[0]
    data_rows = cleaned[1:] if len(cleaned) > 1 else []

    # 生成表格 ID
    table_id = f"table_p{block.page_num}_{id(block) % 10000}"

    chunks: List[PDFChunk] = []

    if not data_rows:
        # 只有表头，直接作为单个 chunk
        text = _rows_to_markdown([header], [])
        chunks.append(_make_table_chunk(
            text=text,
            doc_id=doc_id,
            file_name=file_name,
            file_path=file_path,
            block=block,
            table_id=table_id,
            table_header=header,
            chunk_index=chunk_index_start,
            pdf_type=pdf_type,
        ))
        return chunks

    # 按 MAX_ROWS_PER_CHUNK 分批
    for batch_start in range(0, len(data_rows), MAX_ROWS_PER_CHUNK):
        batch_rows = data_rows[batch_start: batch_start + MAX_ROWS_PER_CHUNK]

        # Markdown 格式
        markdown_text = _rows_to_markdown(header, batch_rows)

        # 逐行字段描述（提升向量覆盖）
        row_desc_text = _rows_to_field_description(header, batch_rows)

        # 合并两种表示
        combined_text = markdown_text + "\n\n" + row_desc_text

        chunk_idx = chunk_index_start + len(chunks)
        chunks.append(_make_table_chunk(
            text=combined_text,
            doc_id=doc_id,
            file_name=file_name,
            file_path=file_path,
            block=block,
            table_id=table_id,
            table_header=header,
            chunk_index=chunk_idx,
            pdf_type=pdf_type,
        ))

    logger.debug(
        f"[TableChunker] 第 {block.page_num} 页表格 → "
        f"{len(chunks)} 个 chunk，共 {len(data_rows)} 行数据"
    )
    return chunks


def _make_table_chunk(
    text: str,
    doc_id: str,
    file_name: str,
    file_path: str,
    block: TextBlock,
    table_id: str,
    table_header: List[str],
    chunk_index: int,
    pdf_type: str,
) -> PDFChunk:
    section_str = " > ".join(block.section_path) if block.section_path else ""
    metadata = ChunkMetadata(
        doc_id=doc_id,
        file_name=file_name,
        file_path=file_path,
        page_num=block.page_num,
        section_path=block.section_path,
        section_str=section_str,
        chunk_type="table",
        block_type="table",
        table_id=table_id,
        table_header=table_header,
        chunk_index=chunk_index,
        char_count=len(text),
        pdf_type=pdf_type,
    )
    return PDFChunk(text=text, metadata=metadata)


def _rows_to_markdown(header: List[str], data_rows: List[List[str]]) -> str:
    """生成 Markdown 表格文本"""
    if not header:
        return ""
    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    header_line = "| " + " | ".join(_escape_md(c) for c in header) + " |"
    lines = [header_line, sep]
    for row in data_rows:
        # 确保行列数与表头一致（不足补空，多余截断）
        padded = (row + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(_escape_md(c) for c in padded) + " |")
    return "\n".join(lines)


def _rows_to_field_description(
    header: List[str],
    data_rows: List[List[str]],
) -> str:
    """
    将每行转为自然语言字段描述，例如：
    '第1行：姓名=张三，年龄=30，部门=技术部'
    """
    if not header or not data_rows:
        return ""
    desc_lines = []
    for i, row in enumerate(data_rows):
        padded = (row + [""] * len(header))[: len(header)]
        pairs = [
            f"{col}={val}"
            for col, val in zip(header, padded)
            if val.strip()
        ]
        if pairs:
            desc_lines.append(f"第{i + 1}行：{'，'.join(pairs)}")
    return "\n".join(desc_lines)


def _escape_md(text: str) -> str:
    """转义 Markdown 表格中的特殊字符"""
    return text.replace("|", "\\|").replace("\n", " ")
