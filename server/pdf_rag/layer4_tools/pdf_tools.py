"""
Layer 4 — PDF Agent 工具集

向 Agent 暴露 5 个专属工具，完整覆盖 PDF 的各类查询场景：

┌─────────────────┬──────────────────────────────────────┬─────────────────────────┐
│ 工具             │ 功能                                  │ 触发场景                │
├─────────────────┼──────────────────────────────────────┼─────────────────────────┤
│ search_pdf      │ 向量检索相关 chunk（文本/表格/图表均含） │ 用户问简单事实           │
│ read_page       │ 读取指定文档指定页的完整内容            │ 需要阅读特定页           │
│ extract_table   │ 按 table_id 返回结构化表格内容          │ 用户问表格数据           │
│ analyze_chart   │ 按 image_id 返回图表视觉描述            │ 用户问图表/流程图        │
│ quote_source    │ 返回标准引用：页码+章节+原文片段         │ 生成带溯源的回答         │
└─────────────────┴──────────────────────────────────────┴─────────────────────────┘

每个工具返回值都包含 citation 字段，便于 Agent 在最终回答中嵌入引用。
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import tool

from server.pdf_rag.pdf_handler import pdf_kb

logger = logging.getLogger(__name__)


# =====================================================================
# 工具 1：向量语义检索
# =====================================================================

@tool
async def search_pdf(
    query: str,
    doc_id: str = "",
    chunk_type: str = "all",
    top_k: int = 6,
) -> str:
    """
    【PDF 检索】在 PDF 知识库中进行向量语义检索，返回最相关的片段及来源引用。

    参数说明：
    - query:      检索问题，尽量具体
    - doc_id:     限定检索某个文档（可为空，空则搜全库）
    - chunk_type: 限定类型 "text"(文本) / "table"(表格) / "image_summary"(图表) / "all"(全部)
    - top_k:      返回结果数量（默认 6）

    返回：包含内容、页码、章节、相似度的检索结果列表。
    调用此工具后，如需精读某个片段，使用 quote_source 获取完整引用。
    """
    logger.info(f"[search_pdf] query='{query}', doc_id='{doc_id}', type={chunk_type}")

    chunk_types = None if chunk_type == "all" else [chunk_type]
    results = pdf_kb.search(
        query=query,
        top_k=top_k,
        doc_id=doc_id or None,
        chunk_types=chunk_types,
    )

    if not results:
        return "未在 PDF 知识库中找到相关内容。请确认 PDF 已入库，或更换检索词。"

    output_parts = [f"PDF 检索结果（共 {len(results)} 条）：\n"]

    for i, r in enumerate(results):
        meta = r["metadata"]
        section_path = meta.get("section_path", [])
        if isinstance(section_path, str):
            try:
                section_path = json.loads(section_path)
            except Exception:
                section_path = []

        section_str = " > ".join(section_path) if section_path else "（无章节信息）"
        chunk_type_label = {
            "text": "正文", "table": "表格", "image_summary": "图表"
        }.get(meta.get("chunk_type", ""), "未知")

        # 截取前 300 字作为预览
        preview = r["text"][:300] + ("..." if len(r["text"]) > 300 else "")

        output_parts.append(
            f"--- 结果 {i + 1} [{chunk_type_label}] ---\n"
            f"来源：{meta.get('file_name', '未知文档')}\n"
            f"页码：第 {meta.get('page_num', '?')} 页\n"
            f"章节：{section_str}\n"
            f"相似度：{r.get('score', 0):.3f}\n"
            f"Chunk_ID：{r['chunk_id']}\n"
            f"内容预览：{preview}\n"
        )

    output_parts.append(
        "\n【系统提示】以上为预览摘要。"
        "如需精确引用，请调用 quote_source(chunk_id=...) 获取完整原文和引用格式。"
        "如需阅读完整页面，请调用 read_page(doc_id=..., page_num=...)。"
    )

    return "\n".join(output_parts)


# =====================================================================
# 工具 2：读取指定页
# =====================================================================

@tool
async def read_page(doc_id: str, page_num: int) -> str:
    """
    【PDF 翻页】读取指定 PDF 文档的指定页完整内容（包含该页所有 chunk）。

    参数说明：
    - doc_id:    文档 ID（可通过 search_pdf 返回的 metadata 获取）
    - page_num:  页码（1-indexed）

    适用场景：用户要求"看第X页"、需要完整读取某页内容、复杂对比需要逐页阅读。
    """
    logger.info(f"[read_page] doc_id='{doc_id}', page_num={page_num}")

    chunks = pdf_kb.get_page_chunks(doc_id=doc_id, page_num=page_num)

    if not chunks:
        return (
            f"未找到文档 {doc_id} 第 {page_num} 页的内容。\n"
            "请确认 doc_id 正确，或使用 search_pdf 先定位文档。"
        )

    output_parts = [f"📄 {chunks[0]['metadata'].get('file_name', doc_id)} — 第 {page_num} 页\n"]
    output_parts.append("=" * 50)

    for chunk in chunks:
        meta = chunk["metadata"]
        chunk_type = meta.get("chunk_type", "text")
        section_str = meta.get("section_str", "")

        type_label = {"text": "正文", "table": "表格", "image_summary": "图表"}.get(chunk_type, "")

        if section_str:
            output_parts.append(f"\n[{type_label}] 所属章节：{section_str}")
        else:
            output_parts.append(f"\n[{type_label}]")

        output_parts.append(chunk["text"])
        output_parts.append(f"Chunk_ID: {chunk['chunk_id']}")
        output_parts.append("-" * 30)

    return "\n".join(output_parts)


# =====================================================================
# 工具 3：抽取表格
# =====================================================================

@tool
async def extract_table(doc_id: str, table_id: str) -> str:
    """
    【PDF 表格提取】按 table_id 精确提取指定表格的完整结构化内容。

    参数说明：
    - doc_id:   文档 ID
    - table_id: 表格 ID（格式为 table_pX_Y，X=页码，可在 search_pdf 结果中找到）

    返回：Markdown 格式表格 + 字段说明，适合直接引用或进一步分析。
    适用场景：用户问"第X页的表格里XXX字段是什么"、需要对比多个表格数据。
    """
    logger.info(f"[extract_table] doc_id='{doc_id}', table_id='{table_id}'")

    chunk = pdf_kb.get_table_chunk(doc_id=doc_id, table_id=table_id)

    if not chunk:
        return (
            f"未找到表格 {table_id}。\n"
            "提示：使用 search_pdf(query='表格内容关键词', chunk_type='table') 先定位表格。"
        )

    meta = chunk["metadata"]
    section_str = meta.get("section_str", "（无章节信息）")
    header = meta.get("table_header", "")
    if isinstance(header, str):
        try:
            header = json.loads(header)
        except Exception:
            header = []

    output = (
        f"📊 表格内容\n"
        f"来源：{meta.get('file_name', doc_id)}\n"
        f"页码：第 {meta.get('page_num', '?')} 页\n"
        f"章节：{section_str}\n"
        f"表格ID：{table_id}\n"
        f"Chunk_ID：{chunk['chunk_id']}\n"
        f"\n{chunk['text']}"
    )
    return output


# =====================================================================
# 工具 4：图表视觉分析
# =====================================================================

@tool
async def analyze_chart(doc_id: str, image_id: str) -> str:
    """
    【PDF 图表分析】获取指定图表/流程图/插图的视觉理解描述。

    参数说明：
    - doc_id:   文档 ID
    - image_id: 图片 ID（格式为 img_pX_Y，X=页码，可在 search_pdf 结果中找到）

    返回：图表的完整自然语言描述（由多模态视觉模型生成）。
    适用场景：用户问"这个流程图说明了什么"、"图3显示了什么趋势"。
    """
    logger.info(f"[analyze_chart] doc_id='{doc_id}', image_id='{image_id}'")

    chunk = pdf_kb.get_image_chunk(doc_id=doc_id, image_id=image_id)

    if not chunk:
        return (
            f"未找到图片 {image_id}。\n"
            "提示：使用 search_pdf(query='图表描述关键词', chunk_type='image_summary') 先定位图表。"
        )

    meta = chunk["metadata"]
    section_str = meta.get("section_str", "（无章节信息）")

    output = (
        f"🖼️ 图表分析\n"
        f"来源：{meta.get('file_name', doc_id)}\n"
        f"页码：第 {meta.get('page_num', '?')} 页\n"
        f"章节：{section_str}\n"
        f"图片ID：{image_id}\n"
        f"Chunk_ID：{chunk['chunk_id']}\n"
        f"\n图表描述：\n{chunk['text']}"
    )
    return output


# =====================================================================
# 工具 5：溯源引用
# =====================================================================

@tool
async def quote_source(chunk_id: str) -> str:
    """
    【PDF 引用溯源】根据 chunk_id 返回完整的原文内容和标准引用格式。

    参数说明：
    - chunk_id: chunk 的唯一 ID（从 search_pdf / read_page 结果中获取）

    返回：完整原文 + 标准引用（文件名、页码、章节路径）。
    在最终回答中必须调用此工具提供引用，避免幻觉。

    【重要】生成回答时，请在相关内容后面附上引用信息，格式如：
    「...某某内容... [来源：XX文档 第X页 第X章第X节]」
    """
    logger.info(f"[quote_source] chunk_id='{chunk_id}'")

    chunk = pdf_kb.get_chunk_by_id(chunk_id=chunk_id)

    if not chunk:
        return f"未找到 chunk_id={chunk_id} 的内容。请检查 ID 是否正确。"

    meta = chunk["metadata"]
    section_path = meta.get("section_path", [])
    if isinstance(section_path, str):
        try:
            section_path = json.loads(section_path)
        except Exception:
            section_path = []

    section_str = " > ".join(section_path) if section_path else "（无章节信息）"
    chunk_type = meta.get("chunk_type", "text")
    type_label = {"text": "正文", "table": "表格", "image_summary": "图表"}.get(chunk_type, "")

    # 标准引用格式
    citation_str = (
        f"[来源：{meta.get('file_name', '未知文档')} "
        f"第 {meta.get('page_num', '?')} 页 "
        f"{'| ' + section_str if section_path else ''}]"
    )

    output = (
        f"📌 引用溯源\n"
        f"文件：{meta.get('file_name', '未知文档')}\n"
        f"页码：第 {meta.get('page_num', '?')} 页\n"
        f"章节：{section_str}\n"
        f"类型：{type_label}\n"
        f"标准引用格式：{citation_str}\n"
        f"\n完整原文：\n{chunk['text']}\n"
        f"\n【系统提示】请在回答中使用上面的「标准引用格式」注明来源。"
    )
    return output


# =====================================================================
# 工具注册表（供 Agent 导入）
# =====================================================================

PDF_TOOLS = [search_pdf, read_page, extract_table, analyze_chart, quote_source]
