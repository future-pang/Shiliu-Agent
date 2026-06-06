"""
Layer 2 — 结构还原引擎

将 Layer 1 输出的原始块（RawPage 列表）转换为结构化文档（StructuredDocument）：

1. 标题层级推断：
   - 根据字体大小的相对关系推断 H1/H2/H3/H4
   - 结合加粗属性、文字长度辅助判断
   - 维护滑动的 section_path（章节路径），为每个块打上溯源标签

2. 页眉/页脚识别：
   - 用跨页频率检测：在 ≥60% 页面出现相同文本 → 页眉或页脚
   - 单独标记，不影响正文章节路径

3. 脚注检测：
   - 页面底部小字、带序号的文本块

4. 所有块都携带 section_path（["第一章", "第二节"]），是溯源的核心。
"""
from __future__ import annotations

import re
import logging
import statistics
from collections import Counter
from typing import List, Optional, Dict, Tuple

from server.pdf_rag.layer2_structure.models import (
    RawPage, RawBlock, TextBlock, StructuredPage, StructuredDocument
)

logger = logging.getLogger(__name__)

# 标题候选：字体比中位大多少比例
HEADING_SIZE_RATIO = 1.15
# 标题候选：最大字符数（长段落不是标题）
HEADING_MAX_CHARS = 120
# 页眉/页脚跨页出现频率阈值（占总页数的比例）
HEADER_FOOTER_FREQ_RATIO = 0.5
# 脚注字体相对中位的比例（更小的字）
FOOTNOTE_SIZE_RATIO = 0.80


def build_structure(
    raw_pages: List[RawPage],
    file_name: str,
    file_path: str,
    doc_id: str,
    pdf_type: str,
) -> StructuredDocument:
    """
    主入口：输入原始页列表，输出结构化文档。
    """
    if not raw_pages:
        return StructuredDocument(
            doc_id=doc_id,
            file_name=file_name,
            file_path=file_path,
            total_pages=0,
            pdf_type=pdf_type,
        )

    # ---- Step 1: 统计全局字体大小分布，确定中位字体大小 ----------------
    median_font_size = _calc_median_font_size(raw_pages)
    logger.debug(f"[Structure] 全局中位字体大小: {median_font_size:.1f}pt")

    # ---- Step 2: 识别跨页重复文本（页眉/页脚）--------------------------
    header_footer_texts = _detect_header_footer_texts(raw_pages)
    logger.debug(f"[Structure] 检测到页眉/页脚文本 {len(header_footer_texts)} 条")

    # ---- Step 3: 逐页逐块进行结构分类 ----------------------------------
    structured_pages: List[StructuredPage] = []
    section_path: List[str] = []    # 当前章节路径，滑动维护

    for raw_page in raw_pages:
        s_page = StructuredPage(page_num=raw_page.page_num)
        page_height = raw_page.height

        for raw_block in raw_page.blocks:
            text_block = _classify_block(
                raw_block=raw_block,
                page_height=page_height,
                median_font_size=median_font_size,
                header_footer_texts=header_footer_texts,
                section_path=section_path,
            )
            if text_block is None:
                continue

            # 更新章节路径（如果这是个标题块）
            if text_block.block_type == "heading" and text_block.text.strip():
                _update_section_path(section_path, text_block)

            # 将当前 section_path 快照写入块（避免后续变更影响）
            text_block.section_path = section_path.copy()

            s_page.blocks.append(text_block)

        structured_pages.append(s_page)

    # ---- Step 4: 提取目录结构（仅收集标题块）---------------------------
    toc = _extract_toc(structured_pages)

    total_pages = max((p.page_num for p in raw_pages), default=0)

    doc = StructuredDocument(
        doc_id=doc_id,
        file_name=file_name,
        file_path=file_path,
        total_pages=total_pages,
        pdf_type=pdf_type,
        pages=structured_pages,
        toc=toc,
    )

    logger.info(
        f"[Structure] {file_name} 结构还原完成: "
        f"{total_pages} 页, {len(toc)} 个标题节点"
    )
    return doc


# =====================================================================
# 私有辅助函数
# =====================================================================

def _calc_median_font_size(raw_pages: List[RawPage]) -> float:
    """统计所有文本块的字体大小，返回中位数"""
    sizes = []
    for page in raw_pages:
        for block in page.blocks:
            if block.block_type in ("text", "header_raw", "footer_raw"):
                if block.font_size and block.font_size > 0:
                    sizes.append(block.font_size)
    if not sizes:
        return 12.0
    return statistics.median(sizes)


def _detect_header_footer_texts(raw_pages: List[RawPage]) -> set:
    """
    检测跨页重复出现的文本（页眉/页脚）。
    对已被标记为 header_raw/footer_raw 的块做频率统计。
    """
    text_counter: Counter = Counter()
    total_pages = len(raw_pages)

    for page in raw_pages:
        page_texts = set()
        for block in page.blocks:
            if block.block_type in ("header_raw", "footer_raw"):
                normalized = _normalize_text(block.text or "")
                if normalized and len(normalized) > 3:
                    page_texts.add(normalized)
        text_counter.update(page_texts)

    threshold = max(2, int(total_pages * HEADER_FOOTER_FREQ_RATIO))
    return {text for text, count in text_counter.items() if count >= threshold}


def _normalize_text(text: str) -> str:
    """去除页码数字和空格，用于文本比对"""
    return re.sub(r'\d+', '', text).strip()


def _classify_block(
    raw_block: RawBlock,
    page_height: float,
    median_font_size: float,
    header_footer_texts: set,
    section_path: List[str],
) -> Optional[TextBlock]:
    """
    将单个 RawBlock 分类为 TextBlock（带语义类型和层级）。
    """
    page_num = raw_block.page_num

    # ---- 表格块 --------------------------------------------------------
    if raw_block.block_type == "table":
        return TextBlock(
            page_num=page_num,
            block_type="table",
            table_data=raw_block.table_data,
            bbox=raw_block.bbox,
        )

    # ---- 图片块 --------------------------------------------------------
    if raw_block.block_type == "image":
        return TextBlock(
            page_num=page_num,
            block_type="image",
            image_bytes=raw_block.image_bytes,
            image_description=raw_block.image_description,
            bbox=raw_block.bbox,
        )

    # ---- 文本块 --------------------------------------------------------
    text = (raw_block.text or "").strip()
    if not text:
        return None

    font_size = raw_block.font_size or median_font_size
    is_bold = raw_block.is_bold or False

    # 检查是否是已识别的页眉/页脚
    if raw_block.block_type in ("header_raw", "footer_raw"):
        normalized = _normalize_text(text)
        if normalized in header_footer_texts or raw_block.block_type == "header_raw":
            return TextBlock(
                page_num=page_num,
                block_type="header" if raw_block.block_type == "header_raw" else "footer",
                text=text,
                font_size=font_size,
                bbox=raw_block.bbox,
            )

    # 脚注检测：字体比中位小 + 处于页面下半部分
    if raw_block.bbox:
        block_top = raw_block.bbox[1]
        is_lower_half = block_top > page_height * 0.70
        is_small_font = font_size < median_font_size * FOOTNOTE_SIZE_RATIO
        if is_lower_half and is_small_font and len(text) < 200:
            return TextBlock(
                page_num=page_num,
                block_type="footnote",
                text=text,
                font_size=font_size,
                bbox=raw_block.bbox,
            )

    # 标题判断：字体比中位大 + 不太长 + 通常加粗或全大写
    is_large_font = font_size >= median_font_size * HEADING_SIZE_RATIO
    is_short = len(text) <= HEADING_MAX_CHARS
    looks_like_heading = (is_large_font or is_bold) and is_short

    # 合同/法规文档的编号标题匹配（第X条、X.Y、第X章等）
    numbered_heading_pattern = re.compile(
        r'^(第[一二三四五六七八九十百千\d]+[章节条款部分编]'
        r'|\d+(\.\d+)*[、\.]?\s*\S'
        r'|[一二三四五六七八九十]+[、.])'
    )
    is_numbered = bool(numbered_heading_pattern.match(text))

    if looks_like_heading or is_numbered:
        level = _infer_heading_level(font_size, median_font_size, is_numbered, text)
        return TextBlock(
            page_num=page_num,
            block_type="heading",
            level=level,
            text=text,
            font_size=font_size,
            is_bold=is_bold,
            bbox=raw_block.bbox,
        )

    # 默认：普通段落
    return TextBlock(
        page_num=page_num,
        block_type="paragraph",
        text=text,
        font_size=font_size,
        is_bold=is_bold,
        bbox=raw_block.bbox,
    )


def _infer_heading_level(
    font_size: float,
    median_font_size: float,
    is_numbered: bool,
    text: str,
) -> int:
    """
    根据字体大小相对中位的比例推断标题层级（1-4）。
    同时结合编号模式辅助判断层级：
      第X章/第X部分 → H1
      第X节         → H2
      第X条         → H3
      X.Y.Z         → 由层级数推断
    """
    # 数字编号层级推断
    if is_numbered:
        if re.match(r'^第[一二三四五六七八九十百千\d]+[章部分编]', text):
            return 1
        if re.match(r'^第[一二三四五六七八九十百千\d]+节', text):
            return 2
        if re.match(r'^第[一二三四五六七八九十百千\d]+条', text):
            return 3
        # X.Y.Z 格式
        dot_match = re.match(r'^(\d+\.)+', text)
        if dot_match:
            depth = dot_match.group().count('.')
            return min(depth, 4)

    # 字体大小比例判断
    ratio = font_size / median_font_size if median_font_size > 0 else 1.0
    if ratio >= 1.6:
        return 1
    elif ratio >= 1.35:
        return 2
    elif ratio >= 1.15:
        return 3
    else:
        return 4


def _update_section_path(section_path: List[str], heading_block: TextBlock):
    """
    根据标题块更新章节路径（滑动窗口）。
    H1 → 清空路径，加入 H1
    H2 → 保留 H1，替换 H2
    H3 → 保留 H1/H2，替换 H3
    以此类推。
    """
    level = heading_block.level or 1
    text = heading_block.text.strip()

    # 截断到当前层级（保留上层路径）
    while len(section_path) >= level:
        section_path.pop()
    section_path.append(text)


def _extract_toc(structured_pages: List[StructuredPage]) -> List[dict]:
    """从结构化页面中提取目录结构"""
    toc = []
    for page in structured_pages:
        for block in page.blocks:
            if block.block_type == "heading":
                toc.append({
                    "level": block.level,
                    "title": block.text,
                    "page_num": page.page_num,
                })
    return toc
