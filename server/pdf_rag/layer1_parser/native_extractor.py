"""
Layer 1 — 原生文本 PDF 抽取器

对 native / mixed 类型 PDF 中有文本的页，使用 pdfplumber 精细抽取：
- 文本块（带字体大小、字体名、加粗推断、坐标）
- 表格（结构化二维数组）
- 图片区域（坐标 + 裁剪出图片字节，供 vision_extractor 处理）

页眉/页脚通过 Y 坐标阈值自动过滤后单独保留。
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Optional

from server.pdf_rag.layer2_structure.models import RawBlock, RawPage

logger = logging.getLogger(__name__)

# 页眉/页脚区域占页面高度的比例阈值
HEADER_RATIO = 0.07   # 顶部 7%
FOOTER_RATIO = 0.07   # 底部 7%

# 字体大小分箱（相对于页面中位字体大小）
# 比中位大 20% 以上 → 候选标题
HEADING_SIZE_MULTIPLIER = 1.20


def _is_in_header_zone(top: float, page_height: float) -> bool:
    return top < page_height * HEADER_RATIO


def _is_in_footer_zone(bottom: float, page_height: float) -> bool:
    return bottom > page_height * (1 - FOOTER_RATIO)


def extract_native_pages(
    file_path: str | Path,
    target_pages: Optional[List[int]] = None,
) -> List[RawPage]:
    """
    抽取原生文本 PDF 中指定页（或全部页）的原始块。

    Args:
        file_path:     PDF 文件路径
        target_pages:  指定页码列表（1-indexed），None 表示全部页

    Returns:
        List[RawPage]，顺序与页码一致
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("请安装 pdfplumber: pip install pdfplumber")

    file_path = Path(file_path)
    raw_pages: List[RawPage] = []

    with pdfplumber.open(str(file_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            if target_pages is not None and page_num not in target_pages:
                continue

            raw_page = RawPage(
                page_num=page_num,
                width=float(page.width),
                height=float(page.height),
            )

            # ---- 1. 抽取表格 ------------------------------------------------
            tables = []
            try:
                tables = page.extract_tables() or []
            except Exception as e:
                logger.warning(f"第 {page_num} 页表格抽取失败: {e}")

            # 记录表格占据的区域，用于后续排除文字重叠
            table_bboxes = []
            for table_idx, table_data in enumerate(tables):
                if not table_data:
                    continue
                # 找该表在页面上的坐标
                try:
                    finder = page.find_tables()
                    if table_idx < len(finder):
                        tb = finder[table_idx]
                        bbox = (tb.bbox[0], tb.bbox[1], tb.bbox[2], tb.bbox[3])
                        table_bboxes.append(bbox)
                    else:
                        bbox = None
                except Exception:
                    bbox = None

                block = RawBlock(
                    block_type="table",
                    page_num=page_num,
                    table_data=table_data,
                    bbox=bbox,
                )
                raw_page.blocks.append(block)

            # ---- 2. 抽取图片 ------------------------------------------------
            try:
                images = page.images or []
            except Exception:
                images = []

            for img_info in images:
                try:
                    # 裁剪图片区域
                    img_bbox = (
                        img_info["x0"], img_info["top"],
                        img_info["x1"], img_info["bottom"],
                    )
                    cropped = page.crop(img_bbox)
                    img_bytes = _render_page_to_bytes(cropped)
                except Exception as e:
                    logger.debug(f"第 {page_num} 页图片裁剪失败: {e}")
                    img_bytes = None

                block = RawBlock(
                    block_type="image",
                    page_num=page_num,
                    image_bytes=img_bytes,
                    bbox=(
                        img_info.get("x0"), img_info.get("top"),
                        img_info.get("x1"), img_info.get("bottom"),
                    ),
                )
                raw_page.blocks.append(block)

            # ---- 3. 抽取文字字符块 -----------------------------------------
            try:
                words = page.extract_words(
                    extra_attrs=["fontname", "size"],
                    keep_blank_chars=False,
                ) or []
            except Exception as e:
                logger.warning(f"第 {page_num} 页文字抽取失败: {e}")
                words = []

            # 按行分组（相同 top ± 2 pt 视为同一行）
            lines = _group_words_into_lines(words)

            for line_info in lines:
                text = line_info["text"]
                top = line_info["top"]
                bottom = line_info["bottom"]
                x0 = line_info["x0"]
                x1 = line_info["x1"]
                font_size = line_info["size"]
                font_name = line_info["fontname"]

                if not text.strip():
                    continue

                # 判断是否在页眉/页脚区
                if _is_in_header_zone(top, float(page.height)):
                    block_type = "header_raw"
                elif _is_in_footer_zone(bottom, float(page.height)):
                    block_type = "footer_raw"
                else:
                    block_type = "text"

                # 粗略判断是否加粗（字体名包含 Bold/Heavy/Black 字样）
                is_bold = any(
                    kw in (font_name or "").lower()
                    for kw in ("bold", "heavy", "black", "demi")
                )

                block = RawBlock(
                    block_type=block_type,
                    page_num=page_num,
                    text=text,
                    font_size=font_size,
                    font_name=font_name,
                    is_bold=is_bold,
                    bbox=(x0, top, x1, bottom),
                )
                raw_page.blocks.append(block)

            raw_pages.append(raw_page)
            logger.debug(
                f"[NativeExtractor] 第 {page_num} 页完成: "
                f"{len(raw_page.blocks)} 个原始块"
            )

    return raw_pages


def _render_page_to_bytes(page_or_crop) -> Optional[bytes]:
    """将 pdfplumber 页面/裁剪区域渲染为 PNG 字节"""
    try:
        pil_img = page_or_crop.to_image(resolution=150).original
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _group_words_into_lines(words: list) -> List[dict]:
    """按 Y 坐标（top）将 word 分组成行，同行合并文字"""
    if not words:
        return []

    # 按 top 排序
    words = sorted(words, key=lambda w: (round(w.get("top", 0), 1), w.get("x0", 0)))

    lines = []
    current_line = None
    LINE_TOLERANCE = 3.0  # top 差值在 3pt 内视为同行

    for word in words:
        top = word.get("top", 0)
        if current_line is None or abs(top - current_line["top"]) > LINE_TOLERANCE:
            if current_line is not None:
                lines.append(current_line)
            current_line = {
                "top": top,
                "bottom": word.get("bottom", top + 12),
                "x0": word.get("x0", 0),
                "x1": word.get("x1", 0),
                "text": word.get("text", ""),
                "size": word.get("size", 12.0),
                "fontname": word.get("fontname", ""),
            }
        else:
            # 同行合并
            current_line["x1"] = max(current_line["x1"], word.get("x1", 0))
            current_line["bottom"] = max(current_line["bottom"], word.get("bottom", 0))
            current_line["text"] += " " + word.get("text", "")
            # 取本行最大字体（用于标题判断）
            if word.get("size", 0) > current_line["size"]:
                current_line["size"] = word["size"]
                current_line["fontname"] = word.get("fontname", "")

    if current_line is not None:
        lines.append(current_line)

    return lines
