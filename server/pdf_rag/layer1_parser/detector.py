"""
Layer 1 — PDF 类型检测器

判断一份 PDF 属于以下三种类型之一：
- native:  原生文本 PDF（可直接抽取文字）
- scanned: 扫描件 PDF（几乎全是图片，需要 OCR）
- mixed:   图文混排 PDF（部分页有文本，含图表/扫描插图）

依赖：pdfplumber
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Tuple, List

logger = logging.getLogger(__name__)

# 每页最少字符数，低于此值视为"无文本页"
TEXT_PAGE_MIN_CHARS = 50
# 文本页占比阈值
NATIVE_THRESHOLD = 0.75   # ≥75% 页有文本 → native
SCANNED_THRESHOLD = 0.15  # ≤15% 页有文本 → scanned


def detect_pdf_type(
    file_path: str | Path,
) -> Tuple[Literal["native", "scanned", "mixed"], List[int], List[int]]:
    """
    检测 PDF 类型。

    Returns:
        (pdf_type, text_page_indices, image_page_indices)
        - text_page_indices:  有可抽取文本的页（1-indexed）
        - image_page_indices: 几乎无文本（需要 OCR 或视觉）的页
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("请安装 pdfplumber: pip install pdfplumber")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {file_path}")

    text_pages: List[int] = []
    image_pages: List[int] = []

    with pdfplumber.open(str(file_path)) as pdf:
        total_pages = len(pdf.pages)
        if total_pages == 0:
            logger.warning(f"PDF 无页面: {file_path.name}")
            return "native", [], []

        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            try:
                text = page.extract_text() or ""
                char_count = len(text.strip())
            except Exception as e:
                logger.warning(f"第 {page_num} 页文本抽取失败: {e}")
                char_count = 0

            if char_count >= TEXT_PAGE_MIN_CHARS:
                text_pages.append(page_num)
            else:
                image_pages.append(page_num)

    text_ratio = len(text_pages) / total_pages if total_pages > 0 else 0

    if text_ratio >= NATIVE_THRESHOLD:
        pdf_type = "native"
    elif text_ratio <= SCANNED_THRESHOLD:
        pdf_type = "scanned"
    else:
        pdf_type = "mixed"

    logger.info(
        f"[Detector] {file_path.name} → 类型={pdf_type} | "
        f"文本页={len(text_pages)}/{total_pages} ({text_ratio:.0%})"
    )
    return pdf_type, text_pages, image_pages
