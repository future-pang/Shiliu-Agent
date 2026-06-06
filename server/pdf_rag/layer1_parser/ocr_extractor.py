"""
Layer 1 — 扫描件 PDF OCR 抽取器

对扫描件 PDF（或 mixed 类型中无文本的页），流程：
  1. pdf2image 将每页 PDF 渲染为高分辨率 PIL 图片
  2. PaddleOCR 对图片进行中英双语 OCR
  3. 将识别结果封装为与 native_extractor 相同结构的 RawPage 列表

注意：
  - PaddleOCR 首次运行会下载模型（约 100MB），后续缓存本地
  - 扫描件不做表格/图片分离，统一以文本块输出（结构还原在 Layer 2）
  - 若 pdf2image 依赖的 poppler 未安装，会抛出清晰错误提示
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Optional

from server.pdf_rag.layer2_structure.models import RawBlock, RawPage

logger = logging.getLogger(__name__)

# OCR 识别置信度阈值，低于此值的结果丢弃
OCR_CONFIDENCE_THRESHOLD = 0.5
# 渲染分辨率（DPI），越高识别越准但越慢
RENDER_DPI = 200


def extract_ocr_pages(
    file_path: str | Path,
    target_pages: Optional[List[int]] = None,
) -> List[RawPage]:
    """
    对扫描件 PDF 进行 OCR 抽取。

    Args:
        file_path:    PDF 文件路径
        target_pages: 指定页码（1-indexed），None 表示全部页

    Returns:
        List[RawPage]，与 native_extractor 结构完全一致
    """
    file_path = Path(file_path)

    # ---- 依赖检查 --------------------------------------------------------
    try:
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFInfoNotInstalledError
    except ImportError:
        raise ImportError(
            "请安装 pdf2image: pip install pdf2image\n"
            "同时需要安装 poppler：\n"
            "  Windows: 下载 https://github.com/oschwartz10612/poppler-windows/releases\n"
            "           解压后将 bin/ 目录加入系统 PATH\n"
            "  Linux:   sudo apt install poppler-utils"
        )

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError(
            "请安装 PaddleOCR: pip install paddlepaddle paddleocr"
        )

    # ---- 初始化 OCR 引擎（单例避免重复加载） ----------------------------
    ocr_engine = _get_ocr_engine()

    # ---- 渲染 PDF 页面为图片 --------------------------------------------
    logger.info(f"[OCR] 开始渲染 {file_path.name} (DPI={RENDER_DPI})")
    try:
        pil_images = convert_from_path(
            str(file_path),
            dpi=RENDER_DPI,
            first_page=min(target_pages) if target_pages else 1,
            last_page=max(target_pages) if target_pages else None,
        )
    except Exception as e:
        raise RuntimeError(
            f"PDF 渲染失败（检查 poppler 是否安装并在 PATH 中）: {e}"
        )

    raw_pages: List[RawPage] = []
    start_page = min(target_pages) if target_pages else 1

    for idx, pil_img in enumerate(pil_images):
        page_num = start_page + idx
        if target_pages is not None and page_num not in target_pages:
            continue

        width, height = pil_img.size
        raw_page = RawPage(
            page_num=page_num,
            width=float(width),
            height=float(height),
        )

        # ---- OCR 识别 ---------------------------------------------------
        img_bytes = _pil_to_bytes(pil_img)
        try:
            ocr_results = ocr_engine.ocr(img_bytes, cls=True)
        except Exception as e:
            logger.error(f"第 {page_num} 页 OCR 失败: {e}")
            raw_pages.append(raw_page)
            continue

        if not ocr_results or not ocr_results[0]:
            logger.warning(f"第 {page_num} 页 OCR 无结果")
            raw_pages.append(raw_page)
            continue

        # ---- 解析 OCR 输出，按行组织 ------------------------------------
        # PaddleOCR 输出格式: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)]
        line_blocks = []
        for line in ocr_results[0]:
            if line is None:
                continue
            bbox_pts, (text, confidence) = line
            if confidence < OCR_CONFIDENCE_THRESHOLD:
                continue
            if not text.strip():
                continue

            # 取包围矩形
            xs = [pt[0] for pt in bbox_pts]
            ys = [pt[1] for pt in bbox_pts]
            x0, y0 = min(xs), min(ys)
            x1, y1 = max(xs), max(ys)

            line_blocks.append({
                "text": text.strip(),
                "bbox": (x0, y0, x1, y1),
                "top": y0,
                "bottom": y1,
                "confidence": confidence,
            })

        # 按行排序（top 升序）
        line_blocks.sort(key=lambda b: b["top"])

        # 合并相邻行（top 差值 < 8px 视为同行）
        merged_lines = _merge_adjacent_lines(line_blocks, y_tolerance=8)

        for line in merged_lines:
            block = RawBlock(
                block_type="text",
                page_num=page_num,
                text=line["text"],
                bbox=line["bbox"],
                # OCR 无字体元数据，暂用 None
                font_size=None,
                font_name=None,
                is_bold=None,
            )
            raw_page.blocks.append(block)

        raw_pages.append(raw_page)
        logger.info(
            f"[OCR] 第 {page_num} 页完成: {len(raw_page.blocks)} 行文字"
        )

    return raw_pages


# =====================================================================
# 私有辅助函数
# =====================================================================

_ocr_engine_instance = None


def _get_ocr_engine():
    """单例 PaddleOCR，避免重复初始化（每次初始化耗时约 2-3 秒）"""
    global _ocr_engine_instance
    if _ocr_engine_instance is None:
        from paddleocr import PaddleOCR
        logger.info("[OCR] 初始化 PaddleOCR 引擎（首次启动会下载模型）...")
        _ocr_engine_instance = PaddleOCR(
            use_angle_cls=True,
            lang="ch",            # 中英文混合识别
            show_log=False,
            use_gpu=False,        # CPU 模式，可按需改为 True
        )
        logger.info("[OCR] PaddleOCR 引擎就绪")
    return _ocr_engine_instance


def _pil_to_bytes(pil_img) -> bytes:
    """PIL Image → PNG bytes（PaddleOCR 接受 bytes/ndarray/path）"""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _merge_adjacent_lines(
    line_blocks: List[dict],
    y_tolerance: float = 8,
) -> List[dict]:
    """
    将 Y 坐标接近的行合并（处理 OCR 同一行被切成多段的情况）。
    """
    if not line_blocks:
        return []

    merged = []
    current = line_blocks[0].copy()

    for line in line_blocks[1:]:
        if abs(line["top"] - current["top"]) <= y_tolerance:
            # 同行合并：文字拼接，bbox 取最大包围矩形
            current["text"] += " " + line["text"]
            cur_bbox = current["bbox"]
            new_bbox = line["bbox"]
            current["bbox"] = (
                min(cur_bbox[0], new_bbox[0]),
                min(cur_bbox[1], new_bbox[1]),
                max(cur_bbox[2], new_bbox[2]),
                max(cur_bbox[3], new_bbox[3]),
            )
            current["bottom"] = max(current["bottom"], line["bottom"])
        else:
            merged.append(current)
            current = line.copy()

    merged.append(current)
    return merged
