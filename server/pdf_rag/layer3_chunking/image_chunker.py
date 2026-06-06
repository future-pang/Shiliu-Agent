"""
Layer 3 — 图片/图表 Chunk 生成器

对 StructuredDocument 中类型为 "image" 的块：
1. 调用 Layer 1 的 vision_extractor 获取视觉描述（若未配置则用占位符）
2. 以描述文本作为 chunk 的向量化内容
3. metadata 中携带 image_id、image_description，便于 Agent 工具精确检索
"""
from __future__ import annotations

import logging
from typing import List, Optional

from server.pdf_rag.layer2_structure.models import TextBlock, ChunkMetadata, PDFChunk
from server.pdf_rag.layer1_parser.vision_extractor import analyze_image

logger = logging.getLogger(__name__)


def chunk_image(
    block: TextBlock,
    doc_id: str,
    file_name: str,
    file_path: str,
    pdf_type: str,
    chunk_index: int = 0,
) -> Optional[PDFChunk]:
    """
    将单个图片块转为 PDFChunk。

    若图片无字节数据且无描述，返回 None（跳过）。
    """
    # 优先用已有描述（OCR 流水线中可能已预填），否则调用视觉模型
    description = block.image_description
    image_bytes = block.image_bytes

    if not description and image_bytes:
        context_hint = " > ".join(block.section_path) if block.section_path else ""
        description = analyze_image(image_bytes, context_hint=context_hint)

    if not description:
        logger.debug(
            f"[ImageChunker] 第 {block.page_num} 页图片跳过（无数据无描述）"
        )
        return None

    # 构造 chunk 文本：在描述前加语境说明，提升向量检索效果
    section_str = " > ".join(block.section_path) if block.section_path else ""
    text = (
        f"[图表/图片内容描述]\n"
        f"位置：第 {block.page_num} 页"
        + (f"，所属章节：{section_str}" if section_str else "")
        + f"\n\n{description}"
    )

    image_id = f"img_p{block.page_num}_{id(block) % 10000}"

    metadata = ChunkMetadata(
        doc_id=doc_id,
        file_name=file_name,
        file_path=file_path,
        page_num=block.page_num,
        section_path=block.section_path,
        section_str=section_str,
        chunk_type="image_summary",
        block_type="image",
        image_id=image_id,
        image_description=description,
        chunk_index=chunk_index,
        char_count=len(text),
        pdf_type=pdf_type,
    )

    logger.debug(
        f"[ImageChunker] 第 {block.page_num} 页图片 → "
        f"描述 {len(description)} 字"
    )
    return PDFChunk(text=text, metadata=metadata)
