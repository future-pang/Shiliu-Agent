"""
Layer 2 数据模型 — PDF 结构化中间表示

所有层之间传递的标准数据结构，Pydantic 强类型约束。
"""
from __future__ import annotations

import uuid
from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field


# =====================================================================
# Layer 1 输出：原始解析块
# =====================================================================

class RawBlock(BaseModel):
    """Layer 1 抽取的原始块（文本/表格/图片三类）"""
    block_type: Literal["text", "table", "image"]
    page_num: int

    # 文本块字段
    text: Optional[str] = None
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    is_bold: Optional[bool] = None

    # 表格块字段（二维数组，行列形式）
    table_data: Optional[List[List[Optional[str]]]] = None

    # 图片块字段
    image_bytes: Optional[bytes] = None   # PIL image bytes (PNG)
    image_description: Optional[str] = None  # 视觉模型生成的描述

    # 通用坐标 (x0, y0, x1, y1)，相对页面左上角
    bbox: Optional[tuple] = None

    class Config:
        arbitrary_types_allowed = True


class RawPage(BaseModel):
    """Layer 1 输出：单页原始抽取结果"""
    page_num: int           # 1-indexed
    width: float
    height: float
    blocks: List[RawBlock] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


# =====================================================================
# Layer 2 输出：结构化文本块
# =====================================================================

class TextBlock(BaseModel):
    """Layer 2 输出：经过结构还原后的语义块"""
    block_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    page_num: int
    block_type: Literal[
        "heading", "paragraph", "table", "image",
        "footer", "header", "footnote", "caption"
    ]
    level: Optional[int] = None           # 标题层级：1/2/3/4
    text: str = ""
    table_data: Optional[List[List[Optional[str]]]] = None
    image_bytes: Optional[bytes] = None
    image_description: Optional[str] = None
    bbox: Optional[tuple] = None

    # 溯源核心：当前块所在的章节路径
    section_path: List[str] = Field(default_factory=list)
    # e.g. ["第一章 总则", "第二节 定义", "第三条"]

    class Config:
        arbitrary_types_allowed = True


class StructuredPage(BaseModel):
    """Layer 2 输出：单页结构化结果"""
    page_num: int
    blocks: List[TextBlock] = Field(default_factory=list)


class StructuredDocument(BaseModel):
    """Layer 2 输出：整份文档结构化结果（传给 Layer 3）"""
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_name: str
    file_path: str
    total_pages: int
    pdf_type: Literal["native", "scanned", "mixed"]
    pages: List[StructuredPage] = Field(default_factory=list)
    toc: List[dict] = Field(default_factory=list)  # 目录结构，可选


# =====================================================================
# Layer 3 输出：带 metadata 的 Chunk
# =====================================================================

class ChunkMetadata(BaseModel):
    """每个 Chunk 携带的完整溯源元数据"""
    doc_id: str
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_name: str
    file_path: str
    page_num: int
    section_path: List[str] = Field(default_factory=list)
    section_str: str = ""          # section_path 拼接字符串，便于展示

    chunk_type: Literal["text", "table", "image_summary"]
    block_type: str = "paragraph"
    heading_level: Optional[int] = None

    # 表格专属
    table_id: Optional[str] = None
    table_header: Optional[List[str]] = None

    # 图片专属
    image_id: Optional[str] = None
    image_description: Optional[str] = None

    chunk_index: int = 0
    total_chunks: int = 0          # 同文档 chunk 总数（后填充）
    char_count: int = 0

    pdf_type: str = "native"       # 原始 PDF 类型，便于溯源


class PDFChunk(BaseModel):
    """Layer 3 输出：最终入库的完整 Chunk"""
    text: str                      # 送去 embedding 的文本
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None

    class Config:
        arbitrary_types_allowed = True
