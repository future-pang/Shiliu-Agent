"""
PDF RAG 四层串联主流水线

入口：PDFPipeline.run(file_path)
流程：
  [Layer 1] 类型检测 → 对应抽取器（native/ocr/mixed）
  [Layer 2] 结构还原引擎
  [Layer 3] 三路切分器（文本 + 表格 + 图片）+ 索引器
  [结果]   返回入库统计信息

独立运行（CLI 模式）：
  python -m server.pdf_rag.pipeline --file path/to/doc.pdf
  python -m server.pdf_rag.pipeline --dir  path/to/pdf_folder/
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """流水线执行结果摘要"""
    file_name: str
    doc_id: str
    pdf_type: str
    total_pages: int
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    total_chunks: int = 0
    success: bool = True
    error: Optional[str] = None

    @property
    def summary(self) -> str:
        if not self.success:
            return f"❌ {self.file_name}: 处理失败 — {self.error}"
        return (
            f"✅ {self.file_name}\n"
            f"   类型: {self.pdf_type} | 页数: {self.total_pages}\n"
            f"   Chunk: {self.total_chunks} 个 "
            f"（文本 {self.text_chunks} + 表格 {self.table_chunks} + 图表 {self.image_chunks}）\n"
            f"   doc_id: {self.doc_id}"
        )


class PDFPipeline:
    """PDF 处理四层流水线"""

    def __init__(self, force_reindex: bool = False):
        self.force_reindex = force_reindex

    def run(self, file_path: str | Path) -> PipelineResult:
        """
        同步入口（内部跑 asyncio）。
        适合在脚本或 FastAPI 后台任务中调用。
        """
        return asyncio.run(self.run_async(file_path))

    async def run_async(self, file_path: str | Path) -> PipelineResult:
        """
        异步入口，四层串联执行。
        """
        file_path = Path(file_path)
        doc_id = str(uuid.uuid4())
        result = PipelineResult(
            file_name=file_path.name,
            doc_id=doc_id,
            pdf_type="unknown",
            total_pages=0,
        )

        logger.info(f"\n{'=' * 60}")
        logger.info(f"PDF 处理流水线启动: {file_path.name}")
        logger.info(f"{'=' * 60}")

        try:
            # ============================================================
            # Layer 1 — 解析层
            # ============================================================
            logger.info(f"\n[Layer 1] 解析层 — 检测 PDF 类型...")
            raw_pages, pdf_type = await self._layer1_parse(file_path)
            result.pdf_type = pdf_type
            result.total_pages = max((p.page_num for p in raw_pages), default=0)
            logger.info(
                f"[Layer 1] 完成: 类型={pdf_type}, {result.total_pages} 页, "
                f"{sum(len(p.blocks) for p in raw_pages)} 个原始块"
            )

            # ============================================================
            # Layer 2 — 结构还原层
            # ============================================================
            logger.info(f"\n[Layer 2] 结构还原层 — 重建文档语义结构...")
            from server.pdf_rag.layer2_structure.structure_builder import build_structure
            structured_doc = build_structure(
                raw_pages=raw_pages,
                file_name=file_path.name,
                file_path=str(file_path),
                doc_id=doc_id,
                pdf_type=pdf_type,
            )
            logger.info(
                f"[Layer 2] 完成: {len(structured_doc.toc)} 个标题节点"
            )

            # ============================================================
            # Layer 3 — 切片与索引层
            # ============================================================
            logger.info(f"\n[Layer 3] 切片层 — 语义切分 + 结构化处理...")
            all_chunks = await self._layer3_chunk(structured_doc)

            text_chunks = [c for c in all_chunks if c.metadata.chunk_type == "text"]
            table_chunks = [c for c in all_chunks if c.metadata.chunk_type == "table"]
            image_chunks = [c for c in all_chunks if c.metadata.chunk_type == "image_summary"]

            result.text_chunks = len(text_chunks)
            result.table_chunks = len(table_chunks)
            result.image_chunks = len(image_chunks)
            result.total_chunks = len(all_chunks)

            logger.info(
                f"[Layer 3] 切分完成: {len(all_chunks)} 个 chunk "
                f"（文本={len(text_chunks)}, 表格={len(table_chunks)}, "
                f"图表={len(image_chunks)}）"
            )

            # ---- 索引 ---------------------------------------------------
            logger.info(f"\n[Layer 3] 索引层 — 向量化 + 写入 ChromaDB...")
            from server.pdf_rag.layer3_chunking.indexer import index_chunks
            indexed_count = index_chunks(
                chunks=all_chunks,
                doc_id=doc_id,
                file_path=str(file_path),
                force_reindex=self.force_reindex,
            )

            if indexed_count == 0 and not self.force_reindex:
                logger.info("[Layer 3] 文件已入库，跳过（使用 force_reindex=True 强制重建）")

            logger.info(f"\n{'=' * 60}")
            logger.info(f"流水线完成: {file_path.name}")
            logger.info(result.summary)

        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"流水线异常: {e}", exc_info=True)

        return result

    # =====================================================================
    # 私有方法
    # =====================================================================

    async def _layer1_parse(self, file_path: Path):
        """
        Layer 1：根据 PDF 类型调用对应抽取器。
        - native  → native_extractor（全页）
        - scanned → ocr_extractor（全页）
        - mixed   → native_extractor（文本页）+ ocr_extractor（图片页）
                    + vision_extractor 处理图片块
        """
        from server.pdf_rag.layer1_parser.detector import detect_pdf_type
        from server.pdf_rag.layer1_parser.native_extractor import extract_native_pages
        from server.pdf_rag.layer1_parser.ocr_extractor import extract_ocr_pages
        from server.pdf_rag.layer1_parser.vision_extractor import analyze_image

        pdf_type, text_pages, image_pages = detect_pdf_type(file_path)

        raw_pages = []

        if pdf_type == "native":
            raw_pages = extract_native_pages(file_path)

        elif pdf_type == "scanned":
            raw_pages = extract_ocr_pages(file_path)

        elif pdf_type == "mixed":
            # 文本页走 native，图片页走 OCR
            if text_pages:
                native_pages = extract_native_pages(file_path, target_pages=text_pages)
                raw_pages.extend(native_pages)
            if image_pages:
                ocr_pages = extract_ocr_pages(file_path, target_pages=image_pages)
                raw_pages.extend(ocr_pages)

            # 对 native 页中的图片块，异步调用视觉模型
            await self._enrich_images_with_vision(raw_pages)

            # 按页码排序
            raw_pages.sort(key=lambda p: p.page_num)

        else:
            # 兜底：全走 native
            raw_pages = extract_native_pages(file_path)

        return raw_pages, pdf_type

    async def _enrich_images_with_vision(self, raw_pages):
        """
        并发调用视觉模型，为每个图片块生成描述（异步执行不阻塞）。
        """
        from server.pdf_rag.layer1_parser.vision_extractor import analyze_image

        async def process_image_block(block):
            if block.block_type == "image" and block.image_bytes:
                loop = asyncio.get_event_loop()
                # 视觉 API 调用是 IO 密集型，用 executor 跑
                description = await loop.run_in_executor(
                    None,
                    lambda: analyze_image(block.image_bytes, context_hint="")
                )
                block.image_description = description

        tasks = []
        for page in raw_pages:
            for block in page.blocks:
                if block.block_type == "image":
                    tasks.append(process_image_block(block))

        if tasks:
            logger.info(f"[Layer 1] 并发处理 {len(tasks)} 个图片块（视觉模型）...")
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _layer3_chunk(self, structured_doc):
        """Layer 3：三路并行切分"""
        from server.pdf_rag.layer3_chunking.semantic_chunker import chunk_text_blocks
        from server.pdf_rag.layer3_chunking.table_chunker import chunk_table
        from server.pdf_rag.layer3_chunking.image_chunker import chunk_image

        all_chunks = []

        # ---- 文本切分 -----------------------------------------------
        text_chunks = chunk_text_blocks(structured_doc, chunk_index_start=0)
        all_chunks.extend(text_chunks)

        # ---- 表格/图片切分 ------------------------------------------
        table_idx = len(all_chunks)
        for page in structured_doc.pages:
            for block in page.blocks:
                if block.block_type == "table":
                    t_chunks = chunk_table(
                        block=block,
                        doc_id=structured_doc.doc_id,
                        file_name=structured_doc.file_name,
                        file_path=structured_doc.file_path,
                        pdf_type=structured_doc.pdf_type,
                        chunk_index_start=table_idx,
                    )
                    all_chunks.extend(t_chunks)
                    table_idx += len(t_chunks)

                elif block.block_type == "image":
                    i_chunk = chunk_image(
                        block=block,
                        doc_id=structured_doc.doc_id,
                        file_name=structured_doc.file_name,
                        file_path=structured_doc.file_path,
                        pdf_type=structured_doc.pdf_type,
                        chunk_index=table_idx,
                    )
                    if i_chunk:
                        all_chunks.append(i_chunk)
                        table_idx += 1

        return all_chunks


# =====================================================================
# 便捷函数：扫描目录入库所有 PDF
# =====================================================================

async def ingest_pdf_directory_async(
    directory: str | Path,
    force_reindex: bool = False,
) -> List[PipelineResult]:
    """扫描目录，对所有 PDF 文件异步串行入库（避免内存爆炸）"""
    directory = Path(directory)
    pdf_files = list(directory.rglob("*.pdf")) + list(directory.rglob("*.PDF"))

    if not pdf_files:
        logger.info(f"目录 {directory} 中没有 PDF 文件")
        return []

    logger.info(f"发现 {len(pdf_files)} 个 PDF 文件，开始入库...")
    pipeline = PDFPipeline(force_reindex=force_reindex)
    results = []

    for pdf_file in pdf_files:
        result = await pipeline.run_async(pdf_file)
        results.append(result)

    # 打印汇总
    success = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"批量入库完成: {len(success)} 成功 / {len(failed)} 失败")
    for r in failed:
        logger.error(f"  失败: {r.summary}")

    return results


# =====================================================================
# CLI 入口
# =====================================================================

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="PDF RAG 入库工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="单个 PDF 文件路径")
    group.add_argument("--dir",  type=str, help="批量扫描目录")
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新入库（忽略去重检查）"
    )
    args = parser.parse_args()

    if args.file:
        pipeline = PDFPipeline(force_reindex=args.force)
        result = pipeline.run(args.file)
        print(result.summary)
    else:
        results = asyncio.run(
            ingest_pdf_directory_async(args.dir, force_reindex=args.force)
        )
        for r in results:
            print(r.summary)
