"""
Layer 1 — 视觉理解抽取器

对无法用文字描述的元素（图表、流程图、复杂表格截图、扫描插图）调用
多模态视觉 LLM，生成自然语言描述，作为该元素的文本表示存入 chunk。

配置读取：
  - settings.pdf_vision_llm → model_config.yaml 中 llm_models.pdf_vision
  - 若未配置，gracefully 降级：直接输出 "[图表，暂未配置视觉模型]"

接口兼容 OpenAI vision API（base64 图片 + text prompt），
适用于 Qwen-VL、GPT-4o、InternVL 等任何 OpenAI-compatible 多模态模型。
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 视觉理解提示词
_VISION_PROMPT_ZH = (
    "请仔细观察这张图片，它来自一份 PDF 文档。\n"
    "请用中文详细描述图片中的内容，包括：\n"
    "1. 如果是图表（折线图/柱状图/饼图等）：描述坐标轴、数据趋势和关键数据点\n"
    "2. 如果是流程图/架构图：描述节点关系和流程步骤\n"
    "3. 如果是表格：描述表格结构、列名和主要数据\n"
    "4. 如果是普通图片/截图：描述其主要内容和与文档的关联\n"
    "请直接给出描述，不要加前缀说明。"
)


def analyze_image(
    image_bytes: bytes,
    context_hint: str = "",
) -> str:
    """
    对图片字节调用视觉模型，返回自然语言描述。

    Args:
        image_bytes:  PNG/JPEG 图片字节
        context_hint: 上下文提示（如所在章节名），帮助模型理解语境

    Returns:
        图片的自然语言描述字符串
    """
    if not image_bytes:
        return "[空图片]"

    vision_conf = _get_vision_config()
    if vision_conf is None:
        logger.warning("[VisionExtractor] 未配置视觉模型 (pdf_vision)，跳过图片理解")
        return "[图表/图片 — 请配置 pdf_vision 模型后重新入库以获得描述]"

    try:
        return _call_vision_api(image_bytes, context_hint, vision_conf)
    except Exception as e:
        logger.error(f"[VisionExtractor] 视觉模型调用失败: {e}")
        return f"[图表/图片描述生成失败: {str(e)[:100]}]"


def _get_vision_config() -> Optional[dict]:
    """
    读取视觉模型配置。
    model_config.yaml 中配置示例：
      llm_models:
        pdf_vision: qwen_vl   # 指向 providers 下的某个 key
    """
    try:
        from configs.settings import settings
        conf = settings.pdf_vision_llm
        if conf:
            logger.debug(f"[VisionExtractor] 使用视觉模型: {conf.get('model_id')}")
        return conf
    except Exception as e:
        logger.debug(f"[VisionExtractor] 读取视觉模型配置失败: {e}")
        return None


def _call_vision_api(
    image_bytes: bytes,
    context_hint: str,
    vision_conf: dict,
) -> str:
    """
    调用 OpenAI-compatible 视觉 API。
    支持任何实现了 /v1/chat/completions 且接受 image_url 的多模态模型。
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai: pip install openai")

    client = OpenAI(
        api_key=vision_conf["api_key"],
        base_url=vision_conf.get("base_url"),
    )

    # base64 编码图片
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    prompt = _VISION_PROMPT_ZH
    if context_hint:
        prompt = f"（上下文：该图片所在章节为「{context_hint}」）\n\n" + prompt

    response = client.chat.completions.create(
        model=vision_conf["model_id"],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_tokens=512,
        temperature=0.1,
    )

    description = response.choices[0].message.content.strip()
    logger.debug(f"[VisionExtractor] 描述生成成功 ({len(description)} 字)")
    return description
