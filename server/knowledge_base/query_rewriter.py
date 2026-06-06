"""
Query Rewriting 五种策略实现

策略一览：
  Strategy 1 — Normalize     : 大模型规范用户表达（口语→专业，纠错，补齐省略语）
  Strategy 2 — MultiAngle    : 多维度改写 + RRF 合并（RAG-Fusion 思路）
  Strategy 3 — Decompose     : 复杂问题分解为子问题，分别检索后合并
  Strategy 4 — HyDE          : 假设答案嵌入（用"可能的答案"做向量检索，而非原始问题）
  Strategy 5 — MacroContext   : 问题宏观化（提炼背景意图，扩展为更宏观的检索锚点）

每个策略都是异步函数，统一接口：
    async def rewrite_xxx(query: str) -> RewriteResult
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List

from openai import AsyncOpenAI
from configs.settings import settings

logger = logging.getLogger(__name__)

# =====================================================================
# 统一输出格式
# =====================================================================

@dataclass
class RewriteResult:
    """改写策略的输出结果"""
    strategy: str                               # 策略名称
    original_query: str                         # 原始问题
    primary_query: str                          # 主要检索用 query
    auxiliary_queries: List[str] = field(default_factory=list)  # 辅助 queries（多路检索用）
    hypothesis: str = ""                        # HyDE 生成的假设答案
    sub_questions: List[str] = field(default_factory=list)      # 分解出的子问题
    reasoning: str = ""                         # 改写理由（可选，便于调试）


# =====================================================================
# 底层 LLM 调用
# =====================================================================

_rewriter_client: AsyncOpenAI | None = None


def _get_rewriter_client() -> AsyncOpenAI:
    global _rewriter_client
    if _rewriter_client is None:
        conf = settings.rag_retrieve  # 用 rag_retrieve 级别的轻量模型
        _rewriter_client = AsyncOpenAI(
            api_key=conf["api_key"],
            base_url=conf.get("base_url"),
            timeout=20.0,
        )
    return _rewriter_client


async def _llm_call(prompt: str, max_tokens: int = 400) -> str:
    """调用改写专用 LLM（轻量快速）"""
    conf = settings.rag_retrieve
    client = _get_rewriter_client()
    try:
        resp = await client.chat.completions.create(
            model=conf["model_id"],
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的查询改写助手，输出简洁、结构化，严格按格式要求回复。"
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"[QueryRewriter] LLM 调用失败: {e}")
        return ""


# =====================================================================
# Strategy 1 — Normalize（规范化）
# =====================================================================

async def rewrite_normalize(query: str) -> RewriteResult:
    """
    策略一：大模型规范用户表达。

    解决问题：
    - 口语化、方言、错别字（"咋弄" → "如何操作"）
    - 缩写/俚语展开（"ROI" → "投资回报率"）
    - 代词消歧（"它是怎么规定的" → "合同第X条是怎么规定的"）
    - 补全省略的主语/宾语

    适用：任何查询，作为第一步预处理（开销极小）。
    """
    prompt = f"""请对以下用户查询进行规范化处理：
1. 纠正错别字和口语化表达
2. 展开缩写和专业术语
3. 补全省略的主语或关键词
4. 保持问题原意，不要过度改写

【原始查询】
{query}

请直接输出规范化后的查询（一行，不加任何前缀或解释）："""

    result = await _llm_call(prompt, max_tokens=100)
    normalized = result.strip() if result else query

    # 极端兜底：如果规范化结果比原始更长两倍，可能跑偏，退回原始
    if len(normalized) > len(query) * 2.5:
        normalized = query

    logger.debug(f"[Normalize] '{query}' → '{normalized}'")
    return RewriteResult(
        strategy="normalize",
        original_query=query,
        primary_query=normalized,
        reasoning=f"规范化: '{query}' → '{normalized}'",
    )


# =====================================================================
# Strategy 2 — MultiAngle（多维度改写）
# =====================================================================

async def rewrite_multi_angle(query: str, n_variants: int = 3) -> RewriteResult:
    """
    策略二：多维度改写 + RRF 合并（RAG-Fusion 核心）。

    思路：同一个问题可以从不同角度、用不同措辞检索，
    多路召回后用 Reciprocal Rank Fusion 合并，覆盖更多相关文档。

    示例：
      原始: "合同违约怎么办"
      变体1: "合同违约的法律责任"
      变体2: "违约金条款的计算方式"
      变体3: "甲方违约乙方的救济途径"

    适用：问题语义模糊、可从多角度解读、检索覆盖率不足的情况。
    """
    prompt = f"""请从 {n_variants} 个不同维度改写以下查询，每个改写角度不同、措辞不同，但都围绕同一核心问题。

【原始查询】
{query}

输出格式（每行一个，不加序号和前缀）：
改写1
改写2
改写3"""

    result = await _llm_call(prompt, max_tokens=200)

    variants = []
    for line in result.strip().split("\n"):
        line = line.strip()
        # 过滤掉序号和空行
        import re
        line = re.sub(r'^[\d\.\-\*、【】]+\s*', '', line).strip()
        if line and line != query and len(line) > 3:
            variants.append(line)

    variants = variants[:n_variants]

    # 如果 LLM 没有输出足够变体，用原始 query 补位
    if not variants:
        variants = [query]

    logger.debug(f"[MultiAngle] '{query}' → {len(variants)} 变体")
    return RewriteResult(
        strategy="multi_angle",
        original_query=query,
        primary_query=query,               # 原始 query 作为主路
        auxiliary_queries=variants,         # 变体作为辅助路
        reasoning=f"生成 {len(variants)} 个多角度变体，将进行 {len(variants)+1} 路并行检索",
    )


# =====================================================================
# Strategy 3 — Decompose（复杂问题分解）
# =====================================================================

async def rewrite_decompose(query: str, max_sub: int = 4) -> RewriteResult:
    """
    策略三：复杂问题分解为独立子问题，分别检索，各个击破。

    适用场景：
    - 多条件问题（"A 且 B 且 C 的情况下..."）
    - 比较性问题（"A 和 B 有什么区别，分别怎么应用"）
    - 多步推理问题（需要先知道 A 才能回答 B）

    示例：
      原始: "在合同履行过程中，如果甲方违约且涉及知识产权纠纷，应该走哪个仲裁流程？"
      子问题1: "合同履行中甲方违约的处理流程"
      子问题2: "知识产权纠纷的仲裁流程"
      子问题3: "同时涉及违约和知识产权的纠纷如何合并处理"
    """
    prompt = f"""以下问题较为复杂，请将其分解为 2-{max_sub} 个独立的子问题，每个子问题可以独立检索回答。

【复杂问题】
{query}

要求：
- 每个子问题必须独立、完整、可单独检索
- 子问题之间不要重复
- 保持与原始问题的逻辑关联

输出格式（每行一个子问题，不加序号和前缀）：
子问题1
子问题2
..."""

    result = await _llm_call(prompt, max_tokens=300)

    import re
    sub_questions = []
    for line in result.strip().split("\n"):
        line = re.sub(r'^[\d\.\-\*、【】子问题]+\s*', '', line).strip()
        if line and len(line) > 5 and line != query:
            sub_questions.append(line)

    sub_questions = sub_questions[:max_sub]

    if not sub_questions:
        # 分解失败，退回原始问题
        sub_questions = [query]

    logger.debug(f"[Decompose] '{query[:30]}...' → {len(sub_questions)} 子问题")
    return RewriteResult(
        strategy="decompose",
        original_query=query,
        primary_query=query,
        sub_questions=sub_questions,
        auxiliary_queries=sub_questions,    # 子问题也作为辅助检索路
        reasoning=f"分解为 {len(sub_questions)} 个子问题独立检索",
    )


# =====================================================================
# Strategy 4 — HyDE（假设文档嵌入）
# =====================================================================

async def rewrite_hyde(query: str) -> RewriteResult:
    """
    策略四：Hypothetical Document Embedding（假设文档嵌入）。

    原理：
      用户问题的向量 ≠ 答案文档的向量（问题和答案在语义空间中往往不对齐）。
      HyDE 让 LLM 先生成一个"可能的答案文档"，用该文档的向量去检索，
      与真实文档的向量更接近，检索精度大幅提升。

    适用场景：
    - 事实型问题（有明确答案的）
    - 文档中有标准描述的内容
    - 问题措辞与文档措辞差异大

    注意：HyDE 依赖 LLM 对领域的理解，若 LLM 完全不了解该领域，
    可能生成错误假设，建议与 Normalize 组合使用。
    """
    prompt = f"""请为以下问题生成一段简短的假设性回答（就像这个问题的答案在文档中可能的样子）。

【问题】
{query}

要求：
- 用正式文档语言写作（而非对话语气）
- 100-150字以内
- 包含可能存在于文档中的专业术语
- 即使不确定也要写出最有可能的内容（这只是用于向量检索，不会直接展示给用户）

请直接输出假设性回答："""

    hypothesis = await _llm_call(prompt, max_tokens=200)

    if not hypothesis:
        hypothesis = query  # 降级：直接用原始问题

    logger.debug(f"[HyDE] 生成假设答案 ({len(hypothesis)} 字)")
    return RewriteResult(
        strategy="hyde",
        original_query=query,
        primary_query=hypothesis,          # 用假设答案作为主要检索向量
        auxiliary_queries=[query],          # 原始 query 作为辅助（防止偏移过大）
        hypothesis=hypothesis,
        reasoning=f"生成假设答案用于向量检索，原始 query 作为辅助路",
    )


# =====================================================================
# Strategy 5 — MacroContext（问题宏观化）
# =====================================================================

async def rewrite_macro_context(query: str) -> RewriteResult:
    """
    策略五：将问题宏观化，提炼背景意图，扩展为更广泛的检索锚点。

    思路：用户问的是具体问题，但回答往往需要大背景/宏观框架支撑。
    宏观化后可以召回提供"大局观"的父节点和章节概述，再结合具体片段回答。

    示例：
      原始: "第三条第二款的违约金怎么计算"
      宏观化: "合同违约责任的整体框架和计算原则"
      →  能召回合同违约责任章节的概述节点（父节点），提供背景

    适用：
    - 需要理解背景才能回答的问题
    - 条款细节问题（需要知道条款所在的章节逻辑）
    - 策略/原则类问题（不仅要具体，还要宏观）
    """
    prompt = f"""请将以下具体问题提炼为更宏观的背景检索语句，用于召回背景信息和整体框架。

【具体问题】
{query}

要求：
- 提炼出这个问题所属的更大主题/领域
- 宏观化后应该能召回该主题的总览、背景、框架类文档
- 一句话，不超过40字

请直接输出宏观化后的检索语句（不加任何前缀）："""

    macro_query = await _llm_call(prompt, max_tokens=80)

    if not macro_query or len(macro_query) < 5:
        macro_query = query

    logger.debug(f"[MacroContext] '{query[:30]}' → '{macro_query}'")
    return RewriteResult(
        strategy="macro_context",
        original_query=query,
        primary_query=query,                # 原始 query 作为主路（具体检索）
        auxiliary_queries=[macro_query],    # 宏观 query 作为辅助路（背景检索）
        reasoning=f"宏观化: '{macro_query}'",
    )
