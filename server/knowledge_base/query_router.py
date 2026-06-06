"""
Query Router — 智能路由与策略编排中心

核心职责：
  1. 接收用户原始 query
  2. 用轻量 LLM 分类 query 类型
  3. 根据类型选择最优的策略组合
  4. 协调多路检索并用 RRF 合并排序
  5. 返回统一的最终 chunk 列表（供 Agent 直接使用）

策略路由矩阵：
  ┌──────────────┬──────────────────────────────────────────────────────┐
  │ query 类型    │ 策略组合                                              │
  ├──────────────┼──────────────────────────────────────────────────────┤
  │ simple       │ Normalize → 单路检索                                  │
  │ factual      │ Normalize + HyDE → 双路检索（HyDE 为主）              │
  │ multi_angle  │ Normalize + MultiAngle → 4 路并行 + RRF              │
  │ complex      │ Normalize + Decompose → N 路子问题并行 + 去重合并      │
  │ background   │ Normalize + MacroContext + HyDE → 三路检索            │
  └──────────────┴──────────────────────────────────────────────────────┘

RRF（Reciprocal Rank Fusion）合并公式：
  score(chunk) = Σ 1 / (k + rank_in_list_i)    k=60（经验常数）
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple

from server.knowledge_base.query_rewriter import (
    RewriteResult,
    rewrite_normalize,
    rewrite_multi_angle,
    rewrite_decompose,
    rewrite_hyde,
    rewrite_macro_context,
)
from server.knowledge_base.query_rewriter import _llm_call

logger = logging.getLogger(__name__)

# RRF 平滑常数（越大对低排名的惩罚越轻）
RRF_K = 60
# 最终返回的 chunk 数上限
FINAL_TOP_N = 10

QueryType = Literal["simple", "factual", "multi_angle", "complex", "background"]


# =====================================================================
# Query 类型分类器
# =====================================================================

async def classify_query(query: str) -> QueryType:
    """
    用轻量 LLM 将 query 分类为 5 种类型之一。

    分类标准：
    - simple     : 短小、清晰、直接，一路检索即可
    - factual    : 问具体事实/数据/条款，有明确答案
    - multi_angle: 问题可从多个维度解读，单路容易遗漏
    - complex    : 包含多个子条件、比较、多步推理
    - background : 需要宏观背景才能回答，或问"框架/原则/概述"类
    """
    prompt = f"""请将以下查询分类为以下类型之一，直接输出类型名（英文小写），不要解释：

类型说明：
- simple     : 简单直接的问题，一路检索即可（如：公司地址是哪里？）
- factual    : 问具体事实/数字/条款内容（如：第三条规定了什么？年营收是多少？）
- multi_angle: 可从多个角度解读，单路检索容易遗漏（如：合同风险有哪些？）
- complex    : 包含多个子条件或需要多步推理（如：A和B哪个更适合，各有什么优缺点？）
- background : 需要宏观背景理解（如：整体策略是什么？框架是怎样的？）

【待分类查询】
{query}

输出（只输出一个类型名）："""

    result = await _llm_call(prompt, max_tokens=20)
    result = result.strip().lower()

    valid_types = {"simple", "factual", "multi_angle", "complex", "background"}
    if result in valid_types:
        logger.debug(f"[Router] 分类: '{query[:30]}' → {result}")
        return result  # type: ignore

    # 兜底逻辑：通过关键词简单判断
    return _fallback_classify(query)


def _fallback_classify(query: str) -> QueryType:
    """LLM 分类失败时的关键词兜底"""
    q = query.lower()

    # 比较/多条件 → complex
    if any(k in q for k in ["区别", "对比", "比较", "vs", "和…的关系", "优缺点", "哪个更"]):
        return "complex"
    # 框架/背景 → background
    if any(k in q for k in ["整体", "概述", "框架", "战略", "背景", "宏观", "思路", "原则"]):
        return "background"
    # 具体条款/数字 → factual
    if any(k in q for k in ["第几条", "条款", "规定", "多少", "几个", "何时", "哪年"]):
        return "factual"
    # 短 query → simple
    if len(query) < 15:
        return "simple"
    # 默认 → multi_angle
    return "multi_angle"


# =====================================================================
# 多路检索执行器
# =====================================================================

async def multi_path_retrieve(
    queries: List[str],
    handler,
    top_k: int = 8,
    query_type: str = "mixed",
) -> List[Tuple[str, List]]:
    """
    对多个 query 并发执行检索，返回 [(query, nodes_list), ...]。
    """
    async def retrieve_one(q: str) -> Tuple[str, List]:
        try:
            # handler.retrieve_with_rerank 是同步方法，用 executor 跑
            loop = asyncio.get_event_loop()
            nodes = await loop.run_in_executor(
                None,
                lambda: handler.retrieve_with_rerank(
                    query_str=q,
                    top_k=top_k,
                    rerank_top_n=top_k // 2,
                    query_type=query_type,
                )
            )
            return q, nodes
        except Exception as e:
            logger.warning(f"[Router] 检索失败 ('{q[:30]}'): {e}")
            return q, []

    results = await asyncio.gather(*[retrieve_one(q) for q in queries])
    return list(results)


# =====================================================================
# RRF 合并
# =====================================================================

def rrf_merge(
    ranked_lists: List[List],
    k: int = RRF_K,
    top_n: int = FINAL_TOP_N,
) -> List:
    """
    Reciprocal Rank Fusion 合并多路检索结果。

    每个 node 按其在各路中的排名计算 RRF 得分，最终合并排序。
    node_id → rrf_score
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    node_map: Dict[str, object] = {}

    for ranked_list in ranked_lists:
        for rank, node_with_score in enumerate(ranked_list):
            node_id = node_with_score.node_id
            rrf_scores[node_id] += 1.0 / (k + rank + 1)
            if node_id not in node_map:
                node_map[node_id] = node_with_score

    # 按 RRF 得分降序
    sorted_ids = sorted(rrf_scores, key=lambda nid: rrf_scores[nid], reverse=True)

    merged = []
    for nid in sorted_ids[:top_n]:
        node = node_map[nid]
        # 将 RRF 得分写入 node 的 score 字段（便于后续展示）
        node.score = rrf_scores[nid]
        merged.append(node)

    return merged


# =====================================================================
# 主路由入口
# =====================================================================

class QueryRouter:
    """
    智能查询路由器（单例使用）。
    自动判断 query 类型并选择最优改写 + 检索策略。
    """

    def __init__(self, handler, auto_classify: bool = True):
        """
        Args:
            handler:        EmeiKnowledgeBase 实例（提供检索能力）
            auto_classify:  True=自动分类，False=全走 multi_angle（保守模式）
        """
        self.handler = handler
        self.auto_classify = auto_classify

    async def route_and_retrieve(
        self,
        query: str,
        top_k: int = 8,
        query_type: str = "mixed",
        force_strategy: Optional[QueryType] = None,
    ) -> Tuple[List, RewriteResult, QueryType]:
        """
        完整流程：分类 → 改写 → 多路检索 → RRF 合并。

        Args:
            query:          用户原始问题
            top_k:          每路检索的 top_k
            query_type:     "micro"|"macro"|"mixed"（传给 handler 的粒度控制）
            force_strategy: 强制指定策略（跳过分类），用于调试

        Returns:
            (merged_nodes, rewrite_result, detected_type)
        """
        logger.info(f"\n[Router] 接收查询: '{query}'")

        # ---- Step 1: 规范化（所有策略的第一步）-----------------------
        normalize_result = await rewrite_normalize(query)
        normalized_query = normalize_result.primary_query
        logger.info(f"[Router] Step1 规范化: '{normalized_query}'")

        # ---- Step 2: 分类 -------------------------------------------
        if force_strategy:
            q_type: QueryType = force_strategy
        elif self.auto_classify:
            q_type = await classify_query(normalized_query)
        else:
            q_type = "multi_angle"

        logger.info(f"[Router] Step2 类型: {q_type}")

        # ---- Step 3: 按类型执行对应策略 ------------------------------
        rewrite_result, all_queries = await self._apply_strategy(
            normalized_query=normalized_query,
            q_type=q_type,
        )
        logger.info(
            f"[Router] Step3 改写完成: {len(all_queries)} 路查询"
        )

        # ---- Step 4: 多路并发检索 ------------------------------------
        path_results = await multi_path_retrieve(
            queries=all_queries,
            handler=self.handler,
            top_k=top_k,
            query_type=query_type,
        )

        # ---- Step 5: RRF 合并 ----------------------------------------
        ranked_lists = [nodes for _, nodes in path_results if nodes]

        if not ranked_lists:
            logger.warning("[Router] 所有检索路均无结果")
            return [], rewrite_result, q_type

        if len(ranked_lists) == 1:
            # 只有一路，无需 RRF
            merged = ranked_lists[0][:FINAL_TOP_N]
        else:
            merged = rrf_merge(ranked_lists, top_n=FINAL_TOP_N)

        logger.info(
            f"[Router] Step5 RRF 合并: {sum(len(r) for r in ranked_lists)} → {len(merged)} 节点"
        )

        return merged, rewrite_result, q_type

    async def _apply_strategy(
        self,
        normalized_query: str,
        q_type: QueryType,
    ) -> Tuple[RewriteResult, List[str]]:
        """
        根据 query 类型执行对应的改写策略，返回 (rewrite_result, all_queries)。

        策略矩阵（在这里调配）：
          simple      → 仅原始 query（无改写开销）
          factual     → HyDE 为主 + 原始 query 为辅
          multi_angle → MultiAngle 3 变体 + 原始 query（4 路）
          complex     → Decompose 子问题（每个子问题独立检索）
          background  → MacroContext + HyDE + 原始 query（3 路）
        """
        if q_type == "simple":
            # 简单问题：直接单路，省掉所有 LLM 开销
            return RewriteResult(
                strategy="simple",
                original_query=normalized_query,
                primary_query=normalized_query,
                reasoning="简单查询，直接单路检索"
            ), [normalized_query]

        elif q_type == "factual":
            # 事实型：HyDE 主路 + 原始 query 辅路
            hyde_result = await rewrite_hyde(normalized_query)
            all_queries = [hyde_result.primary_query, normalized_query]
            return hyde_result, all_queries

        elif q_type == "multi_angle":
            # 多角度：3 变体 + 原始 query = 4 路
            ma_result = await rewrite_multi_angle(normalized_query, n_variants=3)
            all_queries = [normalized_query] + ma_result.auxiliary_queries
            return ma_result, all_queries

        elif q_type == "complex":
            # 复杂分解：并发检索每个子问题
            decomp_result = await rewrite_decompose(normalized_query, max_sub=4)
            # 子问题 + 原始 query（确保整体也被检索）
            all_queries = decomp_result.sub_questions + [normalized_query]
            return decomp_result, all_queries

        elif q_type == "background":
            # 背景宏观：MacroContext + HyDE + 原始 query = 3 路
            macro_task = rewrite_macro_context(normalized_query)
            hyde_task = rewrite_hyde(normalized_query)
            macro_result, hyde_result = await asyncio.gather(macro_task, hyde_task)

            all_queries = [
                normalized_query,                           # 具体检索
                macro_result.auxiliary_queries[0],          # 宏观背景
                hyde_result.primary_query,                  # 假设答案
            ]
            # 合并 result（用 macro 为主）
            macro_result.auxiliary_queries = all_queries[1:]
            macro_result.hypothesis = hyde_result.hypothesis
            return macro_result, all_queries

        else:
            return RewriteResult(
                strategy="fallback",
                original_query=normalized_query,
                primary_query=normalized_query,
            ), [normalized_query]


# =====================================================================
# 全局单例工厂
# =====================================================================

_router_instance: QueryRouter | None = None


def get_router(auto_classify: bool = True) -> QueryRouter:
    """
    获取路由器单例（延迟绑定 handler，避免循环导入）。
    """
    global _router_instance
    if _router_instance is None:
        from server.knowledge_base.handler import kb_handler
        _router_instance = QueryRouter(
            handler=kb_handler,
            auto_classify=auto_classify,
        )
    return _router_instance
