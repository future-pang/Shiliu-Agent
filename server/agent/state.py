import operator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, List, Dict, Any, Optional

# =====================================================================
# DAG 调度器与全局核心结构
# =====================================================================
class TaskPlan(TypedDict, total=False):
    """
    DAG 任务节点定义：记录单个子任务的完整生命周期
    """
    step: int               # 步骤
    description: str        # 任务描述
    agent: str              # 分配的执行器(sql、mcp、rag...)
    args: Dict[str, Any]    # Planner提取的参数字典
    depends_on: List[int]   # 依赖的前置步骤ID列表
    status: str             # "pending"(待处理), "running"(执行中), "done"(已完成), "error"(出错)
    result: Any

def merge_tasks(left: Dict[int, TaskPlan], right: Dict[int, TaskPlan]) -> Dict[int, TaskPlan]:
    """
    任务池聚合函数 ：
    允许 Task Fetcher 和各个 Executor 并行/循环地更新任务的状态和结果，而不会互相覆盖。
    相同 step_id → 用 .update() 局部更新（保留其他字段，只覆盖传入的键）
    不同 step_id → 直接新增到字典中
    """
    if not left:
        return right if right else {}

    merged = left.copy()
    if right:
        for step_id, task_update in right.items():
            if step_id in merged:
                merged[step_id].update(task_update)
            else:
                merged[step_id] = task_update
    return merged

def merge_memory(left: List[dict], right: List[dict]) -> List[dict]:
    left = left or []
    right = right or []
    merged = left + right
    return merged[-20:]

class AgentState(TypedDict):
    messages: Annotated[List[dict], merge_memory]
    long_term_profile: Dict[str, Any]

    current_context: Annotated[Dict[str, Any], operator.ior]
    user_query: str
    session_id: str
    enable_web_search: bool
    is_chat_only: bool
    tasks: Annotated[Dict[int, TaskPlan], merge_tasks]
    final_answer: str

def merge_unique_list(left: List[str], right: List[str]) -> List[str]:
    """去重合并，防止已读列表无限膨胀"""
    return list(set((left or []) + (right or [])))

# =====================================================================
# RAG 专属内部状态
# =====================================================================
class RagInternalState(TypedDict):
    """
    [RAG 私有状态]
    只有 RAG 子图内部的节点（检索、判官等）能看到和修改这些变量。
    """
    # ===== 继承自外层的任务信息  =====
    task_id: int        # 当前正在处理的任务 ID (关联回全局的 tasks)
    query: str          # Planner 规划出的具体搜索词或描述
    query_type: str     # Planner 判定的检索分辨率 (macro/micro/mixed)
    enable_web_search: bool

    has_deep_read: bool

    # ==== RAG 循环控制参数 ====
    rag_loop_count: int

    # ==== RAG 检索数据池 ====
    visited_nodes: Annotated[List[str], merge_unique_list]  # Context Tracker
    rag_context: Annotated[List[dict], operator.add]        # 判官通过的有效片段
    rag_trajectory: Annotated[List[dict], operator.add]     # 工具调用流水
    rag_judgement_passed: bool                              # 判官是否通过 (控制路由)
    rag_judgement_reason: str                               # 判官打回理由 (用于反思)
    rag_missing_entity: str                                 # 用于存放判官提取的缺失实体，触发系统级图谱跃迁

    # ==== LLM 最终裁决 ====
    is_knowledge_sufficient: bool  # LLM 认为知识是否足够（未使用）

    # ==== 子图的最终回传给主图 ====
    rag_final_result: str   # 可以是完美答案，也可以是 force_prompt 的降级总结