import asyncio
from typing import Dict, Any, List, Literal
from langgraph.graph import StateGraph, END, START
from server.agent.state import AgentState, RagInternalState, TaskPlan
from langchain_core.runnables import RunnableConfig

from server.agent.nodes.planner import planner_node
from server.agent.nodes.fetcher import task_fetcher_node, deadlock_resolver_node
from server.agent.nodes.sql_agent import sql_agent_node
from server.agent.nodes.mcp_agent import mcp_agent_node
from server.agent.nodes.synthesizer import synthesizer_node

from server.agent.nodes.rag_nodes import rag_retrieve_node, grader_node, force_prompt_node, graph_leap_node

# ====================================================
# RAG 专属的微型内循环子图
# ====================================================
def build_rag_subgraph():
    """
    构建 RAG 专属评估循环。它在主图中作为一个单独的 Node 运行。
    """
    rag_graph = StateGraph(RagInternalState)

    # ==== RAG ====
    rag_graph.add_node("rag_retrieve", rag_retrieve_node)
    rag_graph.add_node("grader", grader_node)
    rag_graph.add_node("force_prompt", force_prompt_node)
    rag_graph.add_node("graph_leap", graph_leap_node)
    rag_graph.add_edge(START, "rag_retrieve")
    rag_graph.add_edge("rag_retrieve", "grader")

    # ==== 检索评估中心 ====
    def check_grader_result(state: RagInternalState) -> Literal["retry", "leap", "force_degrade", "done"]:
        count = state.get("rag_loop_count", 0)
        judgement_passed = state.get("rag_judgement_passed", False)
        missing_entity = state.get("rag_missing_entity", "") # 提取判官给出的实体

        if judgement_passed:
            return "done"
        elif count >= 3:
            return "force_degrade"
        elif missing_entity:
            return "leap"  # ⚡ 核心分流：只要有确实实体，立刻走系统级跃迁！
        else:
            return "retry"  # incorrect，或者 ambiguous但没提取出实体，打回让Agent重搜

    rag_graph.add_conditional_edges(
        "grader",
        check_grader_result,
        {
            "retry": "rag_retrieve",
            "leap": "graph_leap",
            "force_degrade": "force_prompt",
            "done": END
        }
    )

    rag_graph.add_edge("graph_leap", "rag_retrieve")

    rag_graph.add_edge("force_prompt", END)

    return rag_graph.compile()

rag_subgraph_app = build_rag_subgraph()

# ====================================================
# RAG 的主图包装器
# ====================================================
async def rag_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    直接在 graph.py 里定义包装器。它能直接访问上面的 rag_subgraph_app
    """
    tasks: Dict[int, TaskPlan] = state.get("tasks", {})
    updates: Dict[int, TaskPlan] = {}

    async def run_single_rag(step_id: int, task: TaskPlan):
        print(f"[Wrapper] 正在将任务 {step_id} 委派给 RAG 子图...")
        query_type = task.get("args", {}).get("query_type", "mixed")
        initial_rag_state: RagInternalState = {
            "task_id": step_id,
            "query": task.get("description", ""),
            "query_type": query_type,
            "enable_web_search": state.get("enable_web_search", False),
            "rag_loop_count": 0,
            "visited_nodes": [],
            "rag_context": [],
            "rag_trajectory": [],
            "rag_judgement_passed": False,
            "rag_judgement_reason": "",
            "is_knowledge_sufficient": False,
            "rag_final_result": ""
        }
        try:
            final_rag_state = await rag_subgraph_app.ainvoke(initial_rag_state, config=config)
            rag_final_answer = final_rag_state.get("rag_final_result", "未找到结果")
        except Exception as e:
            rag_final_answer = f"知识库检索过程发生异常: {str(e)}"

        updated_task = task.copy()
        updated_task["status"] = "done"
        updated_task["result"] = rag_final_answer   # 写入检索结果
        return step_id, updated_task

    # ==== 筛选待执行的 RAG 任务 ====
    coros = []
    for step_id, task in tasks.items():
        if task.get("agent") == "rag_agent" and task.get("status") == "running":
            coros.append(run_single_rag(step_id, task))

    # ==== 并发执行所有 RAG 任务 ====
    if coros:
        results = await asyncio.gather(*coros)
        for step_id, updated_task in results:
            updates[step_id] = updated_task

    return {"tasks": updates}

# ====================================================
# LLM-Compiler 的总调度主图
# ====================================================

def route_from_planner(state: AgentState) -> str:
    """
    Planner 规划后的岔路口
    """
    if state.get("is_chat_only"):
        print("[Router] 纯闲聊，直接去统筹器。")
        return "synthesizer"
    return "task_fetcher"

def dispatch_tasks_from_fetcher(state: AgentState) -> list:
    """
    Task Fetcher 节点的核心路由函数。
    它不仅实现并发，还控制着 DAG 的流转方向。
    """

    tasks = state.get("tasks", {})

    # ==== 检查是否所有任务都已完成 ====
    if all(t.get("status") == "done" for t in tasks.values()):
        print("[Fetcher] 所有子任务完成，汇聚到 Synthesizer。")
        return ["synthesizer"]

    ready_agents = set()
    has_pending = False

    # ==== 扫描被 Task Fetcher 标记为 'running' 的任务 ====
    for t_id, task in tasks.items():
        if task.get("status") == "running":
            ready_agents.add(task["agent"])
        elif task.get("status") == "pending":
            has_pending = True

    # ==== 死锁检测机制 ====
    # 如果没有任务处于 running，但还有 pending，说明出现了循环依赖 (A等B，B等A)
    if not ready_agents and has_pending:
        print("[Fetcher] 检测到依赖死锁！触发熔断机制。")
        return ["deadlock_resolver"]


    print(f"[Fetcher] 绿灯放行，并行唤醒专家: {list(ready_agents)}")
    return list(ready_agents)

def build_agent(checkpointer=None):
    """
    构建最终带 LLM-Compiler 架构的 Agentic 系统
    """
    workflow = StateGraph(AgentState)

    # ==== 添加核心节点 ====
    workflow.add_node("planner", planner_node)
    workflow.add_node("task_fetcher", task_fetcher_node)
    workflow.add_node("deadlock_resolver", deadlock_resolver_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("mcp_agent", mcp_agent_node)

    # ==== 把 RAG 子图作为一个节点嵌入进来 ====
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("synthesizer", synthesizer_node)


    # ==== 编排边 ====
    workflow.add_edge(START, "planner")

    workflow.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "task_fetcher": "task_fetcher",
            "synthesizer": "synthesizer"
        }
    )

    # ==== Task Fetcher 的动态派发 ====
    workflow.add_conditional_edges(
        "task_fetcher",
        dispatch_tasks_from_fetcher,
        {
            "sql_agent": "sql_agent",
            "rag_agent": "rag_agent",
            "mcp_agent": "mcp_agent",
            "deadlock_resolver": "deadlock_resolver",
            "synthesizer": "synthesizer"
        }
    )

    workflow.add_edge("sql_agent", "task_fetcher")
    workflow.add_edge("rag_agent", "task_fetcher")
    workflow.add_edge("mcp_agent", "task_fetcher")
    workflow.add_edge("deadlock_resolver", "task_fetcher")
    workflow.add_edge("synthesizer", END)

    return workflow.compile(checkpointer=checkpointer)
