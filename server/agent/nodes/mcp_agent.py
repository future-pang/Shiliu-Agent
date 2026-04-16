import asyncio
from typing import Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from server.tools.mcp_tools import generate_image_tool

from configs.settings import settings
from configs.prompt_loader import get_prompt
from server.agent.llm_factory import get_tool_llm
from server.agent.state import AgentState, TaskPlan

from server.tools.mcp_tools import (
    # 天气工具
    weather_forecast_api,
    weather_api,
    travel_advice_api,
    astronomy_api,
    # 高德地图工具
    walking_plan_api,
    distance_api,
    around_search_api,
    static_map_api,
    # DIY 工具
    get_current_time,
    # 通用工具
    web_search
)

# =====================================================================
# 抽取单个任务的执行逻辑，供并发调用
# =====================================================================
async def _process_single_task_async(step_id: int, task: dict, enable_web_search: bool, config: dict) -> Tuple[int, dict]:
    description = task.get("description", "")
    args = task.get("args", {})

    active_tools = [
        weather_forecast_api,
        weather_api,
        travel_advice_api,
        astronomy_api,
        walking_plan_api,
        distance_api,
        around_search_api,
        static_map_api,
        get_current_time,
        generate_image_tool
    ]
    if enable_web_search:
        active_tools.append(web_search)

    system_prompt = get_prompt("mcp_agent")

    llm = get_tool_llm()

    mcp_app = create_react_agent(llm, tools=active_tools, prompt=system_prompt)

    query_str = f"【你的任务】\n{description}\n\n【已知参数字典】\n{args}"
    inputs = {"messages": [("user", query_str)]}
    updated_task = task.copy()

    try:
        print(f"[MCP Agent] 任务 {step_id} 开始执行 (异步并发)...")
        merged_config = {**config, "recursion_limit": 10}
        response_state = await mcp_app.ainvoke(inputs, config=merged_config)

        final_answer = response_state["messages"][-1].content
        print(f"[MCP Agent] 任务 {step_id} 执行完毕！")

        updated_task["status"] = "done"
        updated_task["result"] = final_answer
    except Exception as e:
        print(f"[MCP Agent] 任务 {step_id} 异常: {e}")
        updated_task["status"] = "done"
        updated_task["result"] = f"外部工具调用失败: {str(e)}"

    return step_id, updated_task

# =====================================================================
# MCP Agent 核心节点逻辑
# =====================================================================
async def mcp_agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    已优化：
    1. 使用单例 LLM（llm_factory），消除重复初始化
    2. 使用缓存 Prompt（prompt_loader），消除重复磁盘 I/O
    """
    print("\n[MCP Agent] 收到外部接口调用任务，准备组装工具箱并【异步并发执行】...")

    tasks: Dict[int, TaskPlan] = state.get("tasks", {})
    updates: Dict[int, TaskPlan] = {}
    enable_web_search = state.get("enable_web_search", False)

    tasks_to_run = []
    for step_id, task in tasks.items():
        if task.get("agent") == "mcp_agent" and task.get("status") == "running":
            tasks_to_run.append(_process_single_task_async(step_id, task, enable_web_search, config))

    if tasks_to_run:
        results = await asyncio.gather(*tasks_to_run)
        for step_id, updated_task in results:
            updates[step_id] = updated_task

    return {"tasks": updates}