from typing import Dict, Any
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from configs.settings import settings
from configs.prompt_loader import get_prompt       # ⚡ 缓存的 Prompt
from server.agent.llm_factory import get_sql_llm  # ⚡ 单例 LLM
from server.agent.state import AgentState, TaskPlan


# =====================================================================
# 定义 SQL 执行工具（测试）
# =====================================================================
@tool
def execute_sql_query(sql_query: str) -> str:
    """
    执行 SQL 查询语句，并返回数据库结果。
    请务必确保 SQL 语句符合 MySQL 语法。
    """
    print(f"      [SQL Tool] 正在执行底层的 SQL: {sql_query}")
    # TODO: 接入真实的 SQLAlchemy 数据库查询逻辑
    if "ticket" in sql_query.lower() or "price" in sql_query.lower():
        return "[MySQL 返回]: 旺季门票 160元，淡季门票 110元"
    elif "elevation" in sql_query.lower() or "海拔" in sql_query.lower():
        return "[MySQL 返回]: 金顶海拔 3079米，万佛顶海拔 3099米"
    else:
        return "[MySQL 返回]: 0 rows affected."


# =====================================================================
# SQL Agent 核心节点逻辑
# =====================================================================
async def sql_agent_node(state: AgentState) -> dict:
    """
    ⚡ 已优化：
    1. async + ainvoke，不再阻塞事件循环
    2. 使用单例 LLM（llm_factory），消除重复初始化
    3. 使用缓存 Prompt（prompt_loader），消除重复磁盘 I/O
    """
    print("\n[SQL Agent] 收到数据查询任务，开始生成并执行 SQL...")

    tasks: Dict[int, TaskPlan] = state.get("tasks", {})
    updates: Dict[int, TaskPlan] = {}

    for step_id, task in tasks.items():
        if task.get("agent") == "sql_agent" and task.get("status") == "running":

            description = task.get("description", "")
            args = task.get("args", {})

            # ⚡ 使用缓存 Prompt（0 磁盘 I/O）
            system_prompt = get_prompt("sql_agent")

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "【任务描述】\n{description}\n\n【提取的约束参数】\n{args}")
            ])

            # ⚡ 使用单例 LLM（0 初始化开销）
            llm = get_sql_llm()
            llm_with_tools = llm.bind_tools([execute_sql_query])
            chain = prompt | llm_with_tools

            try:
                # ⚡ 异步调用，不再阻塞事件循环
                ai_message = await chain.ainvoke({"description": description, "args": args})
                final_answer = ""

                if ai_message.tool_calls:
                    for tool_call in ai_message.tool_calls:
                        if tool_call["name"] == "execute_sql_query":
                            sql_str = tool_call["args"]["sql_query"]
                            db_result = execute_sql_query.invoke({"sql_query": sql_str})
                            final_answer = f"数据库查询结果: {db_result} \n(执行的SQL: {sql_str})"
                else:
                    final_answer = ai_message.content if ai_message.content else "未能生成有效的 SQL 查询。"

                print(f"[SQL Agent] 任务 {step_id} 完成！获取到硬指标数据。")

                updated_task = task.copy()
                updated_task["status"] = "done"
                updated_task["result"] = final_answer
                updates[step_id] = updated_task

            except Exception as e:
                print(f"[SQL Agent] 任务 {step_id} 执行异常: {e}")
                updated_task = task.copy()
                updated_task["status"] = "error"
                updated_task["result"] = f"查询失败: {str(e)}"
                updates[step_id] = updated_task

    return {"tasks": updates}