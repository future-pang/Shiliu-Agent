from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from configs.prompt_loader import get_prompt
from server.agent.state import AgentState, TaskPlan
from server.agent.llm_factory import get_planner_llm

# =====================================================================
# Pydantic 结构化输出
# =====================================================================
class TaskStep(BaseModel):
    step: int = Field(description="步骤编号，从1开始递增")
    description: str = Field(description="该步骤要完成的具体子任务描述，必须详细、明确")
    agent: str = Field(description="负责执行该任务的智能体名称（必须从提供的名录中严格选择）")
    args: Dict[str, Any] = Field(description="提取出的关键参数字典，如地点、价格上限等。无参数则为空字典 {}")
    depends_on: List[int] = Field(description="该任务依赖的前置步骤编号列表。若无依赖可并行，则为空列表 []")

class PlannerOutput(BaseModel):
    is_chat_only: bool = Field(description="用户输入是否为纯粹的日常闲聊")
    plan: List[TaskStep] = Field(description="拆解出的分步任务计划。若为纯闲聊，此列表为空")

# =====================================================================
# Tool 库与 Agent 库
# =====================================================================
MASTER_TOOL_DIRECTORY = {
    "weather_api": "查询特定地点的实时天气预报和气象预警。",
    "astronomy_api": "查询特定地点的日出日落、月升月落及月相信息，辅助规划摄影或景观游览。",
    "travel_advice_api": "获取针对特定地点的生活指数建议，包括穿衣、紫外线、运动、旅游建议等。",
    "weather_forecast_api": "查询特定地点未来3天（含明天、后天）的天气预报。",
    "walking_plan_api": "查询起点到终点的步行规划路线，包含距离、耗时和详细步骤。",
    "distance_api" : "测量两点间的距离，支持直线距离、驾车距离和步行距离。",
    "around_search_api" : "查询特定地点周边的餐饮、住宿、景点等信息，适合做本地生活推荐。",
    "static_map_api" : "生成特定地点的静态地图图片，适合提供视觉化的地理位置展示。",
    "get_current_time": "获取当前真实的系统日期和时间，用于时间基准对齐。",
    "web_search": "通过搜索引擎查询互联网实时信息、突发新闻、景区门票政策及通用百科知识。",
    "generate_image_tool": "文生图工具。专门用于将文字描述转化为精美的视觉图片。当用户要求'制作'、'画一张'、'设计'文创产品或风景画时，请务必派发此任务给 mcp_agent。"
}

MASTER_AGENT_DIRECTORY = {
    # "mcp_agent": "拥有外部 API 工具箱的特工。当前已为该特工装备的工具列表如下：\n{tool_directory}\n只能根据上述【已装备的具体工具】来受理并执行外部查询任务。",
    # "sql_agent": "擅长查询峨眉山硬指标数据，如：具体票价、各景点海拔高度、坐标、营业状态与时间等。处理明确的数字查询。",
    "rag_agent": "擅长查阅西南少数名族和文旅的知识库，如：非遗知识、节日习俗、景点历史渊源、传说故事、旅游攻略、周边特色美食推荐、住宿体验等。",
}

# =====================================================================
# 核心节点执行逻辑
# =====================================================================
async def planner_node(state: AgentState) -> dict:
    """
    1. async + ainvoke，不再阻塞事件循环
    2. 使用单例 LLM（llm_factory），消除重复初始化
    3. 使用缓存 Prompt（prompt_loader），消除重复磁盘 I/O
    """
    print("\n[Planner] 开始分析用户意图与任务拆解...")

    # ==== 提取上下文 ====
    query = state.get("user_query", "")
    enable_web_search = state.get("enable_web_search", False)
    messages = state.get("messages", [])
    history_msgs = messages[:-1] if len(messages) > 0 else []       # 短期记忆

    history_str = "\n".join([f"{'用户' if m['role'] == 'user' else '石榴智策'}: {m['content']}" for m in history_msgs])
    if not history_str:
        history_str = "无"
    long_term_profile = state.get("long_term_profile", "无")         # 长期记忆

    # ==== 动态过滤 Tool 名录 ====
    available_tools = MASTER_TOOL_DIRECTORY.copy()
    if not enable_web_search:
        available_tools.pop("web_search", None)
    tool_desc_lines = [f"      * {t_name}: {t_desc}" for t_name, t_desc in available_tools.items()]
    tool_directory_str = "\n".join(tool_desc_lines)

    # ==== 动态组装当前可用员工名录 ====
    available_agents = MASTER_AGENT_DIRECTORY.copy()
    if "mcp_agent" in available_agents:
        available_agents["mcp_agent"] = available_agents["mcp_agent"].format(
            tool_directory=tool_directory_str
        )
    agents_desc = "\n".join([f"- **{name}**: {desc}" for name, desc in available_agents.items()])

    # ==== 构建 Prompt（使用缓存，0 磁盘 I/O）====
    system_prompt_template = get_prompt("planner")
    profile_str = str(long_term_profile) if long_term_profile else "无"

    strict_json_instruction = (
        "\n\n【输出格式极其严格要求】\n"
        "你必须直接输出纯 JSON 字符串，禁止任何 Markdown 格式。内容必须严格匹配以下结构，严禁修改任何键名（Key）：\n"
        "{{\n"
        "  \"is_chat_only\": false,\n"
        "  \"plan\": [\n"
        "    {{\n"
        "      \"step\": 1,\n"
        "      \"description\": \"这里写具体的任务描述\",\n"
        "      \"agent\": \"mcp_agent\",\n"
        "      \"args\": {{}},\n"
        "      \"depends_on\": []\n"
        "    }}\n"
        "  ]\n"
        "}}\n"
        "注意：'step' 和 'description' 是必填项，绝对不能遗漏或更名！"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template + strict_json_instruction),
        ("human", "【用户输入】\n{query}")
    ])

    llm = get_planner_llm()
    structured_llm = llm.with_structured_output(PlannerOutput)
    chain = prompt | structured_llm

    # ==== 异步调用，不再阻塞事件循环 ====
    try:
        result: PlannerOutput = await chain.ainvoke({
            "agent_directory": agents_desc,
            "short_term_history": history_str,
            "long_term_profile": profile_str,
            "query": query
        })

        # ==== 结果解析与状态更新 ====
        if result.is_chat_only:
            print(" [Planner] 判定为纯闲聊/寒暄，交由统筹器直接处理。")
            return {
                "is_chat_only": True,
                "tasks": {}
            }
        else:
            task_dict: Dict[int, TaskPlan] = {}
            for step_item in result.plan:
                task_dict[step_item.step] = {
                    "step": step_item.step,
                    "description": step_item.description,
                    "agent": step_item.agent,
                    "args": step_item.args,
                    "depends_on": step_item.depends_on,
                    "status": "pending",
                    "result": None
                }

            print(f"[Planner] 成功拆解出 {len(task_dict)} 个子任务 (DAG):")
            for t_id, t in task_dict.items():
                deps = f"依赖->{t['depends_on']}" if t['depends_on'] else "无依赖(可并发)"
                print(f"      [{t_id}] {t['agent']} | {deps} | 参数: {t['args']} | 任务: {t['description']}")

            return {
                "is_chat_only": False,
                "tasks": task_dict
            }

    except Exception as e:
        print(f"[Planner] 结构化解析崩溃或大模型超时: {e}")
        fallback_task: Dict[int, TaskPlan] = {
            1: {
                "step": 1,
                "description": query,
                "agent": "rag_agent",
                "args": {},
                "depends_on": [],
                "status": "pending",
                "result": None
            }
        }
        return {"is_chat_only": False, "tasks": fallback_task}
