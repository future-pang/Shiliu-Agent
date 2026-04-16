import asyncio
from typing import Literal, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate

from configs.settings import settings
from configs.prompt_loader import get_prompt
from server.tools.rag_tools import RAG_TOOLS
from server.tools.mcp_tools import web_search
from server.agent.state import RagInternalState
from server.tools.rag_tools import maps_context
from server.agent.llm_factory import get_rag_llm, get_rag_retrieve_llm, get_grader_llm


class Finding(BaseModel):
    fact: str = Field(description="极度精简的核心事实陈述，字数少一点，绝对不要包含任何修饰语")
    source: str = Field(description="完全照抄原文的【文献溯源标志】，不要加书名号")

class ResearchOutput(BaseModel):
    findings: List[Finding] = Field(description="本次检索发现的所有事实列表")


def format_context(rag_context_list: list) -> str:
    if not rag_context_list:
        return "无任何有效检索信息。"
    parts = []
    for item in rag_context_list:
        if isinstance(item, dict):
            parts.append(f"{item.get('content', '')} [来源: 《{item.get('source', '未知')}》]")
        else:
            parts.append(str(item))
    return "\n---\n".join(parts)


async def rag_retrieve_node(state: RagInternalState) -> dict:
    """
    ⚡ 已优化：
    1. 使用单例 LLM（llm_factory），消除 ChatOpenAI 重复初始化
    2. 使用缓存 Prompt（prompt_loader），消除重复磁盘 I/O
    3. 并发提取结构化内容（asyncio.gather + Semaphore）
    """
    print(f"\n[RAG Agent] 启动全自动深度检索循环 (当前尝试次数: {state.get('rag_loop_count', 0) + 1})")

    query = state.get("query", "")
    query_type = state.get("query_type", "mixed")
    history_visited = state.get("visited_nodes", [])
    history_notice = f"\n【最高警告：以下 Node_ID 已被精读，严禁重复读取】：{', '.join(history_visited)}" if history_visited else ""
    rag_judgement_reason = state.get("rag_judgement_reason", "")

    agent_llm = get_rag_llm()
    extract_llm = get_rag_retrieve_llm()
    structured_llm = extract_llm.with_structured_output(ResearchOutput)

    rag_prompt_template = get_prompt("rag_agent")
    formatted_prompt = rag_prompt_template.format(judgement_reason_str=rag_judgement_reason or "无")

    rag_app = create_react_agent(
        agent_llm,
        tools=RAG_TOOLS,
        prompt=formatted_prompt
    )

    try:
        inputs = {"messages": [("user", f"请针对任务查询并汇报：\n{query}\n【系统指定检索分辨率】：query_type=\"{query_type}\"\n{history_notice}\n提示：尽力搜索即可，严禁死循环。")]}
        response_state = await rag_app.ainvoke(inputs, config={"recursion_limit": 10})

        tool_contents = []
        current_ids = []

        for msg in response_state["messages"]:
            if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") == "read_chunk":
                tool_contents.append(msg.content)
            if hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    if tc['name'] == 'read_chunk':
                        current_ids.append(tc['args']['node_id'])

        expanded_visited = set(history_visited) | set(current_ids)

        # 只要本次调了 read_chunk，或者之前已经调过，物理锁就打开
        has_deep_read = len(current_ids) > 0 or state.get("has_deep_read", False)

        structured_context = []

        if tool_contents:
            #  并发提取所有 chunk 的结构化内容
            sem = asyncio.Semaphore(2)
            async def extract_single_chunk(chunk_content: str):
                async with sem:
                    extraction_prompt = (
                        "你是一个无情的 JSON 数据提取机器。请阅读以下文献片段，提取核心事实。\n"
                        "【最高红线警告】：\n"
                        "1. 绝对禁止写过渡句，只要极度精简的短句。\n"
                        "2. 每一个事实必须配备对应的【文献溯源标志】。\n"
                        f"【原始文献片段】\n{chunk_content}"
                    )
                    return await structured_llm.ainvoke(extraction_prompt)

            tasks = [extract_single_chunk(chunk) for chunk in tool_contents]
            results = await asyncio.gather(*tasks)

            for res in results:
                if hasattr(res, 'findings'):
                    for f in res.findings:
                        structured_context.append({"content": f.fact, "source": f.source})
        else:
            last_msg = response_state["messages"][-1].content
            structured_context = [{"content": last_msg, "source": "检索总结"}]

        return {
            "rag_context": structured_context,
            "rag_loop_count": state.get("rag_loop_count", 0) + 1,
            "visited_nodes": list(expanded_visited),  # 干净清爽的精确已读列表
            "has_deep_read": has_deep_read
        }

    except Exception as e:
        print(f"[RAG Agent] 提取阶段崩溃了: {e}")
        err_info = "\n".join(tool_contents)[:500] if 'tool_contents' in locals() else "无资料"
        return {
            "rag_context": [{"content": f"提取异常，原始片段：{err_info}...", "source": "系统错误"}],
            "rag_loop_count": state.get("rag_loop_count", 0) + 1
        }

# =====================================================================
# CRAG 判官结构化输出定义
# =====================================================================
class GraderOutput(BaseModel):
    grade: Literal["correct", "incorrect", "ambiguous"] = Field(
        description="评估检索内容质量。correct: 完全包含答案；ambiguous: 方向对但缺少关键拼图/细节；incorrect: 毫无关系或方向全错。"
    )
    feedback: str = Field(
        description="评判理由。详细指出当前资料到底缺了什么事实。"
    )
    missing_entity: str = Field(
        default="",
        description="【核心指令】：如果 grade 为 ambiguous，请务必在此提取出最核心的 1 个缺失实体词（如'彝族漆器'、'票价'等），系统将用它去后台暗网图谱中捞取补充资料。如果不需要系统干预则留空。"
    )

# =====================================================================
# 独立判官节点
# =====================================================================
async def grader_node(state: RagInternalState) -> dict:
    """
    1. 使用单例 LLM（llm_factory），消除重复初始化
    2. 使用缓存 Prompt（prompt_loader），消除重复磁盘 I/O
    """
    print("\n[Grader] 判官升堂，开始执行 CRAG 纠错质检 (Correct / Ambiguous / Incorrect)...")

    visited = state.get("visited_nodes", [])
    # ==== 如果 LLM 没调用 read_chunk
    if not visited:
        print("[Grader 物理锁] 拦截！检测到特工试图用摘要蒙混过关！")
        return {
            "rag_judgement_passed": False,
            "rag_judgement_reason": "【系统最高红牌警告】检测到你刚才只看了预览摘要，未调用 `read_chunk` 读取具体正文！严禁拿摘要敷衍交差！请立刻去调工具精读正文！",
            "rag_missing_entity": "",
            "rag_final_result": ""
        }

    query = state.get("query", "")
    rag_context_list = state.get("rag_context", [])
    context_str = format_context(rag_context_list)

    system_prompt = get_prompt("rag_grader")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "【原始任务】\n{query}\n\n【研究员提交的本地上下文】\n{context}")
    ])

    llm = get_grader_llm()
    structured_grader = llm.with_structured_output(GraderOutput)
    chain = prompt | structured_grader

    try:
        result: GraderOutput = await chain.ainvoke({"query": query, "context": context_str})
        grade = result.grade
        feedback = result.feedback
        missing_entity = result.missing_entity if grade == "ambiguous" else ""

        print(f"[Grader] 判决结果: {grade.upper()} | 意见: {feedback} | 需跃迁实体: {missing_entity or '无'}")

        passed = (grade == "correct")
        final_result = context_str if passed else ""

        return {
            "rag_judgement_passed": passed,
            "rag_judgement_reason": feedback,
            "rag_missing_entity": missing_entity,
            "rag_final_result": final_result
        }

    except Exception as e:
        print(f"[Grader] 质检异常: {e}")
        return {"rag_judgement_passed": True, "rag_final_result": context_str}

# =====================================================================
# 优雅降级节点
# =====================================================================
async def force_prompt_node(state: RagInternalState) -> dict:
    """
    ⚡ 已优化：
    1. 使用单例 LLM（llm_factory），消除重复初始化
    2. 使用缓存 Prompt（prompt_loader），消除重复磁盘 I/O
    """
    print("\n[Force Prompt] 检索次数耗尽，触发优雅降级，进行局部总结...")

    query = state.get("query", "")
    rag_context_list = state.get("rag_context", [])
    context_str = format_context(rag_context_list)

    if not context_str.strip():
        context_str = "无任何有效检索信息。"

    web_res = ""
    if state.get("enable_web_search", False):
        print("[Force Prompt] 联网搜索已开启，正在去全网捞取补丁...")
        try:
            search_query = f"{query} {state.get('rag_judgement_reason', '')}"
            web_res = web_search.invoke({"query": search_query})
            web_res = f"\n\n【全网实时补充内容】\n{web_res}"
        except Exception as e:
            print(f"[Force Prompt] 联网补丁失败 (可能是没配置 Key): {e}")
    else:
        print("[Force Prompt] 联网功能已关闭，仅基于本地知识强行交卷。")

    combined_context = f"【本地碎片积累】\n{context_str}{web_res}"

    force_prompt_text = get_prompt("rag_force")
    prompt = ChatPromptTemplate.from_messages([
        ("system", force_prompt_text),
        ("human", f"【原始任务】\n{query}\n\n【目前收集到的资料库】\n{combined_context}")
    ])

    llm = get_rag_llm()

    try:
        response = await llm.ainvoke(prompt.format_messages())
        return {"rag_final_result": response.content}
    except Exception as e:
        print(f"[Force Prompt] 总结失败: {e}")
        return {"rag_final_result": f"由于检索多次未果且总结出错，仅能提供原始碎片：{context_str[:200]}..."}


# =====================================================================
# 系统路由节点：图谱跃迁执行器
# =====================================================================
async def graph_leap_node(state: RagInternalState) -> dict:
    """
    当判官给出 ambiguous 且提取了 missing_entity 时，系统自动执行此节点。
    它绕过 Agent，直接去底层图谱中捞资料，然后把新资料塞进上下文。
    """
    entity = state.get("rag_missing_entity", "")
    print(f"\n[System Leap] 系统接管：根据判官反馈，后台强制触发 '{entity}' 的图谱跃迁！")

    if not entity:
        return {}

    # ==== 调用后台普通函数捞取资料 ====
    leap_result = maps_context(entity=entity)
    current_context = state.get("rag_context", [])

    # ==== 将跃迁出来的暗网知识，无缝追加到当前的上下文中 ====
    current_context.append({
        "content": f"【系统自动图谱跃迁 - 补充事实 (关键词：{entity})】\n{leap_result}",
        "source": "全局暗网图谱关联"
    })

    # ===== 带着新资料，修改 feedback，让 Agent 知道系统已经帮它找好了 ====
    new_reason = f"判官反馈：{state.get('rag_judgement_reason')}\n\n【系统协助通知】：系统已在后台为你强行跃迁捞取了 '{entity}' 的关联资料（已加入你的资料库），请结合新旧资料，重新整理并提交回答！"

    return {
        "rag_context": current_context,
        "rag_judgement_reason": new_reason,
        "rag_missing_entity": ""  # 消费完毕，清空实体防死循环
    }