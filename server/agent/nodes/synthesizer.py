import threading
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate

from configs.settings import settings
from configs.prompt_loader import get_prompt
from server.agent.memory import shiliu_memory
from server.agent.state import AgentState, TaskPlan
from server.agent.llm_factory import get_summarizer_llm


def save_memory_background(query, final_answer, session_id):
    try:
        shiliu_memory.add(
            f"用户的个人特质/偏好(必须中文记录): {query}",
            user_id="shiliu_master"
        )
        print("[Synthesizer Background] 长期特质记忆已更新！")
    except Exception as e:
        print(f"[Synthesizer Background] Mem0 存储再次异常: {e}")

# =====================================================================
# 统筹生成器核心逻辑
# =====================================================================
async def synthesizer_node(state: AgentState) -> dict:
    """
    ⚡ 已优化：
    1. 使用单例 LLM（llm_factory），消除重复初始化
    2. 使用缓存 Prompt（prompt_loader），消除重复磁盘 I/O
    """
    print("\n[Synthesizer] 统筹生成器启动，开始撰写最终回复...")

    session_id = state.get("session_id", "default_user")
    query = state.get("user_query", "")
    is_chat_only = state.get("is_chat_only", False)
    tasks: Dict[int, TaskPlan] = state.get("tasks", {})

    # ==== 读取历史记忆 ====
    messages = state.get("messages", [])
    history_msgs = messages[:-1] if len(messages) > 0 else []

    history_str = "\n".join([f"{'用户' if m['role'] == 'user' else '石榴智策'}: {m['content']}" for m in history_msgs])
    if not history_str:
        history_str = "无"

    history_str = history_str.replace("{", "{{").replace("}", "}}")
    long_term_profile = state.get("long_term_profile", {})

    persona = long_term_profile.get("summary", "无")
    recalled_memories = long_term_profile.get("past_context", "无")

    full_profile_str = f"【核心偏好】:\n{persona}\n\n【相关的往事联想】:\n{recalled_memories}"

    # ==== 拼装所有特工的执行结果 ====
    context_parts = []
    rag_context_list = state.get("rag_context", [])

    for idx, item in enumerate(rag_context_list):
        fact = item.get("content", "")
        source = item.get("source", "未知来源")
        context_parts.append(f"事实 {idx + 1}: {fact} [来源: 《{source}》]")

    context_str = "\n".join(context_parts)

    if not is_chat_only and tasks:
        sorted_tasks = sorted(tasks.values(), key=lambda x: x.get("step", 999))
        for task in sorted_tasks:
            desc = task.get("description", "")
            res = task.get("result", "未获取到有效结果")
            agent_name = task.get("agent", "未知特工")
            context_parts.append(f"[{agent_name} 的调查结果 (任务: {desc})]:\n{res}")

    context_str = "\n\n".join(context_parts) if context_parts else "无后台检索数据。"
    context_str = context_str.replace("{", "{{").replace("}", "}}")

    # ==== 构建 Prompt（使用缓存，0 磁盘 I/O）====
    system_prompt_template = get_prompt("synthesizer")

    system_prompt = system_prompt_template.format(
        short_term_history=history_str,
        long_term_profile=full_profile_str,
        context=context_str
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "【用户的最新输入】\n{query}")
    ])

    llm = get_summarizer_llm()

    # ==== 生成最终回答 ====
    try:
        print("\n[Synthesizer] 正在流式生成最终回复：")
        print("=" * 60)

        final_answer = ""
        async for chunk in llm.astream(prompt.format_messages(query=query)):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                final_answer += chunk.content

        print("\n" + "=" * 60)
        print("[Synthesizer] 流式回复生成完毕！")

        threading.Thread(target=save_memory_background, args=(query, final_answer, session_id)).start()

        return {
            "final_answer": final_answer,
            "messages": [{"role": "assistant", "content": final_answer}]
        }

    except Exception as e:
        print(f"\n[Synthesizer] 生成异常: {e}")
        fallback_ans = "抱歉，我在整理导览信息时脑子短路了一下，能麻烦您再问一次吗？"
        return {
            "final_answer": fallback_ans,
            "messages": [{"role": "assistant", "content": fallback_ans}]
        }