import asyncio
import os
from dotenv import load_dotenv
import argparse
import json
import re
from server.db.models import init_db
from server.knowledge_base.ingestion import ingestion_manager

from server.agent.graph import build_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Agentic RAG"
def run_ingestion():
    """
    入口一：离线数据处理与知识库构建
    """
    print("\n" + "=" * 50)
    print("[离线处理模式] 正在构建乡村振兴数字化基座...")
    print("=" * 50)

    init_db()

    try:
        asyncio.run(ingestion_manager.run_full_ingestion_async())
        print("\n数据灌装完毕！你的文旅大脑已经装载了最新的知识。")
    except Exception as e:
        print(f"\n灌装过程中发生异常: {e}")


async def run_chat_loop_async():
    print("\n" + "=" * 50)
    print("[CLI 测试模式] 石榴智策·终端侦探模式已启动")
    print("提示: 输入 'quit' 或 'exit' 退出系统")
    print("=" * 50)
    from server.agent.memory import shiliu_memory

    checkpointer = MemorySaver()
    agent_app = build_agent(checkpointer=checkpointer)

    # 固定一个测试账号 ID
    session_id = "shiliu_master"
    config = {"configurable": {"thread_id": session_id}}

    while True:
        try:
            user_input = input("\n👤 用户: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue

            print("\n🤖 石榴智策思考中...\n")

            # ==========================================================
            # ⚡ 并发双路记忆激活（热画像 + 冷联想并行）
            # ==========================================================
            try:
                hot_task  = asyncio.to_thread(shiliu_memory.get_all, user_id=session_id)
                cold_task = asyncio.to_thread(shiliu_memory.search, user_input, user_id=session_id, limit=3)
                raw_memories, related_res = await asyncio.gather(hot_task, cold_task)

                # --- 热画像 ---
                memory_list = []
                if isinstance(raw_memories, dict):
                    memory_list = raw_memories.get("results", []) or raw_memories.get("memories", [])
                facts = [m.get("memory") if isinstance(m, dict) else str(m) for m in memory_list]
                facts = [f for f in facts if f and f != "results"]
                profile_summary = "\n".join([f"- {f}" for f in facts]) if facts else "无核心偏好记录"

                # --- 冷联想 ---
                if isinstance(related_res, dict):
                    related_items = related_res.get("results", []) or related_res.get("memories", [])
                elif isinstance(related_res, list):
                    related_items = related_res
                else:
                    related_items = []
                related_memories = []
                for m in related_items:
                    m_text = m.get("memory") if isinstance(m, dict) else str(m)
                    if m_text:
                        related_memories.append(m_text)
                past_context = "\n".join([f"- {rm}" for rm in related_memories]) if related_memories else "无相关历史背景"

                print(f"[Debug]记忆已激活！\n[热画像]: {len(facts)}条 | [冷联想]: {len(related_memories)}条")

            except Exception as e:
                print(f"[Memory] 记忆引擎唤醒失败: {e}")
                profile_summary = "无核心偏好记录"
                past_context = "无相关历史背景"

                # ==== 2. 构造最新的 AgentState ====
            state = {
                "session_id": session_id,
                "user_query": user_input,
                "enable_web_search": False,
                "messages": [{"role": "user", "content": user_input}],
                # ⚡ 现在的画像既包含你是谁，也包含跟这个问题相关的往事
                "long_term_profile": {
                    "summary": profile_summary,  # 核心人设 (SQLite)
                    "past_context": past_context  # 语义相关联想 (Chroma 激活处)
                },
                "tasks": {}
            }

            # ==== 3. 异步事件流侦听 (复刻 Web Server 体验) ====
            async for event in agent_app.astream_events(state, config=config, version="v2"):
                kind = event["event"]
                name = event["name"]

                # 拦截工具启动
                if kind == "on_tool_start":
                    tool_input = event.get("data", {}).get("input", {})
                    args_str = ", ".join([str(v) for v in tool_input.values() if v]) if isinstance(tool_input,
                                                                                                   dict) else str(
                        tool_input)
                    print(f"  [特工行动] -> 调用工具: {name} | 参数: {args_str}")

                # 拦截图片生成
                elif kind == "on_tool_end" and name == "generate_image_tool":
                    output = event.get("data", {}).get("output", "")
                    match = re.search(r"\[IMAGE_URL:\s*(https?://[^\s\]]+)\]", str(output))
                    if match:
                        print(f"  🖼️ [视觉截获] -> 生成了一张精美图片！\n     链接: {match.group(1)}")

                # 拦截节点结束与最终答案
                elif kind == "on_chain_end":
                    output = event.get("data", {}).get("output", {})
                    if not isinstance(output, dict): continue

                    # 播报 Planner 规划步骤
                    if name == "planner" and "tasks" in output:
                        tasks = output["tasks"]
                        for step_id in sorted(tasks.keys()):
                            task = tasks[step_id]
                            print(
                                f"[战略规划] -> 步骤 {step_id}: {task.get('description', '')[:30]}... (交由 {task.get('agent', '')})")

                    # 播报最终回复
                    elif name == "synthesizer" and "final_answer" in output:
                        # print(f"\n慧聚石榴:\n{output['final_answer']}")
                        pass

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n系统异常: {str(e)}")


def run_chat_loop():
    # 将原来的同步执行改为驱动异步的 run_chat_loop_async
    asyncio.run(run_chat_loop_async())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="慧聚石榴 - 乡村文化振兴多智能体编排系统")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['ingest', 'chat'],
        default='chat',
        help="运行模式：'ingest' 为知识库数据灌装，'chat' 为启动交互式问答"
    )

    args = parser.parse_args()

    if args.mode == 'ingest':
        run_ingestion()
    elif args.mode == 'chat':
        run_chat_loop()