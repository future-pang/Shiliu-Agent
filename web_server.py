import os
import re
import json
import asyncio
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from server.agent.memory import shiliu_memory
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from redis.asyncio import Redis
from server.agent.graph import build_agent
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic RAG"
# ====================================================
# 使用 lifespan 管理异步资源
# ====================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6379")
    pwd = os.getenv("REDIS_PASSWORD", "")
    db = os.getenv("REDIS_DB", "0")

    if pwd:
        redis_url = f"redis://:{pwd}@{host}:{port}/{db}"
    else:
        redis_url = f"redis://{host}:{port}/{db}"

    try:
        async with AsyncRedisSaver.from_conn_string(redis_url) as saver:

            print(f"正在连接 Redis 检查点...")
            app.state.agent_app = build_agent(checkpointer=saver)
            print("石榴智策大脑加载完毕！")
            yield
    except Exception as e:
        print(f"Redis 认证或连接失败: {e}")
        raise e

app = FastAPI(title="慧聚石榴 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    session_id: str = "shiliu_master"


# ====================================================
# ⚡ 并发双路记忆加载（热画像 + 冷联想同时跑）
# ====================================================
async def _load_memories_parallel(query: str, user_id: str):
    """
    用 asyncio.gather 并发执行两路记忆读取。
    原串行总耗时 = 热画像耗时 + 冷联想耗时
    优化后总耗时 = max(热画像耗时, 冷联想耗时)
    """
    hot_task  = asyncio.to_thread(shiliu_memory.get_all,  user_id=user_id)
    cold_task = asyncio.to_thread(shiliu_memory.search, query, user_id=user_id, limit=3)
    raw_memories, related_res = await asyncio.gather(hot_task, cold_task)

    # --- 热画像解析 ---
    memory_list = []
    if isinstance(raw_memories, dict):
        memory_list = raw_memories.get("results", []) or raw_memories.get("memories", [])
    elif isinstance(raw_memories, list):
        memory_list = raw_memories
    facts = [m.get("memory") if isinstance(m, dict) else str(m) for m in memory_list]
    facts = [f for f in facts if f and f != "results"]
    profile_summary = "\n".join([f"- {f}" for f in facts]) if facts else "无核心偏好记录"

    # --- 冷联想解析 ---
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

    return facts, profile_summary, related_memories, past_context


# ====================================================
# 核心流式聊天接口
# ====================================================
@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: Request, chat_data: ChatRequest):
    agent_app = request.app.state.agent_app
    target_id = "shiliu_master"
    config = {"configurable": {"thread_id": target_id}}

    # ==========================================================
    # ⚡ 并发双路记忆激活
    # ==========================================================
    try:
        facts, profile_summary, related_memories, past_context = await _load_memories_parallel(
            query=chat_data.query, user_id=target_id
        )
        print(f"[Web Server] 🚀 记忆已激活！[热画像]: {len(facts)}条 | [冷联想]: {len(related_memories)}条")

    except Exception as e:
        print(f"[Web Server Memory] 记忆引擎唤醒失败: {e}")
        facts, related_memories = [], []
        profile_summary = "无核心偏好记录"
        past_context = "无相关历史背景"

    # ==== 组装 State，严格对齐 Synthesizer 的要求 ====
    state = {
        "session_id": target_id,
        "user_query": chat_data.query,
        "enable_web_search": True,
        "messages": [{"role": "user", "content": chat_data.query}],
        "long_term_profile": {
            "summary": profile_summary,
            "past_context": past_context
        },
        "tasks": {}
    }

    async def event_generator():
        memory_msg = f"[系统状态] 记忆引擎已唤醒：调取 {len(facts)} 条画像，关联 {len(related_memories)} 条往事。"
        yield f"data: {json.dumps({'type': 'tool_call', 'msg': memory_msg}, ensure_ascii=False)}\n\n"

        TOOL_MAP = {
            "web_search": "全网检索", "weather_api": "实时天气", "around_search_api": "周边搜索",
            "get_current_time": "系统时间", "walking_plan_api": "路径规划",
            "preview_docs": "知识库初筛", "read_chunk": "档案精读","generate_image_tool": "文创视觉生成"
        }

        async for event in agent_app.astream_events(state, config=config, version="v2"):
            kind = event["event"]
            name = event["name"]

            # 拦截工具启动日志
            if kind == "on_tool_start":
                tool_input = event.get("data", {}).get("input", {})
                args_str = ", ".join([str(v) for v in tool_input.values() if v]) if isinstance(tool_input,
                                                                                               dict) else str(
                    tool_input)
                friendly_name = TOOL_MAP.get(name, name)

                log_msg = f"[特工行动] 正在调用 {friendly_name} ({args_str})"
                if name == "web_search":
                    log_msg = f"[联网搜索] 正在全网深度检索: {args_str}..."

                yield f"data: {json.dumps({'type': 'tool_call', 'msg': log_msg}, ensure_ascii=False)}\n\n"

            elif kind == "on_tool_end" and name == "generate_image_tool":
                import re
                output = event.get("data", {}).get("output", "")

                match = re.search(r"\[IMAGE_URL:\s*(https?://[^\s\]]+)\]", str(output))
                if match:
                    real_img_url = match.group(1)
                    print(f"[Web Server] 截获真实图片链接，绕过大模型直接推送给前端！")
                    yield f"data: {json.dumps({'type': 'image_generated', 'url': real_img_url}, ensure_ascii=False)}\n\n"

            elif kind == "on_chain_end":
                output = event.get("data", {}).get("output", {})
                if not isinstance(output, dict): continue

                if name == "planner" and "tasks" in output:
                    tasks = output["tasks"]
                    for step_id in sorted(tasks.keys()):
                        task = tasks[step_id]
                        msg = f"计划步骤 {step_id}: {task.get('description', '')} (指派给 {task.get('agent', '')})"
                        yield f"data: {json.dumps({'type': 'planner', 'msg': msg}, ensure_ascii=False)}\n\n"

                elif name == "synthesizer" and "final_answer" in output:
                    yield f"data: {json.dumps({'type': 'final_answer', 'content': output['final_answer']}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("web_server:app", host="0.0.0.0", port=8000, reload=True)