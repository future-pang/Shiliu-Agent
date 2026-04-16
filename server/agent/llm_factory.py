"""
LLM 单例工厂模块

使用 @lru_cache 保证整个进程生命周期内每种角色的 LLM 只初始化一次。
原先每次节点调用都会 new 一个 ChatOpenAI 对象（触发 httpx 连接池初始化等），
改为此模块后，对象初始化成本降为 0（首次启动后）。

同时配置了共享 httpx.AsyncClient，在所有异步 LLM 之间复用 TCP 连接池，
减少高并发下频繁建连的开销。
"""
import httpx
from functools import lru_cache
from langchain_openai import ChatOpenAI
from configs.settings import settings


# =====================================================================
# 全局共享 HTTP 连接池（所有异步 LLM 共用）
# =====================================================================
_shared_async_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=30,
        keepalive_expiry=30.0,
    ),
    timeout=httpx.Timeout(
        connect=5.0,
        read=120.0,
        write=10.0,
        pool=5.0,
    ),
)

# 同步场景用的共享客户端（如 planner 在 LangGraph 同步上下文中使用）
_shared_sync_client = httpx.Client(
    limits=httpx.Limits(
        max_connections=50,
        max_keepalive_connections=20,
        keepalive_expiry=30.0,
    ),
    timeout=httpx.Timeout(
        connect=5.0,
        read=120.0,
        write=10.0,
        pool=5.0,
    ),
)


# =====================================================================
# LLM 单例工厂
# =====================================================================

@lru_cache(maxsize=1)
def get_planner_llm() -> ChatOpenAI:
    """Planner：任务规划与拆解（Large 级别）"""
    cfg = settings.planner_llm
    print(f"[LLMFactory]初始化 planner_llm: {cfg['model_id']}")
    return ChatOpenAI(
        model=cfg['model_id'],
        api_key=cfg['api_key'],
        base_url=cfg.get('base_url'),
        temperature=0.0,
        http_async_client=_shared_async_client,
        http_client=_shared_sync_client,
    )


@lru_cache(maxsize=1)
def get_rag_llm() -> ChatOpenAI:
    """RAG Agent：ReAct 主力检索（Medium 级别）"""
    cfg = settings.rag_llm
    print(f"[LLMFactory]初始化 rag_llm: {cfg['model_id']}")
    return ChatOpenAI(
        model=cfg['model_id'],
        api_key=cfg['api_key'],
        base_url=cfg.get('base_url'),
        temperature=0.1,
        http_async_client=_shared_async_client,
        http_client=_shared_sync_client,
    )


@lru_cache(maxsize=1)
def get_rag_retrieve_llm() -> ChatOpenAI:
    """RAG Retrieve：结构化内容提取（Small 级别）"""
    cfg = settings.rag_retrieve
    print(f"[LLMFactory]初始化 rag_retrieve_llm: {cfg['model_id']}")
    return ChatOpenAI(
        model=cfg['model_id'],
        api_key=cfg['api_key'],
        base_url=cfg.get('base_url'),
        temperature=0.0,
        http_async_client=_shared_async_client,
        http_client=_shared_sync_client,
    )


@lru_cache(maxsize=1)
def get_grader_llm() -> ChatOpenAI:
    """Grader：独立判官，质检幻觉（Large 级别）"""
    cfg = settings.grader_llm
    print(f"[LLMFactory]初始化 grader_llm: {cfg['model_id']}")
    return ChatOpenAI(
        model=cfg['model_id'],
        api_key=cfg['api_key'],
        base_url=cfg.get('base_url'),
        temperature=0.0,
        http_async_client=_shared_async_client,
        http_client=_shared_sync_client,
    )


@lru_cache(maxsize=1)
def get_tool_llm() -> ChatOpenAI:
    """MCP Agent：工具调用执行（Medium 级别）"""
    cfg = settings.tool_llm
    print(f"[LLMFactory]初始化 tool_llm: {cfg['model_id']}")
    return ChatOpenAI(
        model=cfg['model_id'],
        api_key=cfg['api_key'],
        base_url=cfg.get('base_url'),
        temperature=0.1,
        http_async_client=_shared_async_client,
        http_client=_shared_sync_client,
    )


@lru_cache(maxsize=1)
def get_summarizer_llm() -> ChatOpenAI:
    """Synthesizer：最终答案汇总（Small 级别）"""
    cfg = settings.summarizer_llm
    print(f"[LLMFactory]初始化 summarizer_llm: {cfg['model_id']}")
    return ChatOpenAI(
        model=cfg['model_id'],
        api_key=cfg['api_key'],
        base_url=cfg.get('base_url'),
        temperature=0.7,
        http_async_client=_shared_async_client,
        http_client=_shared_sync_client,
    )


@lru_cache(maxsize=1)
def get_sql_llm() -> ChatOpenAI:
    """SQL Agent：SQL 生成（Large 级别）"""
    cfg = settings.sql_llm
    print(f"[LLMFactory]初始化 sql_llm: {cfg['model_id']}")
    return ChatOpenAI(
        model=cfg['model_id'],
        api_key=cfg['api_key'],
        base_url=cfg.get('base_url'),
        temperature=0.0,
        http_async_client=_shared_async_client,
        http_client=_shared_sync_client,
    )
