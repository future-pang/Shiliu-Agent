from langchain_core.tools import tool

from server.knowledge_base.handler import kb_handler

# =====================================================================
# 预览搜索 (带已读过滤与高光透传)
# =====================================================================
@tool
async def preview_docs(query: str, query_type: str = "mixed", visited_node_ids: list[str] = None) -> str:
    """
    【步骤 1：查阅目录与摘要】
    当需要检索信息时，必须先调用此工具。
    它只返回高度相关的文档片段的 Node_ID、标题和摘要，**绝对不包含具体正文**。
    参数 query_type：系统已在提示中告知你本次检索的分辨率（macro/micro/mixed），请直接填入。
    如果有系统提示的"已精读节点"，请将它们放入 visited_node_ids 列表中，系统会自动过滤。
    """
    print(f"\n   [Tool] 1. 预览搜索: '{query}' (分辨率: {query_type})")
    visited = visited_node_ids or []

    try:
        nodes = kb_handler.retrieve_with_rerank(
            query_str=query,
            top_k=20,
            rerank_top_n=10,
            query_type=query_type
        )

        if not nodes:
            return "未找到相关文档，请更换提示词。"

        results = []
        valid_count = 0

        for node in nodes:
            node_id = node.node_id

            # 过滤已读节点
            if node_id in visited:
                continue

            metadata = node.node.metadata
            title = metadata.get('document_title', '未知文献')
            summary = metadata.get('summary', '无摘要')

            # 高光透传
            bubbled_snippets = metadata.get('bubbled_snippets', '')
            snippet_text = f"\n{bubbled_snippets}" if bubbled_snippets else ""

            results.append(
                f"--- 预览选项 {valid_count + 1} ---\n"
                f"Node_ID: {node_id}\n"
                f"标题: {title}\n"
                f"摘要: {summary}{snippet_text}\n"
            )

            valid_count += 1
            if valid_count >= 5:
                break

        threat_prompt = (
            "\n\n【系统强制指令】：以上仅为高度浓缩的目录摘要，绝对不足以用来回答用户的具体问题！\n"
            "你必须挑选出至少 2-3 个相关的 Node_ID，并立刻调用 `read_chunk` 工具获取完整正文。\n"
            "如果企图直接用摘要敷衍回答或者只看 1 篇，你将被立刻熔断并判定为幻觉！"
        )

        return "\n".join(results) + threat_prompt

    except Exception as e:
        return f"预览失败: {e}"

# =====================================================================
# 2：精读正文 (带暗网传送门)
# =====================================================================
@tool
async def read_chunk(node_id: str) -> str:
    """
    【步骤 2：精读正文】
    在 preview_docs 提供了 Node_ID 后，调用此工具读取该片段的完整原汁原味的正文内容。
    必须传入准确的 Node_ID 字符串。
    """
    print(f"\n   [Tool] 2. 精读节点: '{node_id}'")
    try:
        node = kb_handler.index.docstore.get_node(node_id)
        content = node.get_content()
        title = node.metadata.get('document_title', '未知文献')

        keywords = node.metadata.get('excerpt_keywords', '暂无关联概念')

        response = (
            f"【文献溯源标志】：{title}\n"
            f"【正文事实】：\n{content}\n\n"
            f"【系统提示：图谱跃迁传送门】\n"
            f"如果当前资料仍不足以完全回答问题，你可以调用 `maps_context` 工具，"
            f"传入以下精确的关联词，进行跨越物理文档的维度跳跃搜索：\n"
            f"可选关联实体：[{keywords}]"
        )
        return response
    except Exception as e:
        return f"读取失败: {e}"

# =====================================================================
# 图谱与物理扩展
# =====================================================================
def maps_context(entity: str) -> str:
    """
    后台专用的图谱跃迁引擎。不再暴露给 Agent。
    """
    # 直接bm25
    print(f"\n   [System] 触发后台图谱跃迁: 正在跨界追踪实体 '{entity}'...")
    try:
        bm25_nodes = kb_handler.bm25_retriever.retrieve(entity)
        if not bm25_nodes:
            return f"跃迁失败：未能在全局图谱中找到 '{entity}' 的相关资料。"

        results = []
        for i, n in enumerate(bm25_nodes[:3]):
            title = n.node.metadata.get('document_title', '未知文献')
            results.append(
                f"【跃迁补充资料 {i + 1}】来源于《{title}》\n"
                f"【正文事实】：{n.node.get_content()[:800]}..."
            )
        return "\n\n".join(results)
    except Exception as e:
        return f"图谱跃迁异常: {e}"

RAG_TOOLS = [preview_docs, read_chunk]