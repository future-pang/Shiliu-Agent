import chromadb
from llama_index.core.base.embeddings.base import similarity
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.postprocessor import LLMRerank
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore

from configs.settings import settings

class EmeiKnowledgeBase:
    def __init__(self):
        self.index = None
        self.node_store = None
        self.bm25_retriever = None

        llm_conf = settings.llm
        Settings.llm = OpenAILike(
            api_key=llm_conf['api_key'],
            api_base=llm_conf['base_url'],
            model=llm_conf['model_id'],
            is_chat_model=True,
            context_window=llm_conf.get('context_window', 128000),
            timeout=60.0
        )

        embed_conf = settings.embed
        Settings.embed_model = OpenAIEmbedding(
            api_key=settings.aliyun_API_KEY,
            api_base=embed_conf['base_url'],
            model_name=embed_conf['model_id'],
            embed_batch_size=embed_conf.get('batch_size', 10)
        )

    def initialize(self):
        """
        直接从磁盘chroma_db中加载持久化的向量仓库
        """
        chroma_conf = settings.chroma_config
        client = chromadb.PersistentClient(path=chroma_conf['absolute_path'])
        chroma_collection = client.get_or_create_collection(chroma_conf['collection_name'])
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # ==== 同时加载 Chroma 和 本地的 docstore ====
        docstore_dir = str(settings.BASE_DIR / "storage" / "docstore")
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=docstore_dir
        )

        self.index = load_index_from_storage(
            self.storage_context,
            embed_model=Settings.embed_model
        )

        all_nodes = list(self.storage_context.docstore.docs.values())
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=20,
        )

        print("知识库加载成功！双路检索引擎（Vector + BM25）初始化完毕。")


    # =====================================================================
    # 带 RRF 双路融合、重排前置、动态折叠的检索管线
    # =====================================================================
    def retrieve_with_rerank(self, query_str: str, top_k: int = 20, rerank_top_n: int = 10, query_type: str = "mixed"):
        """
        参数 query_type:
          - "micro": 只要最精细的叶子，不合并
          - "macro": 只要合并后的宏观父节点（替换掉叶子）
          - "mixed": 既保留合并的宏观父节点，也保留底层的微观叶子
        """
        if not self.index: self.initialize()

        # ==== 双路并发 + RRF 融合 ====
        vector_retriever = self.index.as_retriever(similarity_top_k=top_k)

        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, self.bm25_retriever],
            similarity_top_k=top_k,
            num_queries=1,
            mode="reciprocal_rerank"
        )

        fused_nodes = fusion_retriever.retrieve(query_str)
        if not fused_nodes:
            return []

        # ==== 重排前置 (确保硬证据的高分) ====
        reranker = DashScopeRerank(
            model=settings.reranker_llm['model_id'],
            api_key=settings.reranker_llm['api_key'],
            top_n=rerank_top_n
        )

        reranked_leaf_nodes = reranker.postprocess_nodes(
            fused_nodes,
            query_str=query_str
        )

        if query_type == "micro":
            return reranked_leaf_nodes

        # ==== 轻量级自研 Auto-Merging 与 高光透传 ====
        parent_counts = {}      # 统计每个父节点被命中多少次
        parent_map = {}         # 映射：父节点 ID → 命中的所有叶子节点列表

        # ==== 统计哪些叶子属于同一个父亲 ====
        for n in reranked_leaf_nodes:
            if n.node.parent_node:
                parent_id = n.node.parent_node.node_id
                parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
                if parent_id not in parent_map:
                    parent_map[parent_id] = []
                parent_map[parent_id].append(n)

        final_nodes = []
        merged_parent_ids = set()
        MERGE_THRESHOLD = 3     # 合并的阈值

        for parent_id, count in parent_counts.items():
            if count >= MERGE_THRESHOLD:
                try:
                    # 从图谱库中捞出那个宏观的父节点
                    parent_node = self.storage_context.docstore.get_node(parent_id)

                    high_lights = []    # 高光透传(提取触发改合并的叶子证据)
                    for child_w_score in parent_map[parent_id]:
                        # 截取子节点的前100字作为防致盲证据，去掉换行符
                        snippet = child_w_score.node.get_content()[:100].replace('\n', ' ')
                        high_lights.append(f"[硬核片段]: {snippet}...")

                    # 强行塞入父节点的 metadata 中，供 Agent 预览
                    parent_node.metadata["bubbled_snippets"] = "\n".join(high_lights)

                    # 父节点代表了大局，人为赋予一个极高的合成得分 (10.0)，确保它排在最前面
                    final_nodes.append(NodeWithScore(node=parent_node, score=10.0))
                    merged_parent_ids.add(parent_id)
                except Exception as e:
                    print(f"合并父节点 {parent_id} 失败: {e}")

        # ==== 根据 query_type 决定去留 ====
        for n in reranked_leaf_nodes:
            parent_id = n.node.parent_node.node_id if n.node.parent_node else None

            if parent_id in merged_parent_ids:
                if query_type == "mixed":
                    final_nodes.append(n)
                elif query_type == "macro":
                    pass
            else:
                final_nodes.append(n)

        final_nodes.sort(key=lambda x: x.score, reverse=True)
        return final_nodes

    # =====================================================================
    # 根据 ID 精确读取内容
    # =====================================================================
    def get_node_content_by_id(self, node_id: str) -> str:
        """
        根据 Agent 提供的 node_id，直接从数据库提取全量原文。
        """
        if not self.index: self.initialize()

        try:
            node = self.index.docstore.get_node(node_id)
            return node.get_content()
        except Exception:
            return f"错误：未能在仓库中找到 ID 为 {node_id} 的文档片段。"

    # =====================================================================
    # 上下文双向扩展
    # =====================================================================
    def navigate_context(self, node_id: str, direction: str = "next") -> str:
        """
        利用 LlamaIndex 的 Node Relationship 寻找前后的片段。
        """
        if not self.index: self.initialize()

        try:
            node = self.index.docstore.get_node(node_id)

            target_id = None
            if direction == "next" and node.next_node:
                target_id = node.next_node.node_id
            elif direction == "prev" and node.prev_node:
                target_id = node.prev_node.node_id

            if target_id:
                return self.index.docstore.get_node(target_id).get_content()
            return f"已经到达文档的{'末尾' if direction == 'next' else '开头'}，无法继续扩展。"

        except Exception:
            return "扩展上下文失败。"

kb_handler = EmeiKnowledgeBase()