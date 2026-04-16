import yaml
import asyncio
import chromadb
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from sklearn.cluster import AgglomerativeClustering

from langchain_openai import ChatOpenAI
from llama_index.core.schema import BaseNode
from llama_index.llms.openai_like import OpenAILike
from langchain_core.prompts import ChatPromptTemplate
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

from configs.settings import settings
from server.knowledge_base.reader import emei_reader
from server.knowledge_base.components import get_splitter, get_extractors
from server.knowledge_base.db_utils import get_or_create_file_record, update_file_status


# =====================================================================
# 定义大模型分类的结构化输出
# =====================================================================
class DocStrategyOutput(BaseModel):
    strategy: Literal["hierarchical", "markdown", "semantic", "recursive"] = Field(
        description="最适合该文档的切分策略"
    )
    reasoning: str = Field(description="选择该策略的简短理由")

# =====================================================================
# 知识库加工中心
# =====================================================================
class EmeiIngestionManager:
    def __init__(self):
        self.pipeline: Optional[IngestionPipeline] = None
        self.processed_nodes: List[BaseNode] = []

        llm_config = settings.llm
        self.llm = OpenAILike(
            model=llm_config["model_id"],
            api_key=llm_config["api_key"],
            api_base=llm_config["base_url"],
            is_chat_model=True,
        )

        self.classifier_llm = ChatOpenAI(
            model=settings.classifier_llm["model_id"],
            api_key=settings.classifier_llm["api_key"],
            base_url=settings.classifier_llm.get("base_url"),
            temperature=0.0,
        ).with_structured_output(DocStrategyOutput)

    def _get_classifier_prompt(self) -> str:
        try:
            prompt_path = Path(settings.BASE_DIR) / "configs" / "prompts.yaml"
            with open(prompt_path, "r", encoding="utf-8") as f:
                return (
                    yaml.safe_load(f).get("doc_classifier", {}).get("system_prompt", "")
                )
        except Exception:
            return "请从 markdown, semantic, recursive, hierarchical 中选择一个输出。"

    async def _classify_document_async(self, doc_preview: str) -> str:
        """
        执行异步推理的底层方法
        """
        system_prompt = self._get_classifier_prompt()
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "【文档片段预览】\n{preview}")]
        )
        try:
            chain = prompt | self.classifier_llm
            result: DocStrategyOutput = await chain.ainvoke({"preview": doc_preview})
            print(
                f"[LLM 诊断完成]: 选用 `{result.strategy}` (理由: {result.reasoning})"
            )
            return result.strategy
        except Exception as e:
            print(f"[诊断异常]，降级为常规层级切分 hierarchical。错误: {e}")
            return "hierarchical"

    async def _classify_single_doc_async(self, doc):
        """
        异步诊断单份文档策略的包装器
        """
        file_name = doc.metadata.get("file_name", "未知文件")
        print(f"发起并发分析: {file_name}")
        preview = doc.text[:800]
        strategy = await self._classify_document_async(preview)
        return strategy, doc

    def _build_pipeline_for_strategy(self, strategy_name: str):
        """
        动态为不同批次的文档组装流水线
        """
        conf = settings.ingestion_config
        transformations = []

        # ==== 动态获取 LLM 指定的切分器 ====
        splitter_conf = conf.get("splitter", {})
        splitter_conf["type"] = strategy_name  # 覆写配置文件中的默认切分器类型
        splitter = get_splitter(splitter_conf, settings.embed_model)
        transformations.append(splitter)

        # ==== 提取器 ====
        extractor_conf = settings.extractor_llm
        extractor_llm_instance = OpenAILike(
            model=extractor_conf["model_id"],
            api_key=extractor_conf["api_key"],
            api_base=extractor_conf["base_url"],
            is_chat_model=True,
            timeout=60.0,
        )
        ingestion_cfg = settings.ingestion_config
        extractor_cfg = ingestion_cfg.get("extractors", {})

        extractors = get_extractors(llm=extractor_llm_instance, config=extractor_cfg)
        transformations.extend(extractors)

        # 向量化模型
        transformations.append(settings.embed_model)

        self.pipeline = IngestionPipeline(transformations=transformations)

    async def _build_semantic_tree_async(
        self, leaf_nodes: List[BaseNode], file_name: str
    ) -> List[BaseNode]:
        """
        核心创新：文档级语义主题聚类与建树
        将平级的叶子节点，根据 (标题+摘要) 的向量距离进行聚类，并使用 LLM 生成宏观父节点。
        """
        if len(leaf_nodes) < 2:
            return leaf_nodes  # 节点太少，没必要建树

        print(
            f"  [建树引擎] 正在为 {file_name} 的 {len(leaf_nodes)} 个叶子节点进行语义聚类..."
        )

        # ==== 提取特征：标题+摘要 作为聚类依据 ====
        texts_to_embed = []
        for node in leaf_nodes:
            title = node.metadata.get("document_title", "未知标题")
            summary = node.metadata.get("section_summary", "无摘要")
            texts_to_embed.append(f"标题：{title}\n摘要：{summary}")

        # === 获取临时向量 使用 aget_text_embedding_batch 异步批量获取 ====
        embeddings = await settings.embed_model.aget_text_embedding_batch(
            texts_to_embed
        )

        # ==== 层次聚类 distance_threshold=0.5 是个经验值，余弦距离越小表示越相似 ====
        clustering_model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=0.5, metric="cosine", linkage="average"
        )
        cluster_labels = clustering_model.fit_predict(embeddings)

        # ==== 按类别分组 ====
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(leaf_nodes[idx])

        parent_nodes = []

        # ==== 让 LLM 根据微观摘要生成宏观父节点 ====
        for label, group_nodes in clusters.items():
            if len(group_nodes) < 2:
                continue  # 孤立节点不建父节点

            # ==== 将这组叶子的微观摘要拼接起来 ====
            combined_summaries = "\n".join(
                [f"- {n.metadata.get('section_summary', '')}" for n in group_nodes]
            )

            # ==== 让大模型写宏观总结 ====
            prompt = (
                f"你是一个高级知识提取器。以下是同一章节内几个段落的局部摘要：\n{combined_summaries}\n"
                f"请严格基于以上内容，撰写一段150字以内的宏观全局总结，不要加任何废话。"
            )

            try:
                response = await self.llm.acomplete(prompt)  # 使用默认model
                macro_summary = response.text
            except Exception as e:
                print(f"  [建树引擎] 宏观摘要生成失败: {e}")
                macro_summary = "本章节包含多个相关子主题，具体内容见下属细节。"

            # ==== 创建父节点 ====
            parent_node = TextNode(text=macro_summary)

            # ==== 继承第一个叶子的基础元数据（确保有归属） ====
            if not group_nodes: continue
            base_metadata = group_nodes[0].metadata.copy()
            # ==== 覆写父节点特有的元数据 ====
            base_metadata["document_title"] = (
                base_metadata.get("document_title", "") + " (宏观概述)"
            )
            base_metadata["section_summary"] = macro_summary  # 父节点自己就是宏观摘要
            parent_node.metadata = base_metadata

            # ==== 落实父节点的“霸王条款” ====
            parent_node.excluded_llm_metadata_keys = [
                "section_summary",
                "excerpt_keywords",
                "questions_this_excerpt_can_answer",
            ]

            parent_node.excluded_embed_metadata_keys = [
                "section_summary",
                "questions_this_excerpt_can_answer",
            ]

            # ==== 双向绑定指针 ====
            parent_node.relationships[NodeRelationship.CHILD] = [
                RelatedNodeInfo(node_id=child.node_id) for child in group_nodes
            ]
            for child in group_nodes:
                child.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=parent_node.node_id
                )

            parent_nodes.append(parent_node)

        print(
            f"  [建树引擎] 成功！将 {len(leaf_nodes)} 个叶子聚合成了 {len(parent_nodes)} 棵树 (父节点)。"
        )

        # ==== 返回：父节点 + 原始的叶子节点 ====
        return parent_nodes + leaf_nodes

    def _persist_nodes(self):
        conf = settings.chroma_config
        docstore_dir = str(settings.docstore_path)

        if not self.processed_nodes:
            print("\n4. 没有可持久化的节点，跳过存储。")
            return

        print("\n4. 正在提取叶子节点存入 ChromaDB，并持久化图谱...")

        db = chromadb.PersistentClient(path=conf["absolute_path"])
        chroma_collection = db.get_or_create_collection(conf["collection_name"])
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # 只存没有子节点的叶子块
        leaf_nodes = get_leaf_nodes(self.processed_nodes)
        print(f"提取到 {len(leaf_nodes)} 个叶子节点用于向量检索")

        # 初始化 StorageContext
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 把所有节点都加入 docstore（包含父子关系）
        storage_context.docstore.add_documents(self.processed_nodes)

        # 存入向量库（只放叶子节点）
        if leaf_nodes:
            VectorStoreIndex(
                leaf_nodes,
                storage_context=storage_context,
                embed_model=settings.embed_model,
            )
        else:
            print("未发现叶子节点，跳过向量入库。")

        storage_context.persist(persist_dir=docstore_dir)
        print(f"持久化成功！Chroma 存了向量，{docstore_dir} 存了关系图谱。")

    async def run_full_ingestion_async(self):
        print("\n" + "=" * 40)
        print("知识库 Agent 化加工流启动")

        raw_docs = emei_reader.load_data()
        docs_to_process = []
        doc_id_map = {}
        processed_files = set()

        print("对照 MySQL 进行查重...")
        for doc in raw_docs:
            f_path = doc.metadata.get("file_path")
            if f_path in processed_files:
                if f_path in doc_id_map:
                    doc.metadata["db_id"] = doc_id_map[f_path]
                    docs_to_process.append(doc)
                continue

            processed_files.add(f_path)
            db_record, should_process = get_or_create_file_record(f_path)

            if should_process:
                doc.metadata["db_id"] = db_record.id
                doc_id_map[f_path] = db_record.id
                docs_to_process.append(doc)
                update_file_status(db_record.id, "PROCESSING")

        if not docs_to_process:
            print("所有文件都已最新，无需处理。")
            return

        # ==== LLM 智能分拣文档 ====
        print(
            f"\n 正在【并发】交由大模型诊断 {len(docs_to_process)} 份文档的最佳切分策略..."
        )
        strategy_groups = {
            "markdown": [],
            "semantic": [],
            "recursive": [],
            "hierarchical": [],
        }

        # ==== 并发启动所有文档的诊断任务 ====
        tasks = [self._classify_single_doc_async(doc) for doc in docs_to_process]
        results = await asyncio.gather(*tasks)

        for strategy, doc in results:
            if strategy in strategy_groups:
                strategy_groups[strategy].append(doc)
            else:
                # 对于乱答的 LLM，统一降级到 hierarchical 兜底
                print(
                    f"警告: LLM 返回了未定义的策略 '{strategy}'，已强制降级为 hierarchical 兜底。"
                )
                strategy_groups["hierarchical"].append(doc)

        print("\n 开始按策略分组分批加工...")
        num_workers = settings.ingestion_config.get("num_workers", 4)
        all_nodes = []

        try:
            for strategy, docs in strategy_groups.items():
                if not docs:
                    continue

                print(f"\n启动 [{strategy}] 切分流水线，处理 {len(docs)} 份文档...")
                self._build_pipeline_for_strategy(strategy)

                # ==== LlamaIndex 内部切分与提取 (生成平级节点) ====
                nodes = self.pipeline.run(
                    documents=docs, num_workers=num_workers, show_progress=True
                )

                # ==== 落实元数据“霸王条款” (隔离策略) ====
                for node in nodes:
                    # 对 Agent LLM 隐藏冗长元数据 (防 Token 爆炸)
                    node.excluded_llm_metadata_keys = [
                        "section_summary",
                        "excerpt_keywords",
                        "questions_this_excerpt_can_answer",
                    ]
                    # 对 Embedding 模型隐藏冗长元数据 (保证向量纯净，拒绝喧宾夺主)
                    node.excluded_embed_metadata_keys = [
                        "section_summary",
                        "questions_this_excerpt_can_answer",
                    ]
                    # Tip：不被排除的，都会被 BM25 默认吃掉，成为搜索诱饵！

                # ==== 按文档分组，触发“语义聚类建树” ====
                # 如果已经是 hierarchical 切分，它自带树结构，跳过
                if strategy != "hierarchical":
                    # 按所属文件将 nodes 分组 (避免跨文件错误聚类)
                    nodes_by_file = {}
                    for n in nodes:
                        f_id = n.metadata.get("db_id", "unknown")
                        if f_id not in nodes_by_file:
                            nodes_by_file[f_id] = []
                        nodes_by_file[f_id].append(n)

                    final_tree_nodes = []
                    # 逐个文件建立森林
                    for f_id, doc_nodes in nodes_by_file.items():
                        if not doc_nodes: continue
                        f_name = doc_nodes[0].metadata.get("file_name", "未知文件")
                        # 触发上面写的聚类建树算法
                        tree_nodes = await self._build_semantic_tree_async(
                            doc_nodes, f_name
                        )
                        final_tree_nodes.extend(tree_nodes)

                    all_nodes.extend(final_tree_nodes)
                else:
                    # 如果是 hierarchical，直接加入
                    all_nodes.extend(nodes)

            # 更新 processed_nodes，准备存入底层的 Docstore 和 Chroma
            self.processed_nodes = all_nodes
            self._persist_nodes()

            from collections import Counter

            file_node_counts = Counter([n.metadata["db_id"] for n in all_nodes])
            for f_path, db_id in doc_id_map.items():
                count = file_node_counts.get(db_id, 0)
                update_file_status(db_id, "SUCCESS", node_count=count)
                print(f"文件 ID {db_id} 入库成功，生成片段数: {count}")

        except Exception as e:
            print(f"加工流出错了: {e}")
            for f_path, db_id in doc_id_map.items():
                update_file_status(db_id, "FAILED", error=str(e))

ingestion_manager = EmeiIngestionManager()

if __name__ == "__main__":
    from server.db.models import init_db

    init_db()
    asyncio.run(ingestion_manager.run_full_ingestion_async())
