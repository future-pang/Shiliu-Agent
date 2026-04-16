import yaml
from typing import List
from configs.settings import settings, BASE_DIR
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
    LangchainNodeParser,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llama_index.core.extractors import (
    TitleExtractor,
    SummaryExtractor,
    KeywordExtractor,
    BaseExtractor,
    QuestionsAnsweredExtractor,
)

prompt_path = BASE_DIR / "configs" / "prompts.yaml"


def load_prompts():
    if not prompt_path.exists():
        raise FileNotFoundError(f"找不到 Prompt 配置文件: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =====================================================================
# 切分器选择
# =====================================================================
def get_splitter(config: dict, embed_model=None):
    """
    根据 YAML 配置返回对应的切分器实例
    """
    s_type = config.get("type", "hierarchical")
    size = config.get("chunk_size", 512)
    overlap = config.get("chunk_overlap", 50)

    print(f"正在初始化切分器: {s_type} (Size: {size})")

    # 语义切分器
    if s_type == "semantic":
        if not embed_model:
            raise ValueError("使用语义切分必须传入 embed_model")
        sem_conf = config.get("semantic", {})
        return SemanticSplitterNodeParser(
            buffer_size=sem_conf.get("buffer_size", 1),
            breakpoint_percentile_threshold=sem_conf.get(
                "breakpoint_percentile_threshold", 95
            ),
            embed_model=embed_model,
        )

    # Markdown 切分器
    elif s_type == "markdown":
        return MarkdownNodeParser()

    # 递归字符切分器 (LangChain)
    elif s_type == "recursive":
        lc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap
        )
        return LangchainNodeParser(lc_splitter=lc_splitter)

    else:
        # 兜底方案：层级切分（固定构建树形结构，丢弃外部的死板节点）
        print(f"正在初始化父子层级切分器 (1024 -> 512 -> 128)")
        return HierarchicalNodeParser.from_defaults(chunk_sizes=[1024, 512, 128])


# =====================================================================
# 提取器选择
# =====================================================================
def get_extractors(config: dict, llm) -> List[BaseExtractor]:
    """
    根据 YAML 配置和 prompt.yaml 返回提取器列表
    """
    extractors = []
    all_prompts = load_prompts()

    print("正在组装元数据提取器...")

    # ==== 标题提取器 ====
    if config.get("enable_title"):
        t_prompt = all_prompts["title_extractor"]["template"]
        extractors.append(
            TitleExtractor(
                nodes=config.get("title_nodes", 5),
                llm=llm,
                node_template=t_prompt,
                combine_template=t_prompt,
            )
        )
        print("   [+] TitleExtractor (Title)")

    # ==== 摘要提取器 ====
    if config.get("enable_summary"):
        s_prompt = all_prompts["summary_extractor"]["template"]
        extractors.append(
            SummaryExtractor(
                summaries=["prev", "self"], llm=llm, prompt_template=s_prompt
            )
        )
        print("   [+] SummaryExtractor (Summary)")

    # ==== 实体/关键字提取器 ====
    if config.get("enable_entity"):
        e_prompt = all_prompts["entity_extractor"]["template"]
        extractors.append(
            KeywordExtractor(
                keywords=config.get("entity_nodes", 5),
                llm=llm,
                prompt_template=e_prompt,
            )
        )
        print("   [+] KeywordExtractor (Entity)")

    # ==== 问答提取器 ====
    if config.get("enable_qa"):
        q_prompt = all_prompts["qa_extractor"]["template"]
        extractors.append(
            QuestionsAnsweredExtractor(
                questions=config.get("qa_nodes", 3), llm=llm, prompt_template=q_prompt
            )
        )
        print("   [+] QuestionsAnsweredExtractor (QA)")

    return extractors
