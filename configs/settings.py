import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from llama_index.embeddings.openai import OpenAIEmbedding
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    BASE_DIR: Path = BASE_DIR

    volcengine_API_KEY: str
    aliyun_API_KEY: str
    MYSQL_PASSWORD: str
    QWEATHER_API_KEY: str
    AMAP_API_KEY: str
    TAVILY_API_KEY: str = ""

    DATA_PATH: Path = BASE_DIR / "data"
    _config: dict = {}

    @property
    def amap_base_url(self):
        return (
            self._config.get("tools", {})
            .get("amap", {})
            .get("base_url", "https://restapi.amap.com/v3")
        )

    @property
    def qweather_base_url(self):
        return (
            self._config.get("tools", {})
            .get("qweather", {})
            .get("base_url", "https://m42vhfja65.re.qweatherapi.com")
        )

    @property
    def docstore_path(self):
        return self.BASE_DIR / "storage" / "docstore"

    def __init__(self, **values):
        super().__init__(**values)
        yaml_path = BASE_DIR / "configs" / "model_config.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def _get_llm_config(self, model_key):
        """内部复用逻辑：根据 key 获取具体配置并注入 API Key"""
        conf = self._config["llm_models"]["providers"][model_key].copy()
        if "dashscope" in conf["base_url"] or "aliyuncs" in conf["base_url"]:
            conf["api_key"] = self.aliyun_API_KEY
        else:
            conf["api_key"] = self.volcengine_API_KEY
        return conf

    # ==========================================
    # 动态 LLM 路由暴露层 (映射 YAML 的配置)
    # ==========================================
    @property
    def llm(self):
        return self._get_llm_config(self._config["llm_models"]["default"])

    @property
    def classifier_llm(self):
        """入库分类专用"""
        selected = self._config["llm_models"].get(
            "classifier", self._config["llm_models"]["default"]
        )
        return self._get_llm_config(selected)

    @property
    def extractor_llm(self):
        """元数据提取专用"""
        selected = self._config["llm_models"].get(
            "extractor", self._config["llm_models"]["default"]
        )
        return self._get_llm_config(selected)

    @property
    def memory_llm(self):
        """Mem0 记忆提炼专用"""
        selected = self._config["llm_models"].get(
            "memory", self._config["llm_models"]["default"]
        )
        return self._get_llm_config(selected)

    @property
    def image_llm(self):
        """文生图 API 专用"""
        selected = self._config["llm_models"].get(
            "image", self._config["llm_models"]["default"]
        )
        return self._get_llm_config(selected)

    @property
    def planner_llm(self):
        """专门用于任务规划与拆解 (Large 级别)"""  #
        selected = self._config["llm_models"].get(
            "planner", self._config["llm_models"]["default"]
        )
        return self._get_llm_config(selected)

    @property
    def tool_llm(self):
        """专门用于工具参数提取 (Medium 级别)"""  #
        selected = self._config["llm_models"].get(
            "tool", self._config["llm_models"]["default"]
        )
        return self._get_llm_config(selected)

    @property
    def summarizer_llm(self):
        """专门用于最终答案汇总 (Small 级别)"""  #
        selected = self._config["llm_models"].get(
            "summarizer", self._config["llm_models"]["default"]
        )
        return self._get_llm_config(selected)

    @property
    def sql_llm(self):
        """专门用于生成 SQL (Large 级别)"""  #
        selected = self._config["llm_models"].get(
            "sql", self._config["llm_models"]["default"]
        )
        return self._get_llm_config(selected)

    @property
    def rag_llm(self):
        """供 RAG 内部高频 ReAct 试探使用 (Medium 级别)"""
        selected = self._config["llm_models"].get(
            "rag", self._config["llm_models"]["default"]
        )
        return self._get_llm_config(selected)

    @property
    def rag_retrieve(self):
        """供 RAG 内部高频提取 (Small 级别)"""
        selected = self._config["llm_models"].get(
            "rag_retrieve", self._config["llm_models"]["default"]
        )
        return self._get_llm_config(selected)

    @property
    def grader_llm(self):
        """供独立判官审查幻觉使用 (Large 级别)"""
        selected = self._config["llm_models"].get(
            "grader", self._config["llm_models"]["default"]
        )
        return self._get_llm_config(selected)

    @property
    def reranker_llm(self):
        """供独立判官重新排序使用 (Medium 级别)"""
        selected = self._config["llm_models"].get(
            "reranker", self._config["llm_models"]["default"]
        )
        return self._get_llm_config(selected)

    # ==========================================
    # 其他配置层
    # ==========================================
    @property
    def embed(self):
        default = self._config["embedding_models"]["default"]
        return self._config["embedding_models"]["providers"][default]

    @property
    def embed_model(self):
        conf = self.embed
        return OpenAIEmbedding(
            api_key=self.aliyun_API_KEY,
            api_base=conf["base_url"],
            model_name=conf["model_id"],
            embed_batch_size=conf.get("batch_size", 10),
        )

    @property
    def mysql_url(self):
        db = self._config["database"]
        return f"mysql+pymysql://root:{self.MYSQL_PASSWORD}@{db['host']}:{db['port']}/{db['db_name']}"

    @property
    def chroma_config(self):
        conf = self._config["database"]["vector_db"]
        conf["absolute_path"] = str(BASE_DIR / conf["persist_path"])
        return conf

    @property
    def tools_config(self):
        return self._config.get("tools", {})

    @property
    def data_cleaning_config(self):
        return self.tools_config.get("data_cleaning", {})

    def _deep_merge(self, base: dict, override: dict) -> dict:
        result = base.copy()
        for k, v in (override or {}).items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    @property
    def ingestion_config(self):
        conf = self._config.get("ingestion", {})
        defaults = {
            "num_workers": 1,
            "splitter": {"type": "recursive", "chunk_size": 512, "chunk_overlap": 50},
            "extractors": {
                "enable_title": True,
                "title_nodes": 5,
                "enable_summary": True,
                "enable_entity": True,
                "entity_nodes": 5,
                "enable_qa": True,
                "qa_nodes": 5,
            },
        }
        return self._deep_merge(defaults, conf)

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")


settings = Settings()
