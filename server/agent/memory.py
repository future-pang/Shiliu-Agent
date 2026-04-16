import os
from mem0 import Memory
from configs.settings import settings

def init_memory():
    """底层的原始初始化逻辑"""
    chroma_path = str(settings.BASE_DIR / "storage" / "mem0_chroma")            # ChromaDB 向量库
    history_db_path = str(settings.BASE_DIR / "storage" / "mem0_history.db")    # SQLite 历史库
    embed_conf = settings.embed

    os.environ["OPENAI_API_KEY"] = settings.aliyun_API_KEY
    os.environ["OPENAI_BASE_URL"] = embed_conf['base_url']

    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": settings.memory_llm['model_id'],
                "api_key": settings.memory_llm['api_key'],
                "openai_base_url": settings.memory_llm['base_url'],
                "temperature": 0.1,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": embed_conf['model_id']}
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "shiliu_long_term_memory",
                "path": chroma_path,
            }
        },
        "history_db_path": history_db_path,
    }
    return Memory.from_config(config)

class MemoryManager:
    def __init__(self):
        self._memory = None

    @property
    def instance(self):
        """
        懒加载：只有在第一次访问实例时，才会触发真正的 init_memory()
        """
        if self._memory is None:
            print("[Memory] 正在按需启动长期记忆中心...")
            self._memory = init_memory()
        return self._memory

    # ==== 添加对话到记忆库 ====
    def add(self, *args, **kwargs):
        return self.instance.add(*args, **kwargs)

    # ==== 语义搜索相关记忆 ====
    def search(self, *args, **kwargs):
        return self.instance.search(*args, **kwargs)

    # ==== 获取用户的所有核心记忆 ====
    def get_all(self, *args, **kwargs):
        return self.instance.get_all(*args, **kwargs)

shiliu_memory = MemoryManager()