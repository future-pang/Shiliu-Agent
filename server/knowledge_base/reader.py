import os
from datetime import datetime
from configs.settings import settings
from typing import List, Dict, Callable
from llama_index.core import SimpleDirectoryReader, Document

class EmeiReader:
    """
    工业级数据读取器：支持多格式识别与自动元数据注入
    """
    def __init__(self):
        self.data_path = settings.DATA_PATH

    def _custom_metadata_handler(self, file_path: str) -> Dict:
        """
        元数据注入器：在读取瞬间为文档打上”身份证“
        """
        stat = os.stat(file_path)
        category = os.path.basename(os.path.dirname(file_path))
        return {
            "file_name" : os.path.basename(file_path),
            "file_path" : file_path,
            "category": category,
            "file_size" : stat.st_size,
            "extension": os.path.splitext(file_path)[1],
            "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            "project": "Emei-Master"
        }

    def load_data(self) -> List[Document]:
        """
        核心方法：扫描 data 目录并返回标准 Document 列表
        """
        if not self.data_path.exists():
            print(f"目录 {self.data_path} 不存在，正在为您自动创建...")
            self.data_path.mkdir(parents=True, exist_ok=True)
            return []

        # 全能型SimpleDirectoryReader
        reader = SimpleDirectoryReader(
            input_dir=str(self.data_path),
            recursive=True,
            file_metadata=self._custom_metadata_handler
        )

        documents = reader.load_data()

        if not documents:
            print(f"目录 {self.data_path} 目前是空的，请放入文档。")
        else:
            print(f"Reader 成功加载了 {len(documents)} 个原始文档对象")
        return documents

emei_reader = EmeiReader()