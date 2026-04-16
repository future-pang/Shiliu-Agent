"""
全局 Prompt 缓存模块

使用 @lru_cache 保证整个进程生命周期内只读一次 YAML 磁盘文件。
原先每个节点每次调用都会触发 open() + yaml.safe_load()，
改为此模块后，磁盘 I/O 成本降为 0（首次启动后）。
"""
import yaml
from functools import lru_cache
from pathlib import Path

from configs.settings import settings
@lru_cache(maxsize=1)
def load_all_prompts() -> dict:
    """
    加载并缓存 prompts.yaml 的全部内容。
    lru_cache 保证整个进程只执行一次磁盘读取 + YAML 解析。
    """
    prompt_path = settings.BASE_DIR / "configs" / "prompts.yaml"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            print(f"[PromptLoader]prompts.yaml 已加载并缓存（共 {len(data)} 个 Prompt 块）")
            return data
    except Exception as e:
        print(f"[PromptLoader]读取 prompts.yaml 失败，返回空字典: {e}")
        return {}


def get_prompt(key: str, sub_key: str = "system_prompt") -> str:
    """
    便捷获取指定 Prompt。
    :param key:     prompts.yaml 中的顶级键名，如 'planner'、'synthesizer'
    :param sub_key: 子键名，默认 'system_prompt'
    :return:        Prompt 字符串，找不到时返回空字符串
    """
    return load_all_prompts().get(key, {}).get(sub_key, "")
