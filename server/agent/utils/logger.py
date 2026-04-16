import logging
import sys
import os
from logging.handlers import RotatingFileHandler

# 1. 确保日志目录存在
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 2. 核心配置：拦截所有日志写入文件，不在控制台显示
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "agent_run.log"),
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding="utf-8",
)
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)

# 获取根 Logger 并清空默认终端输出，绑定文件输出
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
root_logger.addHandler(file_handler)

# 特别处理 httpx 的日志级别，防止底层的 HTTP Request 刷屏日志文件
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def print_status(msg: str):
    """
    终端输出简洁提示。
    使用 \r 覆盖上一行，避免控制台被刷屏。
    """
    # \033[K 用于清除光标到行尾的内容
    sys.stdout.write(f"\r\033[K⏳ 状态: {msg}")
    sys.stdout.flush()
    # 同时也记录到文件日志中
    logging.info(f"STATUS -> {msg}")
