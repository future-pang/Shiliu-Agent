# Emei Master

基于多智能体与 RAG 的文旅问答项目。

## 快速开始（pip）

1. 创建虚拟环境并激活

```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

2. 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. 复制并配置环境变量

在项目根目录新建 `.env` 文件，示例：

```env
# LLM / Embedding
volcengine_API_KEY=your_volcengine_api_key
aliyun_API_KEY=your_aliyun_api_key

# Database
MYSQL_PASSWORD=your_mysql_password

# MCP Tools
QWEATHER_API_KEY=your_qweather_api_key
AMAP_API_KEY=your_amap_api_key
TAVILY_API_KEY=your_tavily_api_key

# Optional: Redis (web_server.py)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
```

## 启动方式

```bash
# 交互问答
python main.py --mode chat

# 知识库灌装
python main.py --mode ingest

# Web 服务
python web_server.py
```

## 配置说明

- 业务配置集中在 `configs/model_config.yaml`
- 运行时环境变量从根目录 `.env` 读取（见 `configs/settings.py`）
- 请勿提交 `.env`、`storage/`、`data/` 等本地文件（已在 `.gitignore` 中忽略）
