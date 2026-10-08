# 中北大学智能问答助手 (NUC_QA)

> 基于 RAG（检索增强生成）的校园知识库问答系统，让 AI 真正"懂"中北大学。

[![CI](https://github.com/flower-open-rich/ai-doc-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/flower-open-rich/ai-doc-qa/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 项目简介

NUC_QA 是一个针对中北大学的智能问答系统。用户上传学校相关文档（招生简章、专业介绍、宿舍指南、校园规定等），系统自动建立向量索引，然后用户可以自然语言提问，AI 基于文档内容给出准确回答，并标注答案来源。

**核心价值**：解决通用大模型"不懂具体学校"的问题——通过 RAG 让 LLM 答得准、答得有依据。

## ✨ 功能特性

- 📄 **多格式文档支持**：PDF / Markdown / Word / 纯文本
- 🔍 **语义检索**：基于向量数据库的语义匹配，比关键词检索准确得多
- 🎯 **相关性过滤**：按相似度阈值剔除无关片段，减少 LLM 被噪声干扰
- 💬 **多轮对话**：支持上下文连续问答（session 会话隔离 + 自动淘汰）
- 📍 **引用溯源**：每个答案标注原文出处
- 🚀 **流式输出**：SSE 实时返回，体验像 ChatGPT
- 💰 **零成本可跑**：对话用智谱免费模型 `glm-4-flash`，向量化用本地开源模型，全程不花钱
- 🐳 **一键部署**：Docker Compose 部署，开箱即用

## 🛠️ 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| Web 框架 | FastAPI | 高性能异步框架，自带 OpenAPI 文档 |
| LLM 编排 | LangChain | 业界主流 RAG 框架 |
| 对话模型 | 智谱 GLM-4-Flash | 国产模型，**免费**，学生党友好 |
| Embedding | BAAI/bge-small-zh-v1.5 | 本地开源中文向量模型，免费、离线可用 |
| 向量数据库 | Chroma | 轻量级，本地存储，无需独立部署 |
| 前端 | 原生 HTML/JS | 单文件零构建，由 FastAPI 直接托管 |
| 部署 | Docker + docker-compose | 容器化，可移植 |
| CI/CD | GitHub Actions | 自动跑 lint + 类型检查 + 测试 |
| 代码质量 | ruff + mypy + pre-commit | 现代化工具链 |

### 关于"对话模型和 Embedding 分开配置"

这是一个刻意的设计。智谱的免费额度只覆盖**对话模型**，而 **Embedding 是按量计费的**，
所以两者解耦后才能做到：**对话走云端免费模型 + 向量化走本地免费模型 = 全程零成本**。

想切换到云端 Embedding（效果更好、也不占本地算力）时，只改 `.env` 一行即可：

```bash
# 本地免费（默认）
EMBEDDING_PROVIDER=local

# 换成智谱（需充值）或 OpenAI
EMBEDDING_PROVIDER=zhipu
EMBEDDING_MODEL=embedding-3
```

> ⚠️ 换了 Embedding 模型后**必须重建向量库**，因为不同模型的向量维度不同：
> `python -m app.services.ingest_service`
> 服务启动时会自动做维度一致性自检，不一致会在日志里明确报错。

## 🏗️ 架构流程

```
[用户提问]
   ↓
[问题向量化]  ← 本地 bge-small-zh（免费）
   ↓
[向量库 Top-K 检索] ← [文档库（已分块+向量化）]
   ↓
[拼接 Prompt: 检索片段 + 对话历史 + 用户问题]
   ↓
[LLM 生成回答] ← 智谱 glm-4-flash（免费）
   ↓
[返回答案 + 引用来源]
```

## 📁 项目结构

```
NUC_QA/
├── app/                        # 后端代码
│   ├── api/                    # API 路由层（只做参数接收/返回，不写业务）
│   │   ├── routes_health.py    #   /health
│   │   ├── routes_chat.py      #   /chat（纯 LLM 对话）
│   │   ├── routes_qa.py        #   /qa（RAG 问答）
│   │   ├── routes_qa_stream.py #   /qa/stream（SSE 流式）
│   │   └── routes_documents.py #   /documents/*（上传/列表）
│   ├── core/                   # 核心逻辑
│   │   ├── config.py           #   pydantic-settings 配置中心
│   │   ├── llm.py              #   对话模型工厂（可切换厂商）
│   │   ├── embedding.py        #   向量模型工厂（local/zhipu/openai）
│   │   ├── vectorstore.py      #   Chroma 封装 + 维度自检
│   │   ├── document_loader.py  #   文档加载 + Markdown 清洗 + 分块
│   │   ├── session_store.py    #   多轮会话内存存储（TTL + LRU 淘汰）
│   │   └── logging.py          #   统一日志配置
│   ├── services/               # 业务服务层
│   │   ├── chat_service.py     #   纯对话业务
│   │   ├── qa_service.py       #   RAG 问答业务（检索 + Prompt 组装）
│   │   └── ingest_service.py   #   离线建库脚本
│   ├── models/schemas.py       # Pydantic 请求/响应模型
│   ├── static/index.html       # 前端页面（单文件，零构建）
│   └── main.py                 # FastAPI 入口
├── tests/                      # pytest 测试用例
├── data/
│   ├── raw_docs/               # 中北大学原始文档
│   └── chroma/                 # 向量库持久化（已 gitignore）
├── .github/workflows/ci.yml    # CI 配置
├── .env.example                # 环境变量模板
├── pyproject.toml              # ruff / mypy / pytest 配置
├── Dockerfile                  # 容器化构建文件
├── docker-compose.yml          # 一键启动
├── Makefile                    # 常用命令快捷方式
└── requirements.txt            # Python 依赖
```

## 🚀 快速开始

### 方式一：本地开发

```bash
# 1. 克隆仓库
git clone https://github.com/your-name/NUC_QA.git
cd NUC_QA

# 2. 创建虚拟环境
conda create -n nuc_qa python=3.11 -y
conda activate nuc_qa

# 3. 安装依赖
make install

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，填入智谱 API Key（去 https://open.bigmodel.cn/ 免费注册领取）

# 5. 建立向量索引（首次需要下载本地 Embedding 模型，约 90MB）
make ingest

# 6. 启动开发服务器
make run
```

访问 http://localhost:8000 即可。API 文档：http://localhost:8000/docs

### 方式二：Docker 一键启动

```bash
cp .env.example .env
# 编辑 .env 填入 API key
make docker-up
```

## 📚 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/chat` | 纯 LLM 对话（GET，便于浏览器直接测试） |
| POST | `/chat` | 纯 LLM 对话（支持多轮 `session_id`） |
| GET | `/qa` | RAG 问答（GET，便于测试） |
| POST | `/qa` | RAG 问答（支持多轮 `session_id`） |
| POST | `/qa/stream` | RAG 问答流式输出（SSE） |
| POST | `/documents/upload` | 上传文档并自动建索引 |
| GET | `/documents/list` | 列出已入库文档 + 向量库总条数 |
| GET | `/` | 前端聊天页面 |
| GET | `/docs` | 自动生成的 OpenAPI 交互文档 |

### SSE 事件协议（`/qa/stream`）

流式接口按顺序推送以下事件，每条格式为 `data: {json}\n\n`：

| `type` | `data` | 含义 |
|--------|--------|------|
| `session_id` | 字符串 | 会话 ID，后续请求带上它即可延续上下文 |
| `sources` | 字符串数组 | 本次回答引用的来源文件 |
| `token` | 字符串 | 增量文本片段 |
| `done` | `null` | 生成结束 |
| `error` | 字符串 | 出错信息 |

请求示例：

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "中北大学宿舍几人间？"}'
```

## 🧪 测试

```bash
make test        # 跑全部测试
make lint        # ruff 静态检查
make type-check  # mypy 类型检查
```

测试刻意不依赖真实 LLM/Embedding API（用 stub 替身），所以**没有 API Key 也能跑通**，
这也是 CI 里能直接执行的原因。

## 🗺️ 开发路线图

- [x] 项目脚手架与工程化配置
- [x] FastAPI 后端骨架 + 健康检查
- [x] 接入 LLM（智谱 GLM-4-Flash）
- [x] 文档处理（PDF / Markdown / Word 解析 + 分块）
- [x] RAG 核心（Embedding + 向量检索）
- [x] 问答接口（含引用溯源）
- [x] 流式输出（SSE）
- [x] 多轮对话（会话存储 + 历史裁剪）
- [x] 前端界面（单文件零构建）
- [x] 单元测试 + CI
- [ ] 部署上线
- [ ] 检索优化（混合检索 BM25 + 向量、重排序 rerank）

## 🤝 贡献指南

欢迎提 Issue 和 PR。提交前请运行 `make lint` 和 `make test` 确保代码风格与测试通过。

## 📄 License

[MIT](LICENSE)
