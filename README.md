# 中北大学智能问答助手 (NUC_QA)

> 基于 RAG（检索增强生成）的校园知识库问答系统，让 AI 真正"懂"中北大学。

[![CI](https://github.com/your-name/NUC_QA/actions/workflows/ci.yml/badge.svg)](https://github.com/your-name/NUC_QA/actions/workflows/ci.yml)
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
- 💬 **多轮对话**：支持上下文连续问答
- 📍 **引用溯源**：每个答案标注原文出处，可点击查看
- 🚀 **流式输出**：SSE 实时返回，体验像 ChatGPT
- 🐳 **一键部署**：Docker Compose 部署，开箱即用

## 🛠️ 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| Web 框架 | FastAPI | 高性能异步框架，自带 OpenAPI 文档 |
| LLM 编排 | LangChain | 业界主流 RAG 框架 |
| LLM 模型 | 智谱 GLM-4 | 国产模型，便宜稳定，学生党友好 |
| Embedding | 智谱 embedding-2 | 中文向量化效果好 |
| 向量数据库 | Chroma | 轻量级，本地存储，无需独立部署 |
| 前端 | (待定) | React 或 Vue3 |
| 部署 | Docker + docker-compose | 容器化，可移植 |
| CI/CD | GitHub Actions | 自动跑 lint + test |
| 代码质量 | ruff + mypy + pre-commit | 现代化工具链 |

## 🏗️ 架构流程

```
[用户提问] 
   ↓
[问题向量化]
   ↓
[向量库 Top-K 检索] ← [文档库（已分块+向量化）]
   ↓
[拼接 Prompt: 检索片段 + 用户问题]
   ↓
[LLM 生成回答]
   ↓
[返回答案 + 引用来源]
```

## 📁 项目结构

```
NUC_QA/
├── app/                  # 后端代码
│   ├── api/              # API 路由层
│   ├── core/             # 核心逻辑（LLM、RAG、配置）
│   ├── models/           # Pydantic 数据模型
│   └── main.py           # FastAPI 入口
├── tests/                # 单元测试 + 集成测试
├── data/                 # 中北大学原始文档
├── vector_db/            # 向量数据库持久化
├── .github/workflows/    # CI 配置
├── .env.example          # 环境变量模板
├── pyproject.toml        # 代码质量工具配置
├── Dockerfile            # 容器化构建文件
├── docker-compose.yml    # 一键启动
├── Makefile              # 常用命令快捷方式
└── requirements.txt     # Python 依赖
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
# 编辑 .env 填入智谱 API key

# 5. 启动开发服务器
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
| POST | `/api/v1/upload` | 上传文档到知识库 |
| POST | `/api/v1/chat` | 单轮问答 |
| POST | `/api/v1/chat/stream` | 流式问答（SSE） |
| GET | `/api/v1/documents` | 列出已索引的文档 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除指定文档 |

## 🗺️ 开发路线图

- [x] 项目脚手架与工程化配置
- [ ] FastAPI 后端骨架 + 健康检查
- [ ] 接入 LLM（智谱 GLM-4）
- [ ] 文档处理（PDF/MD 解析 + 分块）
- [ ] RAG 核心（Embedding + 向量检索）
- [ ] 问答接口（含引用溯源）
- [ ] 流式输出（SSE）
- [ ] 前端界面
- [ ] 部署上线

## 🤝 贡献指南

欢迎提 Issue 和 PR。提交前请运行 `make lint` 确保代码风格统一。

## 📄 License

[MIT](LICENSE)
