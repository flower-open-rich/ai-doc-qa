.PHONY: help install dev ingest run lint format format-check type-check test test-cov check docker-build docker-up docker-down clean

# 默认目标：列出所有可用命令
help:
	@echo "可用命令："
	@echo "  make install      安装运行依赖"
	@echo "  make dev          安装开发依赖（含 ruff/mypy/pytest）并注册 pre-commit"
	@echo "  make ingest       把 data/raw_docs 的资料灌入向量库（首次需下载 Embedding 模型）"
	@echo "  make run          启动开发服务器（热重载）"
	@echo "  make lint         ruff 静态检查"
	@echo "  make format       ruff 自动格式化"
	@echo "  make type-check   mypy 类型检查"
	@echo "  make test         跑测试"
	@echo "  make test-cov     跑测试 + 覆盖率报告"
	@echo "  make check        lint + 类型检查 + 测试（提交前跑一遍）"
	@echo "  make docker-up    docker compose 一键启动"
	@echo "  make clean        清理缓存文件"

# 一键安装依赖（运行环境）
install:
	pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装开发依赖（含 ruff/mypy/pytest）并注册 pre-commit
dev:
	pip install -r requirements-dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
	pre-commit install

# 建立/重建向量索引
# 运行前请确认 .env 里的 EMBEDDING_PROVIDER 已配好；
# 换过 Embedding 模型必须重跑本命令，否则向量维度对不上。
ingest:
	python -m app.services.ingest_service

# 启动开发服务器（带热重载）
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 代码检查（lint）
lint:
	ruff check .

# 自动格式化代码
format:
	ruff format .

# 只检查格式，不修改文件（CI 用）
format-check:
	ruff format --check .

# 类型检查
type-check:
	mypy app/

# 跑测试
test:
	pytest tests/ -v

# 跑测试 + 覆盖率报告
test-cov:
	pytest tests/ --cov=app --cov-report=term-missing --cov-report=html

# 提交前完整检查
check: lint type-check test

# Docker 构建镜像
docker-build:
	docker build -t nuc-qa .

# Docker 一键启动
docker-up:
	docker compose up -d

# Docker 停止
docker-down:
	docker compose down

# 清理缓存文件（跨平台写法，不依赖 Unix 的 find/rm）
clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov', '.coverage']]"
