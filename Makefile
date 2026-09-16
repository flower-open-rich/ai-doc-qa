.PHONY: install dev run lint format type-check test test-cov docker-build docker-up clean

# 一键安装依赖（开发环境）
install:
	pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装开发依赖（含 ruff/mypy/pre-commit）
dev:
	pip install -r requirements.txt ruff mypy pytest pre-commit -i https://pypi.tuna.tsinghua.edu.cn/simple
	pre-commit install

# 启动开发服务器（带热重载）
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 代码检查（lint）
lint:
	ruff check .

# 自动格式化代码
format:
	ruff format .

# 类型检查
type-check:
	mypy app/

# 跑测试
test:
	pytest tests/ -v

# 跑测试 + 覆盖率报告
test-cov:
	pytest tests/ --cov=app --cov-report=term-missing --cov-report=html

# Docker 构建镜像
docker-build:
	docker build -t nuc-qa .

# Docker 一键启动
docker-up:
	docker-compose up -d

# 清理缓存文件
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
