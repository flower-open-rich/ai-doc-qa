# Dockerfile: 容器化部署 - 一行命令把项目打包成可移植镜像
# 构建：docker build -t nuc-qa .
# 运行：docker run -p 8000:8000 --env-file .env -v "$(pwd)/data:/app/data" nuc-qa
FROM python:3.11-slim

# 设置时区 + 不写 .pyc + 实时输出日志
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 先复制依赖，利用 Docker 缓存加速
# 说明：这一步会装 torch（本地 Embedding 模型需要），镜像会比较大（约 1GB+）。
# 如果只想用云端 Embedding 来瘦身，可以把 EMBEDDING_PROVIDER 设为 zhipu，
# 并从 requirements.txt 里去掉 sentence-transformers / torch / langchain-huggingface。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY app/ ./app/
COPY data/ ./data/

EXPOSE 8000

# 健康检查：这里刻意不用 curl —— python:3.11-slim 基础镜像里没有 curl，
# 写了会导致容器永远处于 unhealthy 状态。用 Python 标准库探测最稳妥。
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"]

# 启动命令（生产环境去掉 --reload）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
