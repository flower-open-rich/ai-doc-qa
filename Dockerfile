# Dockerfile: 容器化部署 - 一行命令把项目打包成可移植镜像
# 构建：docker build -t nuc-qa .
# 运行：docker run -p 8000:8000 --env-file .env nuc-qa
FROM python:3.11-slim

# 设置时区 + 不写 .pyc + 实时输出日志
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 先复制依赖，利用 Docker 缓存加速
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY app/ ./app/
COPY data/ ./data/

EXPOSE 8000

# 启动命令（生产环境去掉 --reload）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
