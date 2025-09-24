# 使用官方 Python 运行时作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 将依赖文件复制到工作目录
COPY requirements.txt .

# 安装依赖
# --no-cache-dir: 不缓存包，减小镜像体积
# --upgrade pip: 升级pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 将项目所有文件复制到工作目录
COPY . .

# 暴露 FastAPI 应用运行的端口
EXPOSE 8000

# 运行 FastAPI 应用的命令
# uvicorn app.api:app --host 0.0.0.0 --port 8000
# app.api:app 指的是 app 目录下的 api.py 文件中的 app 实例
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]