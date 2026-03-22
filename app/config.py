import os
from dotenv import load_dotenv

load_dotenv()

# 数据库（默认连接到远程服务器）
DB_HOST     = os.getenv("DB_HOST", "49.234.182.250")
DB_PORT     = int(os.getenv("DB_PORT", 133075))
DB_NAME     = os.getenv("DB_NAME", "healthineersvisualization")
DB_USER     = os.getenv("DB_USER", "root1")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# 服务
PORT      = int(os.getenv("PORT", 8000))
MODEL_DIR = os.getenv("MODEL_DIR", "models/")
DATA_DIR  = os.getenv("DATA_DIR", "data/")

# 模型更新触发阈值（新增记录数）
UPDATE_THRESHOLD = int(os.getenv("UPDATE_THRESHOLD", 30))
