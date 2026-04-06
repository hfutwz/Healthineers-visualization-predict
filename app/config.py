import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

# 数据库
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 3306))
DB_NAME     = os.getenv("DB_NAME", "healthineersvisualization")
DB_USER     = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# 密码中可能含 @、#、! 等特殊字符，必须 URL 编码后再拼接连接串
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# 服务
PORT      = int(os.getenv("PORT", 8000))
MODEL_DIR = os.getenv("MODEL_DIR", "models/")

# 模型更新触发阈值（新增记录数）
UPDATE_THRESHOLD = int(os.getenv("UPDATE_THRESHOLD", 30))
