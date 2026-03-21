"""
数据服务：从 MySQL 读取训练所需数据
"""

from sqlalchemy import create_engine, text
from app.config import DATABASE_URL


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


TRAINING_SQL = text("""
    SELECT
        i.patient_id,
        i.admission_date,
        i.admission_time,
        i.time_period,
        i.season,
        i.injury_cause_category,
        i.injury_location,
        i.longitude,
        i.latitude
    FROM injuryrecord i
    WHERE i.time_period IS NOT NULL
      AND i.injury_cause_category IS NOT NULL
    ORDER BY i.admission_date
""")


def fetch_all_training_data() -> list[dict]:
    """读取全量训练数据，返回 list of dict"""
    with get_engine().connect() as conn:
        result = conn.execute(TRAINING_SQL)
        columns = result.keys()
        return [dict(zip(columns, row)) for row in result]


def count_injury_records() -> int:
    """统计 injuryrecord 表总行数（用于判断是否触发增量更新）"""
    sql = text("SELECT COUNT(*) FROM injuryrecord WHERE time_period IS NOT NULL AND injury_cause_category IS NOT NULL")
    with get_engine().connect() as conn:
        return conn.execute(sql).scalar() or 0
