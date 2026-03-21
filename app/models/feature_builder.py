"""
特征构建器
★ 新增/修改特征时，只需修改 FEATURE_CONFIG，其他文件无需改动 ★

FEATURE_CONFIG 格式：
    (特征名, 数据库字段名, 转换函数或None, 缺省值)
"""

from app.utils.geo_utils import extract_district


def _parse_month(date_val) -> int:
    try:
        return int(str(date_val)[5:7])
    except Exception:
        return 6


def _parse_year_offset(date_val, base=2024) -> int:
    try:
        return int(str(date_val)[:4]) - base
    except Exception:
        return 0


def _normalize_cause(val) -> int:
    """伤因归类：'其他:...' 系列全部归入 4"""
    if val is None:
        return 4
    s = str(val)
    mapping = {'交通伤': 0, '高坠伤': 1, '机械伤': 2, '跌倒': 3}
    for k, v in mapping.items():
        if s == k:
            return v
    return 4   # 其他


def _resolve_district(val) -> str | None:
    return extract_district(str(val)) if val else None


# ============================================================
# ★ 特征配置表（新增字段：加一行，其余代码无需改动）★
# 格式：(特征名, 数据库字段, 转换函数, 缺省值)
# ============================================================
FEATURE_CONFIG = [
    ('time_period',           'time_period',           None,                 -1),
    ('season',                'season',                None,                 -1),
    ('month',                 'admission_date',        _parse_month,          6),
    ('year_offset',           'admission_date',        _parse_year_offset,    0),
    ('district',              'injury_location',       _resolve_district,  None),
    ('injury_cause_category', 'injury_cause_category', _normalize_cause,      4),   # 目标变量
    # ── 未来可扩展 ──────────────────────────────────────────
    # ('iss_level',  'iss_score',  lambda x: min(int(x)//9, 3) if x else 0, 0),
    # ('age_group',  'age',        lambda x: min(int(x)//20, 4) if x else 2, 2),
]


def build_record(raw_row: dict) -> dict:
    """
    将数据库原始行 → 模型输入字典
    raw_row: 来自 data_service 的 JOIN 查询结果
    """
    result = {}
    for feat_name, source_field, transform, default in FEATURE_CONFIG:
        val = raw_row.get(source_field, None)
        if transform is not None and val is not None:
            try:
                val = transform(val)
            except Exception:
                val = default
        result[feat_name] = val if val is not None else default
    return result


def feature_names() -> list[str]:
    """返回当前所有特征名（不含目标变量）"""
    return [name for name, *_ in FEATURE_CONFIG if name != 'injury_cause_category']
