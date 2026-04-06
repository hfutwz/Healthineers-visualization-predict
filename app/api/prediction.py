"""
预测 API 路由
8 个接口：6个预测 + 2个模型管理
"""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.models.statistical_model import TraumaStatisticalModel, CAUSE_NAMES, ALLOWED_DISTRICTS
from app.services import training_service

router = APIRouter()

# ─── 全局模型实例 + 版本缓存（启动时加载）───────────────────
_model: Optional[TraumaStatisticalModel] = None
_model_version: str = "unknown"   # 缓存版本号，避免每次预测都读磁盘


def get_model() -> TraumaStatisticalModel:
    global _model
    if _model is None:
        _model = training_service.load_current_model()
        if _model is None:
            raise HTTPException(503, detail="模型尚未训练，请先调用 /api/model/train")
    return _model


def reload_model():
    """训练完成后重新加载模型到内存，同步更新版本缓存"""
    global _model, _model_version
    _model = training_service.load_current_model()
    status = training_service.get_model_status()
    _model_version = status.get('version', 'unknown')


# ─── 格式化工具 ──────────────────────────────────────────────

def _normalize_district_param(district: str) -> str:
    d = (district or '').strip()
    if d not in ALLOWED_DISTRICTS:
        raise HTTPException(status_code=422, detail=f"地区不在允许列表内: {district!r}")
    return d


def _fmt(raw: dict) -> dict:
    """统一格式化伤因概率响应（使用内存缓存的版本号，避免磁盘 IO）"""
    proba = {CAUSE_NAMES[c]: round(raw.get(c, 0.0), 4) for c in range(5)}
    top = max(proba, key=proba.get)
    return {
        'probabilities': proba,
        'top_cause': top,
        'confidence': raw.get('_confidence', 'medium'),
        'sample_n': raw.get('_sample_n', 0),
        'is_fallback': raw.get('_fallback', False),
        'model_version': _model_version,
    }


# ─── 预测接口 ────────────────────────────────────────────────

@router.get("/predict/cause-by-period", summary="T5: 时段→伤因分布")
def cause_by_period(
    time_period: int = Query(..., ge=0, le=5, description="0夜间/1早高峰/2午高峰/3下午/4晚高峰/5晚上"),
    season: Optional[int] = Query(None, ge=0, le=3, description="0春/1夏/2秋/3冬，不传则仅按时段"),
):
    m = get_model()
    if season is not None:
        raw = m.predict_by_period_season(time_period, season)
    else:
        raw = m.predict_by_period(time_period)
    return _fmt(raw)


@router.get("/predict/cause-by-season", summary="T2: 季节→伤因分布")
def cause_by_season(
    season: int = Query(..., ge=0, le=3),
):
    return _fmt(get_model().predict_by_season(season))


@router.get("/predict/cause-by-district", summary="T4: 地区→伤因分布")
def cause_by_district(
    district: str = Query(..., description="上海行政区，如'宝山区'"),
):
    d = _normalize_district_param(district)
    return _fmt(get_model().predict_by_district(d))


class ComprehensiveQuery(BaseModel):
    """time_period / season / district 均可为 null 表示「全部」，三维同时为空则返回全局伤因分布。"""

    model_config = ConfigDict(extra='ignore')

    time_period: Optional[int] = Field(default=None, ge=0, le=5)
    season: Optional[int] = Field(default=None, ge=0, le=3)
    district: Optional[str] = None


@router.post("/predict/comprehensive", summary="T1: 综合预测（区域+时段+季节，支持「全部」）")
def comprehensive(body: ComprehensiveQuery):
    m = get_model()
    district = body.district
    if district is not None and str(district).strip():
        district = _normalize_district_param(str(district))
    else:
        district = None
    raw = m.predict_comprehensive_optional(
        time_period=body.time_period,
        season=body.season,
        district=district,
    )
    return _fmt(raw)


@router.get("/predict/time-distribution", summary="T2: 伤因→时段/季节分布（injury_cause=null 表示全部）")
def time_distribution(
    injury_cause: Optional[int] = Query(None, ge=0, le=4, description="0交通/1高坠/2机械/3跌倒/4其他，不传表示全部伤因"),
):
    m = get_model()
    from app.models.statistical_model import TIME_PERIOD_NAMES, SEASON_NAMES
    if injury_cause is None:
        # 全选：汇总所有伤因的时段/季节分布
        period_raw = m.all_causes_time_distribution()
        season_raw = m.all_causes_season_distribution()
        cause_label = '全部'
    else:
        period_raw = m.cause_time_distribution(injury_cause)
        season_raw = m.cause_season_distribution(injury_cause)
        cause_label = CAUSE_NAMES[injury_cause]
    return {
        'period': {TIME_PERIOD_NAMES[p]: v for p, v in period_raw.items()},
        'season': {SEASON_NAMES[s]: v for s, v in season_raw.items()},
        'cause': cause_label,
    }


@router.get("/predict/district-distribution", summary="T3/T6: 各区域伤情分布（热力图数据）")
def district_distribution(
    injury_cause: Optional[int] = Query(None, ge=0, le=4),
):
    return get_model().district_distribution(injury_cause)


@router.get("/predict/district-profile", summary="类型3: 地区→时段/季节/伤因分布（district=null 表示全市）")
def district_profile(
    district: Optional[str] = Query(None, description="上海行政区，不传表示全市汇总"),
):
    m = get_model()
    if district is None or not district.strip():
        # 全选：全市汇总
        out = m.district_profile_all()
    else:
        d = _normalize_district_param(district)
        out = m.district_profile(d)
        if not out:
            raise HTTPException(status_code=404, detail=f"地区 {d} 无统计数据")
    return out


@router.get("/predict/district-by-period-cause", summary="类型4: 时段+伤因→地区分布（支持全选）")
def district_by_period_cause(
    time_period: Optional[int] = Query(None, ge=0, le=5, description="不传表示全部时段"),
    injury_cause: Optional[int] = Query(None, ge=0, le=4, description="不传表示全部伤因"),
):
    return get_model().district_by_period_cause_optional(time_period, injury_cause)


# ─── 模型管理接口 ────────────────────────────────────────────

@router.get("/api/model/status", summary="查看当前模型状态")
def model_status():
    return training_service.get_model_status()


@router.post("/api/model/train", summary="全量训练（首次初始化）")
def model_train():
    result = training_service.train_full()
    if result.get('status') == 'success':
        reload_model()
    return result


@router.post("/api/model/trigger-update", summary="增量更新（导入新数据后调用）")
def trigger_update():
    result = training_service.incremental_update()
    if result.get('status') == 'success':
        reload_model()
    return result


@router.get("/api/model/history", summary="历史版本列表")
def model_history():
    return training_service.get_model_history()
