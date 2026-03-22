"""
预测 API 路由

路由结构（两组，共用同一 router，无前缀）：
  /predict/*      — 6 个预测接口（供前端/Java 调用）
  /api/model/*    — 4 个模型管理接口（训练、版本、同步）
"""

import os
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from app.models.statistical_model import TraumaStatisticalModel, CAUSE_NAMES
from app.services import training_service
from app.services import base_training_service

router = APIRouter()

# ─── 全局模型实例（启动时由 lifespan 加载）───────────────────
_model: Optional[TraumaStatisticalModel] = None


def get_model() -> TraumaStatisticalModel:
    global _model
    if _model is None:
        # 懒加载兜底（lifespan 失败时）
        _model = training_service.load_current_model()
        if _model is None:
            raise HTTPException(
                status_code=503,
                detail="模型尚未训练，请先调用 POST /api/model/train?source=excel"
            )
    return _model


def reload_model():
    """训练完成后重新加载模型到内存"""
    global _model
    _model = training_service.load_current_model()


# ─── 格式化工具 ──────────────────────────────────────────────

def _fmt(raw: dict) -> dict:
    """统一格式化伤因概率响应"""
    proba = {CAUSE_NAMES[c]: round(raw.get(c, 0.0), 4) for c in range(5)}
    top = max(proba, key=proba.get)
    meta = training_service.get_model_status()
    return {
        'probabilities': proba,
        'top_cause': top,
        'confidence': raw.get('_confidence', 'medium'),
        'sample_n': raw.get('_sample_n', 0),
        'is_fallback': raw.get('_fallback', False),
        'model_version': meta.get('version', 'unknown'),
    }


# ═══════════════════════════════════════════════════════════════
# 预测接口（/predict/*）
# ═══════════════════════════════════════════════════════════════

@router.get("/predict/cause-by-period", summary="T5: 时段→伤因分布")
def cause_by_period(
    time_period: int = Query(..., ge=0, le=5,
                             description="0夜间/1早高峰/2午高峰/3下午/4晚高峰/5晚上"),
    season: Optional[int] = Query(None, ge=0, le=3,
                                  description="0春/1夏/2秋/3冬，不传则仅按时段"),
):
    m = get_model()
    raw = m.predict_by_period_season(time_period, season) \
        if season is not None else m.predict_by_period(time_period)
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
    return _fmt(get_model().predict_by_district(district))


class ComprehensiveQuery(BaseModel):
    time_period: int
    season: Optional[int] = None
    district: Optional[str] = None


@router.post("/predict/comprehensive", summary="T1: 综合预测（区域+时段+季节）")
def comprehensive(body: ComprehensiveQuery):
    """
    综合预测优先级：
    1. 区域 + 时段（最精确）
    2. 时段 + 季节
    3. 仅时段（兜底）
    """
    m = get_model()
    if body.district:
        # 有区域时用区域+时段（season 暂不参与，模型尚不支持三维联合）
        raw = m.predict_by_district_period(body.district, body.time_period)
    elif body.season is not None:
        raw = m.predict_by_period_season(body.time_period, body.season)
    else:
        raw = m.predict_by_period(body.time_period)
    return _fmt(raw)


@router.get("/predict/time-distribution", summary="T2扩展: 伤因→时段/季节分布")
def time_distribution(
    injury_cause: int = Query(..., ge=0, le=4,
                              description="0交通/1高坠/2机械/3跌倒/4其他"),
):
    m = get_model()
    from app.models.statistical_model import TIME_PERIOD_NAMES, SEASON_NAMES
    period_raw = m.cause_time_distribution(injury_cause)
    season_raw = m.cause_season_distribution(injury_cause)
    return {
        'period': {TIME_PERIOD_NAMES[p]: v for p, v in period_raw.items()},
        'season': {SEASON_NAMES[s]: v for s, v in season_raw.items()},
        'cause': CAUSE_NAMES[injury_cause],
    }


@router.get("/predict/district-distribution", summary="T3/T6: 各区域伤情分布（热力图数据）")
def district_distribution(
    injury_cause: Optional[int] = Query(None, ge=0, le=4),
):
    return get_model().district_distribution(injury_cause)


# ═══════════════════════════════════════════════════════════════
# 模型管理接口（/api/model/*）
# ═══════════════════════════════════════════════════════════════

@router.get("/api/model/status", summary="查看当前模型状态")
def model_status():
    """获取模型状态（含版本、样本量、模式）"""
    status = training_service.get_model_status()
    version_info = base_training_service.get_version_info()
    return {
        **status,
        'version_detail': version_info,
        'mode': version_info.get('mode', 'unknown'),
        'base_samples': version_info.get('base_samples', 0),
        'incremental_samples': version_info.get('incremental_samples', 0),
    }


@router.get("/api/model/version", summary="获取版本详细信息")
def model_version():
    """获取模型版本信息（基础版/增量版/样本数，供前端版本标签显示）"""
    return base_training_service.get_version_info()


@router.post("/api/model/train", summary="训练模型（支持 excel / database / both）")
def model_train(
    source: str = Query(
        "both",
        enum=["excel", "database", "both"],
        description="excel=仅Excel基础训练 | database=仅数据库 | both=Excel+数据库增量（默认）"
    ),
    force: bool = Query(False, description="是否强制重新训练（覆盖已有模型）"),
):
    """
    训练模型
    - source=excel  : 仅用 Excel 数据训练基础模型
    - source=both   : 先确保基础模型存在，再执行增量训练（推荐）
    - source=database: 仅用数据库数据（兼容旧流程）
    """
    try:
        if source == "excel":
            result = base_training_service.train_from_excel(force=force)
            if result.get('status') == 'trained':
                reload_model()
            return result

        elif source == "both":
            # Step1：确保基础模型存在
            if not os.path.exists(base_training_service.BASE_MODEL_FILE) or force:
                base_result = base_training_service.train_from_excel(force=True)
                if base_result.get('status') not in ('trained', 'skipped'):
                    return base_result
                reload_model()  # 基础模型训练后立即加载

            # Step2：增量训练
            result = base_training_service.incremental_train_from_db(force=force)
            # 无论 trained 还是 no_new_data，都 reload（基础模型可能是新的）
            reload_model()
            return result

        else:  # source == "database"
            result = training_service.train_full()
            if result.get('status') == 'success':
                reload_model()
            return result

    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@router.post("/api/model/train-base", summary="仅从 Excel 训练基础模型")
def train_base(force: bool = Query(False, description="是否强制重新训练")):
    """独立接口：仅从 Excel 训练基础模型（不影响增量版本）"""
    try:
        result = base_training_service.train_from_excel(force=force)
        if result.get('status') == 'trained':
            reload_model()
        return result
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@router.post("/api/model/train-incremental", summary="增量训练（Excel基础 + 数据库新数据）")
def train_incremental(force: bool = Query(False, description="是否强制训练（即使无新数据）")):
    """
    增量训练：Excel 2000+ 条基础数据 + 数据库全量数据合并训练。
    供 Java 后端 / 前端「同步预测模型」按钮调用。

    前置条件：基础模型必须存在（先调 /api/model/train-base 或 /api/model/train?source=excel）
    """
    try:
        if not os.path.exists(base_training_service.BASE_MODEL_FILE):
            # 自动初始化基础模型
            base_result = base_training_service.train_from_excel(force=False)
            if base_result.get('status') not in ('trained', 'skipped'):
                return {
                    'status': 'error',
                    'message': f'基础模型不存在且自动训练失败: {base_result.get("message", "")}'
                }
            reload_model()

        result = base_training_service.incremental_train_from_db(force=force)
        reload_model()  # 无论有无新数据都 reload，确保内存模型最新
        return result

    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@router.post("/api/model/trigger-update", summary="增量更新（兼容旧版接口）")
def trigger_update():
    """兼容旧版的增量更新接口，内部调用 train_incremental"""
    return train_incremental(force=False)


@router.get("/api/model/history", summary="历史版本列表")
def model_history():
    return training_service.get_model_history()
