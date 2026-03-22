"""
预测 API 路由
10 个接口：6个预测 + 4个模型管理（含版本控制和增量训练）
"""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from app.models.statistical_model import TraumaStatisticalModel, CAUSE_NAMES
from app.services import training_service
from app.services import base_training_service

router = APIRouter()

# ─── 全局模型实例（启动时加载）──────────────────────────────
_model: Optional[TraumaStatisticalModel] = None


def get_model() -> TraumaStatisticalModel:
    global _model
    if _model is None:
        _model = training_service.load_current_model()
        if _model is None:
            raise HTTPException(503, detail="模型尚未训练，请先调用 /api/model/train")
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
    return _fmt(get_model().predict_by_district(district))


class ComprehensiveQuery(BaseModel):
    time_period: int
    season: Optional[int] = None
    district: Optional[str] = None


@router.post("/predict/comprehensive", summary="T1: 综合预测（区域+时段+季节）")
def comprehensive(body: ComprehensiveQuery):
    m = get_model()
    if body.district and body.season is not None:
        raw = m.predict_by_district_period(body.district, body.time_period)
    elif body.district:
        raw = m.predict_by_district_period(body.district, body.time_period)
    elif body.season is not None:
        raw = m.predict_by_period_season(body.time_period, body.season)
    else:
        raw = m.predict_by_period(body.time_period)
    return _fmt(raw)


@router.get("/predict/time-distribution", summary="T2: 伤因→时段/季节分布")
def time_distribution(
    injury_cause: int = Query(..., ge=0, le=4, description="0交通/1高坠/2机械/3跌倒/4其他"),
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


# ─── 模型管理接口 ────────────────────────────────────────────

@router.get("/api/model/status", summary="查看当前模型状态")
def model_status():
    """获取模型状态（兼容旧版 + 新版版本信息）"""
    status = training_service.get_model_status()
    version_info = base_training_service.get_version_info()
    
    return {
        **status,
        'version_detail': version_info,
        'mode': version_info.get('mode', 'unknown'),
        'base_samples': version_info.get('base_samples', 0),
        'incremental_samples': version_info.get('incremental_samples', 0)
    }


@router.get("/api/model/version", summary="获取版本详细信息")
def model_version():
    """获取模型版本信息（供前端显示）"""
    return base_training_service.get_version_info()


@router.post("/api/model/train", summary="训练模型（支持基础和增量）")
def model_train(
    source: str = Query("both", enum=["excel", "database", "both"], description="训练数据源"),
    force: bool = Query(False, description="强制重新训练")
):
    """
    训练模型
    - source=excel: 仅使用 Excel 数据训练基础模型（会清空已有增量）
    - source=database: 仅使用数据库数据（保留）
    - source=both: 基础模型 + 数据库增量（默认，推荐）
    """
    try:
        if source == "excel":
            # 仅 Excel 基础训练
            result = base_training_service.train_from_excel(force=force)
            if result.get('status') == 'trained':
                reload_model()
            return result
            
        elif source == "both":
            # 基础 + 增量
            # 1. 确保基础模型存在
            if not os.path.exists(base_training_service.BASE_MODEL_FILE) or force:
                base_result = base_training_service.train_from_excel(force=force)
                if base_result.get('status') not in ['trained', 'skipped']:
                    return base_result
            
            # 2. 执行增量训练
            result = base_training_service.incremental_train_from_db(force=force)
            if result.get('status') == 'incremental_trained':
                reload_model()
            return result
            
        else:
            # 仅数据库（原有逻辑）
            result = training_service.train_full()
            if result.get('status') == 'success':
                reload_model()
            return result
            
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@router.post("/api/model/train-base", summary="仅从 Excel 训练基础模型")
def train_base(force: bool = Query(False, description="强制重新训练")):
    """独立接口：仅从 Excel 训练基础模型"""
    try:
        result = base_training_service.train_from_excel(force=force)
        if result.get('status') == 'trained':
            reload_model()
        return result
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@router.post("/api/model/train-incremental", summary="增量训练（基础+数据库新数据）")
def train_incremental(force: bool = Query(False, description="强制训练")):
    """
    增量训练：基础模型 + 数据库新增数据
    供 Java 后端 / 前端 点击按钮调用
    """
    try:
        # 检查基础模型
        if not os.path.exists(base_training_service.BASE_MODEL_FILE):
            return {
                'status': 'error',
                'message': '基础模型不存在，请先执行基础训练或调用 /api/model/train?source=excel'
            }
        
        result = base_training_service.incremental_train_from_db(force=force)
        if result.get('status') == 'incremental_trained':
            reload_model()
        return result
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@router.post("/api/model/trigger-update", summary="增量更新（导入新数据后调用，兼容旧版）")
def trigger_update():
    """兼容旧版的增量更新接口"""
    return train_incremental(force=False)


@router.get("/api/model/history", summary="历史版本列表")
def model_history():
    return training_service.get_model_history()


# ─── 辅助导入 ────────────────────────────────────────────
import os  # 用于文件检查
