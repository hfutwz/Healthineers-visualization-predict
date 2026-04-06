"""
训练服务：模型训练 + 增量更新 + 版本管理
"""

import os
import json
import datetime
import joblib

from app.config import MODEL_DIR, UPDATE_THRESHOLD
from app.models.statistical_model import TraumaStatisticalModel
from app.models.feature_builder import build_record, FEATURE_CONFIG
from app.services.data_service import fetch_all_training_data, count_injury_records


META_FILE = os.path.join(MODEL_DIR, "current.meta.json")
MODEL_FILE = os.path.join(MODEL_DIR, "current.pkl")


def _load_meta() -> dict:
    if os.path.exists(META_FILE):
        with open(META_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {'trained_count': 0, 'version_num': 0, 'version': None}


def _save_meta(meta: dict):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_current_model() -> "TraumaStatisticalModel | None":
    """加载当前模型，不存在则返回 None"""
    if os.path.exists(MODEL_FILE):
        return TraumaStatisticalModel.load(MODEL_FILE)
    return None


def get_model_status() -> dict:
    meta = _load_meta()
    trained = meta.get('trained_count', 0)
    return {
        'version': meta.get('version'),
        'version_num': meta.get('version_num', 0),
        'trained_count': trained,
        'sample_count': trained,          # 前端 modelStatus.sample_count 别名
        'district_count': meta.get('district_count', 0),
        'created_at': meta.get('created_at'),
        'metrics': meta.get('metrics', {}),
        'features': meta.get('features', []),
        'model_ready': os.path.exists(MODEL_FILE),
    }


def get_model_history() -> "list[dict]":
    """读取所有历史版本 meta（目录不存在时返回空列表）"""
    if not os.path.exists(MODEL_DIR):
        return []
    history = []
    for fname in sorted(os.listdir(MODEL_DIR)):
        if fname.endswith('.meta.json') and fname != 'current.meta.json':
            path = os.path.join(MODEL_DIR, fname)
            try:
                with open(path, encoding='utf-8') as f:
                    history.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass  # 损坏的 meta 文件跳过
    return sorted(history, key=lambda x: x.get('version_num', 0), reverse=True)


def train_full() -> dict:
    """全量训练（首次初始化或手动触发）"""
    return _do_train(force=True)


def incremental_update() -> dict:
    """增量更新（旧接口：Python自己连数据库判断）"""
    return _do_train(force=False)


def incremental_push(new_records: list[dict]) -> dict:
    """
    增量推送训练（新接口）：Java 将新增记录直接推送过来，不连数据库。
    策略：加载当前模型的内部计数表 → 合并新记录 → 用全量数据重拟合 → 保存新版本。
    """
    if not new_records:
        return {'status': 'skipped', 'reason': '没有新数据', 'count': 0}

    # 加载现有模型
    current = load_current_model()
    if current is None:
        return {'status': 'error', 'reason': '基础模型不存在，请先执行全量训练'}

    # 从现有模型重建"虚拟记录列表"（从计数表还原）以便合并新数据后重拟合
    existing_records = _expand_model_to_records(current)

    # 合并：已有记录 + 新记录
    all_records = existing_records + new_records

    # 重新拟合
    model = TraumaStatisticalModel()
    model.fit(all_records)

    # 评估（用新记录作为测试集，sample量少时直接记录）
    if len(new_records) >= 10:
        metrics = _evaluate(model, new_records)
    else:
        metrics = {'top1_accuracy': None, 'sample_count': len(new_records), 'note': '新增样本过少，跳过评估'}

    # 版本管理
    meta = _load_meta()
    version_num = meta.get('version_num', 0) + 1
    version = f"v{version_num}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_incr"

    os.makedirs(MODEL_DIR, exist_ok=True)
    versioned_pkl  = os.path.join(MODEL_DIR, f"{version}.pkl")
    versioned_meta = os.path.join(MODEL_DIR, f"{version}.meta.json")
    model.save(versioned_pkl)

    new_meta = {
        'version': version,
        'version_num': version_num,
        'trained_count': len(all_records),
        'sample_count': len(all_records),
        'delta': len(new_records),
        'district_count': model.meta['district_count'],
        'created_at': datetime.datetime.now().isoformat(),
        'metrics': metrics,
        'features': [f[0] for f in FEATURE_CONFIG],
        'incremental': True,
    }
    with open(versioned_meta, 'w', encoding='utf-8') as f:
        json.dump(new_meta, f, ensure_ascii=False, indent=2)

    model.save(MODEL_FILE)
    _save_meta(new_meta)

    return {
        'status': 'success',
        'version': version,
        'delta': len(new_records),
        'trained_count': len(all_records),
        'metrics': metrics,
    }


def _expand_model_to_records(model: "TraumaStatisticalModel") -> list[dict]:
    """
    从模型内部计数表还原虚拟记录列表（用于合并后重拟合）。
    优先用 district_period_cause（最细粒度），补充 period_season_cause。
    """
    records = []

    # 1. 从 district_period_cause 还原（带地区的记录）
    seen_pc = {}  # 记录 (period, cause) 已从这里贡献了多少
    for (d, p, c), count in model.district_period_cause.items():
        for _ in range(count):
            records.append({'time_period': p, 'season': -1, 'district': d, 'injury_cause_category': c})
        seen_pc[(p, c)] = seen_pc.get((p, c), 0) + count

    # 2. 从 period_season_cause 还原缺少地区信息的记录（补差值）
    for (p, s, c), count in model.period_season_cause.items():
        already = seen_pc.get((p, c), 0)
        extra = count - already
        if extra > 0:
            for _ in range(extra):
                records.append({'time_period': p, 'season': s, 'district': None, 'injury_cause_category': c})

    return records


def _do_train(force: bool = False) -> dict:
    meta = _load_meta()
    current_n = count_injury_records()
    last_n = meta.get('trained_count', 0)
    delta = current_n - last_n

    if not force and delta < UPDATE_THRESHOLD:
        return {
            'status': 'skipped',
            'reason': f'新增 {delta} 条，未达阈值 {UPDATE_THRESHOLD}',
            'delta': delta,
            'current_count': current_n,
        }

    # 读取全量数据
    raw_records = fetch_all_training_data()

    # 特征构建
    records = [build_record(r) for r in raw_records]

    # 按时间顺序切分：前80%训练，后20%测试（时间序列切分，避免数据泄露）
    split = max(1, int(len(records) * 0.8))
    train_records = records[:split]
    test_records = records[split:]

    # 用训练集训练统计层
    model = TraumaStatisticalModel()
    model.fit(train_records)

    # 在测试集上评估（out-of-sample）
    metrics = _evaluate(model, test_records)
    metrics['train_count'] = len(train_records)
    metrics['test_count'] = len(test_records)

    # 评估完后用全量数据重训（让模型覆盖最新数据）
    model_full = TraumaStatisticalModel()
    model_full.fit(records)

    # 版本号
    version_num = meta.get('version_num', 0) + 1
    version = f"v{version_num}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"

    # 保存本次版本（带时间戳）——保存全量训练的模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    versioned_pkl  = os.path.join(MODEL_DIR, f"{version}.pkl")
    versioned_meta = os.path.join(MODEL_DIR, f"{version}.meta.json")
    model_full.save(versioned_pkl)

    new_meta = {
        'version': version,
        'version_num': version_num,
        'trained_count': current_n,
        'delta': delta,
        'district_count': model_full.meta['district_count'],
        'created_at': datetime.datetime.now().isoformat(),
        'metrics': metrics,
        'features': [f[0] for f in FEATURE_CONFIG],
    }
    with open(versioned_meta, 'w', encoding='utf-8') as f:
        json.dump(new_meta, f, ensure_ascii=False, indent=2)

    # 覆盖 current（current.pkl / current.meta.json）——用全量模型
    model_full.save(MODEL_FILE)
    _save_meta(new_meta)

    return {
        'status': 'success',
        'version': version,
        'delta': delta,
        'trained_count': current_n,
        'metrics': metrics,
    }


def _evaluate(model: TraumaStatisticalModel, records: list[dict]) -> dict:
    """
    Top-1 命中率：用 predict_by_period_season 预测，
    与实际伤因对比
    """
    hits = 0
    total = 0
    for r in records:
        p = r.get('time_period', -1)
        s = r.get('season', -1)
        c = r.get('injury_cause_category')
        if p < 0 or s < 0 or c is None:
            continue
        pred = model.predict_by_period_season(p, s)
        predicted = max((k for k in range(5)), key=lambda k: pred.get(k, 0))
        if predicted == c:
            hits += 1
        total += 1

    return {
        'top1_accuracy': round(hits / total, 4) if total else 0.0,
        'sample_count': total,
    }
