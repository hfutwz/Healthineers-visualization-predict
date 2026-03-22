"""
基础训练服务：从 Excel 文件训练基础模型
支持增量更新（基础模型 + 数据库新增数据）
"""

import os
import json
import pandas as pd
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from app.config import MODEL_DIR, DATA_DIR, DATABASE_URL
from app.models.statistical_model import TraumaStatisticalModel, CAUSE_NAMES
from app.models.feature_builder import build_record, FEATURE_CONFIG
from app.services.data_service import fetch_all_training_data, count_injury_records


BASE_MODEL_FILE = os.path.join(MODEL_DIR, "base_model.pkl")
INCREMENTAL_MODEL_FILE = os.path.join(MODEL_DIR, "incremental_model.pkl")
VERSION_FILE = os.path.join(MODEL_DIR, "version.json")

# Excel 文件路径
DEFAULT_EXCEL_PATH = os.path.join(DATA_DIR, "trauma_patient_data.xlsx")


def _load_version() -> Dict[str, Any]:
    """加载版本信息"""
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'current_version': None,
        'base_version': None,
        'incremental_version': None,
        'base_samples': 0,
        'incremental_samples': 0,
        'total_samples': 0,
        'last_db_count': 0,   # 上次同步时数据库的记录数（用于判断是否有新数据）
        'last_training': None,
        'mode': None,  # 'base' 或 'incremental'
        'created_at': None
    }


def _save_version(info: Dict[str, Any]):
    """保存版本信息"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(VERSION_FILE, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def _parse_excel_data(df: pd.DataFrame) -> List[Dict]:
    """解析 Excel 数据为训练记录格式"""
    records = []
    
    # 列名映射（Excel列名 -> 代码字段名）
    column_mapping = {
        'patient_id': 'patient_id',
        'admission_date': 'admission_date',
        'admission_time': 'admission_time',
        'time_period': 'time_period',
        'season': 'season',
        'injury_cause_category': 'injury_cause_category',
        'injury_location': 'injury_location',
        'longitude': 'longitude',
        'latitude': 'latitude',
        # 尝试小写变体
        'Patient_ID': 'patient_id',
        'Time_Period': 'time_period',
        'Season': 'season',
        'Injury_Cause_Category': 'injury_cause_category',
    }
    
    # 标准化列名
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    for _, row in df.iterrows():
        try:
            record = {
                'patient_id': str(row.get('patient_id', '')),
                'admission_date': row.get('admission_date'),
                'admission_time': row.get('admission_time'),
                'time_period': _parse_int(row.get('time_period')),
                'season': _parse_int(row.get('season')),
                'injury_cause_category': _parse_int(row.get('injury_cause_category')),
                'injury_location': row.get('injury_location', ''),
                'longitude': _parse_float(row.get('longitude')),
                'latitude': _parse_float(row.get('latitude')),
            }
            # 只保留有效记录（有 period 和 cause）
            if record['time_period'] is not None and record['injury_cause_category'] is not None:
                records.append(record)
        except Exception as e:
            # 跳过解析失败的行
            continue
    
    return records


def _parse_int(val):
    """安全解析整数"""
    if pd.isna(val):
        return None
    try:
        return int(float(val))
    except:
        return None


def _parse_float(val):
    """安全解析浮点数"""
    if pd.isna(val):
        return None
    try:
        return float(val)
    except:
        return None


def train_from_excel(excel_path: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """
    从 Excel 文件训练基础模型
    
    Args:
        excel_path: Excel 文件路径，默认使用 data/trauma_patient_data.xlsx
        force: 是否强制重新训练（即使已有基础模型）
    
    Returns:
        训练结果信息
    """
    # 检查是否已有基础模型
    if os.path.exists(BASE_MODEL_FILE) and not force:
        version_info = _load_version()
        return {
            'status': 'skipped',
            'message': '基础模型已存在，使用 force=true 重新训练',
            'version': version_info.get('base_version'),
            'base_samples': version_info.get('base_samples', 0)
        }
    
    # 确定 Excel 路径
    if excel_path is None:
        excel_path = DEFAULT_EXCEL_PATH
    
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel 文件不存在: {excel_path}")
    
    # 读取 Excel
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        raise ValueError(f"Excel 读取失败: {str(e)}")
    
    if len(df) == 0:
        raise ValueError("Excel 文件为空")
    
    # 解析数据
    raw_records = _parse_excel_data(df)
    
    if len(raw_records) == 0:
        raise ValueError("未能从 Excel 解析出有效记录")
    
    # 特征构建
    records = [build_record(r) for r in raw_records]
    records = [r for r in records if r is not None]  # 过滤无效记录
    
    if len(records) == 0:
        raise ValueError("特征构建后无有效记录")
    
    # 训练模型
    model = TraumaStatisticalModel()
    model.fit(records)
    
    # 生成版本号
    base_version = f"BASE_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 保存基础模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(BASE_MODEL_FILE)
    
    # 同时保存为当前模型
    current_model_file = os.path.join(MODEL_DIR, "current.pkl")
    model.save(current_model_file)
    
    # 更新版本信息
    version_info = {
        'current_version': base_version,
        'base_version': base_version,
        'incremental_version': None,
        'base_samples': len(records),
        'incremental_samples': 0,
        'total_samples': len(records),
        'last_training': datetime.datetime.now().isoformat(),
        'mode': 'base',
        'created_at': datetime.datetime.now().isoformat()
    }
    _save_version(version_info)
    
    # 保存 meta 文件（兼容原系统）
    meta_file = os.path.join(MODEL_DIR, "current.meta.json")
    meta = {
        'version': base_version,
        'version_num': 1,
        'trained_count': len(records),
        'delta': len(records),
        'district_count': model.meta.get('district_count', 0),
        'created_at': datetime.datetime.now().isoformat(),
        'metrics': {},
        'features': [f[0] for f in FEATURE_CONFIG],
    }
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    return {
        'status': 'trained',
        'mode': 'base',
        'base_version': base_version,
        'base_samples': len(records),
        'total_samples': len(records),
        'message': f'基础模型训练完成，使用 {len(records)} 条 Excel 数据'
    }


def incremental_train_from_db(force: bool = False) -> Dict[str, Any]:
    """
    增量训练：Excel 基础数据 + 数据库全量数据，合并后重新训练
    
    修复点：
    1. 用 last_db_count 追踪上次同步时的数据库记录数，正确判断是否有新数据
    2. 合并 Excel 基础数据 + 数据库数据一起训练，避免数据库数据少时模型退化
    
    Args:
        force: 是否强制训练（即使无新数据）
    
    Returns:
        训练结果信息
    """
    # 检查基础模型
    if not os.path.exists(BASE_MODEL_FILE):
        raise FileNotFoundError("基础模型不存在，请先执行基础训练（train_from_excel）")
    
    # 加载当前版本信息
    version_info = _load_version()
    
    # 用 last_db_count 判断数据库是否有新数据（修复：不再用 incremental_samples 做判断）
    current_db_count = count_injury_records()
    last_db_count = version_info.get('last_db_count', 0)
    new_db_samples = current_db_count - last_db_count
    
    # 检查是否有新数据
    if new_db_samples <= 0 and not force:
        return {
            'status': 'no_new_data',
            'message': f'数据库无新增数据（上次同步时 {last_db_count} 条，当前仍为 {current_db_count} 条）',
            'current_version': version_info.get('current_version'),
            'total_samples': version_info.get('total_samples', 0),
            'last_db_count': last_db_count,
            'current_db_count': current_db_count
        }
    
    # ── 合并 Excel 基础数据 + 数据库数据（修复：确保基础 2000 条始终参与训练）──
    excel_records: List[Dict] = []
    if os.path.exists(DEFAULT_EXCEL_PATH):
        try:
            df = pd.read_excel(DEFAULT_EXCEL_PATH)
            excel_records = _parse_excel_data(df)
        except Exception as e:
            # Excel 读取失败不阻断增量训练，记录警告
            print(f"[WARNING] 读取 Excel 基础数据失败，本次仅使用数据库数据: {e}")
    
    # 从数据库读取全量数据
    db_raw_records = fetch_all_training_data()
    
    # 合并：Excel 基础数据 + 数据库数据（数据库数据覆盖重复 patient_id）
    # 用 patient_id 去重，数据库数据优先（更新的记录以数据库为准）
    db_patient_ids = {str(r.get('patient_id', '')) for r in db_raw_records}
    excel_only = [r for r in excel_records if str(r.get('patient_id', '')) not in db_patient_ids]
    all_raw = excel_only + db_raw_records
    
    # 特征构建
    records = [build_record(r) for r in all_raw]
    records = [r for r in records if r is not None]
    
    if len(records) == 0:
        raise ValueError("合并后无有效训练记录")
    
    # 训练模型
    model = TraumaStatisticalModel()
    model.fit(records)
    
    # 生成增量版本号
    incremental_version = f"INC_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 保存增量模型
    model.save(INCREMENTAL_MODEL_FILE)
    
    # 更新当前模型
    current_model_file = os.path.join(MODEL_DIR, "current.pkl")
    model.save(current_model_file)
    
    # 统计样本数
    base_samples = version_info.get('base_samples', 0)  # Excel 原始基础数量
    incremental_samples = len(db_raw_records)            # 数据库贡献的样本数
    total_samples = len(records)                          # 实际合并后训练样本数
    
    # 更新版本信息（修复：记录 last_db_count）
    version_info.update({
        'current_version': incremental_version,
        'incremental_version': incremental_version,
        'incremental_samples': incremental_samples,
        'total_samples': total_samples,
        'last_db_count': current_db_count,   # 记录本次同步时数据库记录数
        'last_training': datetime.datetime.now().isoformat(),
        'mode': 'incremental'
    })
    _save_version(version_info)
    
    # 更新 meta 文件
    meta_file = os.path.join(MODEL_DIR, "current.meta.json")
    try:
        version_num = int(version_info.get('base_version', 'BASE_0').split('_')[1]) + 1
    except (IndexError, ValueError):
        version_num = 2
    
    meta = {
        'version': incremental_version,
        'version_num': version_num,
        'trained_count': total_samples,
        'delta': new_db_samples,
        'district_count': model.meta.get('district_count', 0),
        'created_at': datetime.datetime.now().isoformat(),
        'metrics': {},
        'features': [f[0] for f in FEATURE_CONFIG],
    }
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    return {
        'status': 'incremental_trained',
        'mode': 'incremental',
        'base_version': version_info.get('base_version'),
        'incremental_version': incremental_version,
        'base_samples': base_samples,
        'incremental_samples': incremental_samples,
        'total_samples': total_samples,
        'new_db_samples': new_db_samples,
        'message': (
            f'增量训练完成，Excel基础{len(excel_only)}条 + 数据库{incremental_samples}条 = 共{total_samples}条'
        )
    }


def get_version_info() -> Dict[str, Any]:
    """获取当前版本信息"""
    version_info = _load_version()
    
    # 检查文件存在性
    base_exists = os.path.exists(BASE_MODEL_FILE)
    inc_exists = os.path.exists(INCREMENTAL_MODEL_FILE)
    current_exists = os.path.exists(os.path.join(MODEL_DIR, "current.pkl"))
    
    return {
        **version_info,
        'current_version': version_info.get('current_version', '未训练'),
        'mode': version_info.get('mode', 'unknown'),
        'model_files': {
            'base': base_exists,
            'incremental': inc_exists,
            'current': current_exists
        }
    }


def ensure_base_model() -> bool:
    """确保基础模型存在（用于启动检查）"""
    if os.path.exists(BASE_MODEL_FILE):
        return True
    
    # 尝试从 Excel 自动训练
    try:
        if os.path.exists(DEFAULT_EXCEL_PATH):
            train_from_excel()
            return True
    except Exception as e:
        print(f"自动训练基础模型失败: {e}")
    
    return False
