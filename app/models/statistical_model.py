"""
时空因导统计模型
基于历史频率的条件概率查询，支持 6 个预测任务
"""

import datetime
import joblib
from collections import defaultdict
from typing import Optional


CAUSE_NAMES = {0: '交通伤', 1: '高坠伤', 2: '机械伤', 3: '跌倒', 4: '其他'}
TIME_PERIOD_NAMES = {0: '夜间', 1: '早高峰', 2: '午高峰', 3: '下午', 4: '晚高峰', 5: '晚上'}
SEASON_NAMES = {0: '春', 1: '夏', 2: '秋', 3: '冬'}


class TraumaStatisticalModel:

    def __init__(self):
        self.period_cause          = {}   # (period, cause) → count
        self.season_cause          = {}   # (season, cause) → count
        self.district_cause        = {}   # (district, cause) → count
        self.period_season_cause   = {}   # (period, season, cause) → count
        self.district_period_cause = {}   # (district, period, cause) → count
        self.total_cause           = {}   # cause → count  全局基准

        self.meta = {
            'trained_count': 0,
            'district_count': 0,
            'version': None,
            'created_at': None,
        }

    # ─────────────────────────────────────────────────────────
    # 训练
    # ─────────────────────────────────────────────────────────

    def fit(self, records: list[dict]) -> 'TraumaStatisticalModel':
        """
        records: list of dict，由 feature_builder.build_record() 生成
        每条含 time_period / season / district / injury_cause_category
        """
        pc  = defaultdict(int)
        sc  = defaultdict(int)
        dc  = defaultdict(int)
        psc = defaultdict(int)
        dpc = defaultdict(int)
        tc  = defaultdict(int)
        district_count = 0

        for r in records:
            p = r.get('time_period')
            s = r.get('season')
            d = r.get('district')
            c = r.get('injury_cause_category')

            if c is None or c == -1:
                continue

            tc[c] += 1
            if p is not None and p >= 0:
                pc[(p, c)] += 1
            if s is not None and s >= 0:
                sc[(s, c)] += 1
            if d:
                dc[(d, c)] += 1
                district_count += 1
            if p is not None and p >= 0 and s is not None and s >= 0:
                psc[(p, s, c)] += 1
            if d and p is not None and p >= 0:
                dpc[(d, p, c)] += 1

        self.period_cause          = dict(pc)
        self.season_cause          = dict(sc)
        self.district_cause        = dict(dc)
        self.period_season_cause   = dict(psc)
        self.district_period_cause = dict(dpc)
        self.total_cause           = dict(tc)

        self.meta['trained_count'] = len(records)
        self.meta['district_count'] = district_count
        self.meta['created_at'] = datetime.datetime.now().isoformat()

        return self

    # ─────────────────────────────────────────────────────────
    # 6 个预测接口
    # ─────────────────────────────────────────────────────────

    def predict_by_period(self, time_period: int) -> dict:
        """T5: 某时段 → 伤因分布"""
        counts = {c: self.period_cause.get((time_period, c), 0) for c in range(5)}
        return self._to_proba(counts, fallback=self._global())

    def predict_by_season(self, season: int) -> dict:
        """T2: 某季节 → 伤因分布"""
        counts = {c: self.season_cause.get((season, c), 0) for c in range(5)}
        return self._to_proba(counts, fallback=self._global())

    def predict_by_district(self, district: str) -> dict:
        """T4: 某区 → 伤因分布（样本<20条时降级为全局分布）"""
        counts = {c: self.district_cause.get((district, c), 0) for c in range(5)}
        n = sum(counts.values())
        if n < 20:
            # 样本不足，降级到全局分布，并标注 fallback
            result = self._global()
            result['_fallback'] = True
            result['_sample_n'] = n
            result['_confidence'] = 'low'
            return result
        return self._to_proba(counts)

    def predict_by_period_season(self, time_period: int, season: int) -> dict:
        """T1简化: 时段 + 季节 → 伤因分布"""
        counts = {c: self.period_season_cause.get((time_period, season, c), 0) for c in range(5)}
        return self._to_proba(counts, fallback=self.predict_by_period(time_period))

    def predict_by_district_period(self, district: str, time_period: int) -> dict:
        """T1完整: 区域 + 时段 → 伤因分布"""
        counts = {c: self.district_period_cause.get((district, time_period, c), 0) for c in range(5)}
        return self._to_proba(counts, fallback=self.predict_by_district(district))

    def predict_by_district_period_season(self, district: str, time_period: int, season: int) -> dict:
        """T1三维: 区域 + 时段 + 季节 → 伤因分布（逐级降级）"""
        # 三维组合样本通常极少，直接降级到 区域+时段
        return self.predict_by_district_period(district, time_period)

    def cause_time_distribution(self, injury_cause: int) -> dict:
        """T2: 某伤因 → 各时段分布"""
        counts = {p: self.period_cause.get((p, injury_cause), 0) for p in range(6)}
        return self._normalize_generic(counts)

    def cause_season_distribution(self, injury_cause: int) -> dict:
        """T2扩展: 某伤因 → 各季节分布"""
        counts = {s: self.season_cause.get((s, injury_cause), 0) for s in range(4)}
        return self._normalize_generic(counts)

    def district_distribution(self, injury_cause: Optional[int] = None) -> dict:
        """T3/T6: 各区域发生量（可按伤因过滤，供热力图使用）"""
        result = defaultdict(int)
        for (d, c), v in self.district_cause.items():
            if injury_cause is None or c == injury_cause:
                result[d] += v
        return dict(result)

    # ─────────────────────────────────────────────────────────
    # 内部工具
    # ─────────────────────────────────────────────────────────

    def _global(self) -> dict:
        """全局伤因分布（兜底）"""
        return self._to_proba(self.total_cause)

    def _to_proba(self, counts: dict, fallback: Optional[dict] = None) -> dict:
        total = sum(counts.values())
        if total == 0:
            if fallback:
                fb = dict(fallback)
                fb['_fallback'] = True
                return fb
            return {c: 0.2 for c in range(5)}

        proba = {c: v / total for c, v in counts.items()}
        proba['_sample_n'] = total
        proba['_confidence'] = 'high' if total >= 50 else ('medium' if total >= 20 else 'low')
        proba['_fallback'] = False
        return proba

    def _normalize_generic(self, counts: dict) -> dict:
        total = sum(counts.values())
        if total == 0:
            return {k: 1.0 / len(counts) for k in counts}
        return {k: round(v / total, 4) for k, v in counts.items()}

    # ─────────────────────────────────────────────────────────
    # 序列化
    # ─────────────────────────────────────────────────────────

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'TraumaStatisticalModel':
        return joblib.load(path)
