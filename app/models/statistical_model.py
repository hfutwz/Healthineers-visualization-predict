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

# 与前端 predictionOptions 一致，用于接口校验
ALLOWED_DISTRICTS = frozenset({
    '黄浦区', '徐汇区', '长宁区', '静安区', '普陀区', '虹口区', '杨浦区', '浦东新区',
    '闵行区', '宝山区', '嘉定区', '金山区', '松江区', '青浦区', '奉贤区', '崇明区',
})


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

    def district_by_period_cause(self, time_period: int, injury_cause: int) -> dict:
        """某时段 + 某伤因 → 各行政区原始计数（类型 4）"""
        result = defaultdict(int)
        for (d, p, c), v in self.district_period_cause.items():
            if p == time_period and c == injury_cause:
                result[d] += v
        return dict(result)

    def district_by_period_cause_optional(
        self,
        time_period: Optional[int] = None,
        injury_cause: Optional[int] = None,
    ) -> dict:
        """
        类型4 全选版：时段/伤因任一为 None 表示全选，汇总后返回各区计数。
        """
        result = defaultdict(int)
        for (d, p, c), v in self.district_period_cause.items():
            if time_period is not None and p != time_period:
                continue
            if injury_cause is not None and c != injury_cause:
                continue
            result[d] += v
        # 若 district_period_cause 中无数据，回退到 district_cause
        if not result:
            for (d, c), v in self.district_cause.items():
                if injury_cause is not None and c != injury_cause:
                    continue
                result[d] += v
        return dict(result)

    def all_causes_time_distribution(self) -> dict:
        """全部伤因汇总 → 各时段分布（injury_cause 全选时使用）"""
        counts = {p: 0 for p in range(6)}
        for (p, c), v in self.period_cause.items():
            counts[p] = counts.get(p, 0) + v
        return self._normalize_generic(counts)

    def all_causes_season_distribution(self) -> dict:
        """全部伤因汇总 → 各季节分布（injury_cause 全选时使用）"""
        counts = {s: 0 for s in range(4)}
        for (s, c), v in self.season_cause.items():
            counts[s] = counts.get(s, 0) + v
        return self._normalize_generic(counts)

    def district_profile_all(self) -> dict:
        """
        全市汇总的时段/季节/伤因分布（district 全选时使用）。
        """
        n_p = {p: 0 for p in range(6)}
        n_s = {s: 0 for s in range(4)}
        n_c = {c: 0 for c in range(5)}

        for (p, c), v in self.period_cause.items():
            n_p[p] = n_p.get(p, 0) + v
        for (s, c), v in self.season_cause.items():
            n_s[s] = n_s.get(s, 0) + v
        for c, v in self.total_cause.items():
            n_c[c] = v

        period_norm = self._normalize_generic(n_p)
        season_norm = self._normalize_generic(n_s)
        cause_norm = self._normalize_generic(n_c)

        return {
            'district': '全市',
            'period': {TIME_PERIOD_NAMES[p]: round(period_norm.get(p, 0.0), 4) for p in range(6)},
            'season': {SEASON_NAMES[s]: round(season_norm.get(s, 0.0), 4) for s in range(4)},
            'causes': {CAUSE_NAMES[c]: round(cause_norm.get(c, 0.0), 4) for c in range(5)},
        }

    def predict_comprehensive_optional(
        self,
        time_period: Optional[int] = None,
        season: Optional[int] = None,
        district: Optional[str] = None,
    ) -> dict:
        """
        综合伤因概率：时段 / 季节 / 地区任一项可为 None 表示「全部」并在该维上边际化或合理组合。
        """
        d = (district or '').strip() or None
        p_set = time_period is not None
        s_set = season is not None
        d_set = d is not None

        if d_set and p_set and s_set:
            return self.predict_by_district_period_season(d, time_period, season)
        if d_set and p_set:
            return self.predict_by_district_period(d, time_period)
        if d_set and s_set and not p_set:
            return self._merge_proba_int_keys(self.predict_by_district(d), self.predict_by_season(season))
        if d_set and not p_set and not s_set:
            return self.predict_by_district(d)
        if not d_set and p_set and s_set:
            return self.predict_by_period_season(time_period, season)
        if not d_set and p_set:
            return self.predict_by_period(time_period)
        if not d_set and not p_set and s_set:
            return self.predict_by_season(season)
        return self._global()

    def district_profile(self, district: str) -> dict:
        """
        类型 3：某地区 → 时段占比、季节占比、伤因占比（季节在无联合表时用 P(季|时段)×P(时段|地区) 近似）。
        """
        d = district.strip()
        n_dp = {p: 0 for p in range(6)}
        n_c = {c: 0 for c in range(5)}
        for (dist, p, c), v in self.district_period_cause.items():
            if dist != d:
                continue
            n_dp[p] += v
            n_c[c] += v
        tot = sum(n_c.values())
        if tot <= 0:
            return {}

        period_raw = self._normalize_generic({p: n_dp[p] for p in range(6)})
        period = {TIME_PERIOD_NAMES[p]: round(period_raw.get(p, 0.0), 4) for p in range(6)}

        n_ps = defaultdict(int)
        for (p, s, _c), v in self.period_season_cause.items():
            n_ps[(p, s)] += v
        n_p_from_ps = defaultdict(int)
        for (p, s), v in n_ps.items():
            n_p_from_ps[p] += v

        season_acc = defaultdict(float)
        for s in range(4):
            acc = 0.0
            for p in range(6):
                w = n_dp[p] / tot
                ps = n_ps.get((p, s), 0)
                denom = n_p_from_ps.get(p, 0) or 1
                acc += w * (ps / denom)
            season_acc[s] = acc
        season_norm = self._normalize_generic(dict(season_acc))
        season = {SEASON_NAMES[s]: round(season_norm.get(s, 0.0), 4) for s in range(4)}

        cause_norm = self._normalize_generic(n_c)
        causes = {CAUSE_NAMES[c]: round(cause_norm.get(c, 0.0), 4) for c in range(5)}

        return {
            'district': d,
            'period': period,
            'season': season,
            'causes': causes,
        }

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

    def _merge_proba_int_keys(self, a: dict, b: dict) -> dict:
        """地区+季节（无联合表）时，对伤因概率做独立乘积再归一化。"""
        pa = {c: float(a.get(c, 0.0)) for c in range(5)}
        pb = {c: float(b.get(c, 0.0)) for c in range(5)}
        w = {c: max(1e-12, pa[c]) * max(1e-12, pb[c]) for c in range(5)}
        t = sum(w.values())
        if t <= 0:
            return self._global()
        merged = {c: w[c] / t for c in range(5)}
        sa = int(a.get('_sample_n', 0) or 0)
        sb = int(b.get('_sample_n', 0) or 0)
        merged['_sample_n'] = min(sa, sb) if sa and sb else (sa or sb or 1)
        merged['_confidence'] = 'medium'
        merged['_fallback'] = bool(a.get('_fallback')) or bool(b.get('_fallback'))
        return merged

    # ─────────────────────────────────────────────────────────
    # 序列化
    # ─────────────────────────────────────────────────────────

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'TraumaStatisticalModel':
        return joblib.load(path)
