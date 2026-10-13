"""궁·능 관람객 예측 공통 모듈: 데이터 읽기, 검산, 패널, 달력 규칙, 베이스라인, 백테스트, 비용."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1]

# ── 기본 가정 (가이드 요구사항 4) ───────────────────────────────
C_U = 600_000          # 1명 모자랄 때 비용(원/일)
C_O = 150_000          # 1명 남을 때 비용(원/일)
PER_STAFF = 2_000      # 인력 1명이 맡는 하루 관람객
TAU = C_U / (C_U + C_O)  # 임계비율 = 0.8

# ── 백테스트 기본 설정 ─────────────────────────────────────────
ORIGINS = pd.date_range("2025-07-01", periods=26, freq="7D")  # 화요일, 7일 간격
HORIZON = 7


# ════════════════════════════════════════════════════════════
# 1. 읽기와 검산
# ════════════════════════════════════════════════════════════
def load(data_dir: Path = DATA):
    v = pd.read_parquet(data_dir / "heritage_visitors_daily.parquet")
    w = pd.read_parquet(data_dir / "weather_daily.parquet")
    c = pd.read_csv(data_dir / "calendar_kr_2015_2025.csv", parse_dates=["date"])
    s = pd.read_csv(data_dir / "sites.csv", parse_dates=["first_date", "last_date"])
    v["date"] = pd.to_datetime(v["date"])
    w["date"] = pd.to_datetime(w["date"])
    return v, w, c, s


def check_anchors(v, w, c, data_dir: Path = DATA) -> pd.DataFrame:
    a = json.load(open(data_dir / "anchors.json", encoding="utf-8"))
    foreign = ["foreign_en", "foreign_ja", "foreign_zh", "foreign_etc"]
    rows = [
        ("관람객 표 행 수", len(v), a["rows"]),
        ("개소 수", v["site"].nunique(), a["n_sites"]),
        ("total 합", int(v["total"].sum()), a["sums"]["total"]),
        ("외국인 4열 합", int(v[foreign].sum().sum()), a["sums"]["foreign_all"]),
        ("total 이 0 인 행", int((v["total"] == 0).sum()), a["zero_total_site_days"]["all"]),
        ("날씨 표 행 수", len(w), a["weather"]["rows"]),
        ("달력 행 수", len(c), a["calendar"]["rows"]),
    ]
    out = pd.DataFrame(rows, columns=["항목", "읽은 값", "anchors"])
    out["일치"] = out["읽은 값"] == out["anchors"]
    return out


# ════════════════════════════════════════════════════════════
# 2. 달력: 2015~2026, 무료 개방일 규칙
# ════════════════════════════════════════════════════════════
def make_calendar(start="2015-01-01", end="2026-12-31") -> pd.DataFrame:
    """holidays.KR(0.103)로 calendar_kr_2015_2025.csv 와 같은 열을 만든다(2026년 권고표용)."""
    import holidays

    dates = pd.date_range(start, end, freq="D")
    kr = holidays.KR(years=range(dates[0].year - 1, dates[-1].year + 2), language="ko")
    cal = pd.DataFrame({"date": dates})
    cal["weekday"] = cal["date"].dt.dayofweek.map(dict(enumerate("월화수목금토일")))
    cal["holiday_name"] = [kr.get(d.date()) for d in dates]
    cal["is_holiday"] = cal["holiday_name"].notna().astype(int)

    def anchor_days(name):
        return pd.to_datetime(sorted(d for d, n in kr.items() if name in n.split("; ")))

    for col, name in [("seollal_offset", "설날"), ("chuseok_offset", "추석")]:
        days = anchor_days(name)
        idx = np.searchsorted(days.values, dates.values)
        lo = days[np.clip(idx - 1, 0, len(days) - 1)]
        hi = days[np.clip(idx, 0, len(days) - 1)]
        d_lo = (dates - lo).days
        d_hi = (dates - hi).days
        off = np.where(np.abs(d_lo) <= np.abs(d_hi), d_lo, d_hi)
        cal[col] = pd.Series(off, dtype="float").where(np.abs(off) <= 15)

    last_wed = cal[cal["weekday"] == "수"].groupby(cal["date"].dt.to_period("M"))["date"].max()
    cal["is_last_wednesday"] = cal["date"].isin(last_wed).astype(int)
    return cal


def add_free_rule(cal: pd.DataFrame) -> pd.DataFrame:
    """예측에 쓰는 무료 개방일 표시. 달력 규칙만 쓴다(자주 하는 실수 3·5번).

    - 명절 연휴: 설날·추석 당일을 포함해 이어지는 '쉬는 날'(주말 또는 공휴일) 덩어리 전체
    - 문화가 있는 날: 매월 마지막 수요일
    """
    cal = cal.sort_values("date").reset_index(drop=True).copy()
    off = (cal["is_holiday"] == 1) | cal["weekday"].isin(["토", "일"])
    run_id = (off != off.shift()).cumsum()
    anchor = (cal["seollal_offset"] == 0) | (cal["chuseok_offset"] == 0)
    runs_with_anchor = set(run_id[anchor & off])
    cal["is_myeongjeol"] = (off & run_id.isin(runs_with_anchor)).astype(int)
    cal["is_free_rule"] = ((cal["is_myeongjeol"] == 1) | (cal["is_last_wednesday"] == 1)).astype(int)
    cal["is_offday"] = off.astype(int)
    return cal


# ════════════════════════════════════════════════════════════
# 3. 패널: 개소 × 연속 날짜
# ════════════════════════════════════════════════════════════
def build_panel(v, c, s) -> pd.DataFrame:
    """개소마다 자료 기간의 모든 날짜를 깐다.

    status: open(total>0) / closed(total==0, 휴관) / missing(행 없음, 2020-05-30~07-21)
    y     : open 일 때만 관람객 수, 나머지는 NaN → 학습·평가에서 빠진다(실수 1·2번).
    """
    parts = []
    for r in s.itertuples():
        d = pd.DataFrame({"date": pd.date_range(r.first_date, r.last_date, freq="D")})
        d["site"], d["group"], d["closed_weekday"] = r.site, r.group, r.closed_weekday
        parts.append(d)
    p = pd.concat(parts, ignore_index=True)
    p = p.merge(v.drop(columns="group"), on=["date", "site"], how="left")
    p["status"] = np.select(
        [p["total"].isna(), p["total"] == 0], ["missing", "closed"], default="open"
    )
    p["y"] = p["total"].where(p["status"] == "open").astype("float")
    p = p.merge(c, on="date", how="left")
    p["is_closed_weekday"] = (p["weekday"] == p["closed_weekday"]).astype(int)
    return p.sort_values(["site", "date"]).reset_index(drop=True)


# ════════════════════════════════════════════════════════════
# 4. 베이스라인 (원점 전날까지의 기록만 사용)
# ════════════════════════════════════════════════════════════
def _site_series(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """개소별 y 시계열(날짜 인덱스, 연속). 행 순서가 아니라 날짜로 찾기 위함(실수 2번)."""
    return {k: g.set_index("date")["y"] for k, g in panel.groupby("site")}


def fc_naive(y: pd.Series, cutoff, targets):
    hist = y.loc[:cutoff].dropna()
    return pd.Series(hist.iloc[-1] if len(hist) else np.nan, index=targets)


def fc_snaive7(y: pd.Series, cutoff, targets):
    out = {}
    for t in targets:
        cand = t - pd.Timedelta(days=7)
        while cand > cutoff:
            cand -= pd.Timedelta(days=7)
        val = np.nan
        while cand >= y.index[0]:
            val = y.get(cand, np.nan)
            if pd.notna(val):
                break
            cand -= pd.Timedelta(days=7)
        out[t] = val
    return pd.Series(out)


def fc_mean7(y: pd.Series, cutoff, targets, keep_zeros: bool = False, total: pd.Series | None = None):
    """직전 7일 평균. 기본은 문을 연 날만 평균. keep_zeros=True 면 휴관일 0 을 그대로 넣는다(실수 1번 측정용)."""
    win = slice(cutoff - pd.Timedelta(days=6), cutoff)
    if keep_zeros:
        val = total.loc[win].mean()
    else:
        val = y.loc[win].mean()
        if np.isnan(val):  # 7일 내내 휴관이면 직전 개방일 7개 평균으로 대체
            val = y.loc[:cutoff].dropna().iloc[-7:].mean()
    return pd.Series(val, index=targets)


BASELINES = {"naive": fc_naive, "snaive7": fc_snaive7, "mean7": fc_mean7}


def backtest(panel, fc_fn, origins=ORIGINS, horizon=HORIZON, lead=1, name=None, **kw) -> pd.DataFrame:
    """원점마다 cutoff = origin - lead 일까지의 기록으로 origin ~ origin+horizon-1 을 예측.

    lead=1 은 원점 전날 근무표 확정(기본). 결과는 개소 × 날짜 행(휴관·결측 포함, 평가 때 거른다).
    """
    ys = _site_series(panel)
    totals = {k: g.set_index("date")["total"] for k, g in panel.groupby("site")} if kw.get("keep_zeros") else {}
    rows = []
    for o in origins:
        cutoff = o - pd.Timedelta(days=lead)
        targets = pd.date_range(o, periods=horizon, freq="D")
        for site, y in ys.items():
            if cutoff < y.index[0] or targets[-1] > y.index[-1]:
                continue
            extra = {"total": totals[site]} if site in totals else {}
            f = fc_fn(y, cutoff, targets, **kw, **extra)
            rows.append(pd.DataFrame({"origin": o, "site": site, "date": targets, "yhat": f.reindex(targets).values}))
    return attach(panel, pd.concat(rows, ignore_index=True), name or getattr(fc_fn, "__name__", "model"))


def attach(panel, fc: pd.DataFrame, name: str) -> pd.DataFrame:
    """예측 표(origin, site, date, yhat[, q10, q80, q90])에 실제값·날짜 유형을 붙인다.

    yhat  = 점 예측(MASE·WAPE 채점). qstaff = 인력으로 바꿀 값(없으면 yhat 그대로 — 베이스라인 규칙).
    """
    cols = ["group", "y", "total", "status", "is_holiday", "is_free_rule", "is_offday", "weekday"]
    out = fc.join(panel.set_index(["site", "date"])[cols], on=["site", "date"])
    if "qstaff" not in out:
        out["qstaff"] = out["q80"] if "q80" in out else out["yhat"]
    out["model"] = name
    return out


# ════════════════════════════════════════════════════════════
# 5. 서비스 수준 보정 (FVA 2행)
# ════════════════════════════════════════════════════════════
def service_level_multiplier(panel, origins=ORIGINS, weeks=8, tau=TAU, by="site", horizon=HORIZON, lead=1):
    """원점 o 마다, o 이전 `weeks`개 원점의 직전 7일 평균 백테스트에서 '실제 ÷ 예측'의 tau 분위수.

    o 이전 원점의 예측 대상 날짜는 모두 o-1 이하라 미래 정보가 들어가지 않는다.
    by='site' 는 개소별, 'group' 은 개소 유형별 배수.
    """
    first = origins[0] - pd.Timedelta(weeks=weeks)
    past_origins = pd.date_range(first, origins[-1] - pd.Timedelta(days=7), freq="7D")
    hist = backtest(panel, fc_mean7, past_origins, horizon, lead, name="mean7")
    hist = hist[hist["status"] == "open"].assign(r=lambda d: d["y"] / d["yhat"])
    out = []
    for o in origins:
        h = hist[(hist["origin"] < o) & (hist["origin"] >= o - pd.Timedelta(weeks=weeks)) & (hist["date"] < o - pd.Timedelta(days=lead - 1))]
        q = h.groupby(by)["r"].quantile(tau).rename("mult").reset_index()
        q["origin"] = o
        out.append(q)
    return pd.concat(out, ignore_index=True)


def apply_multiplier(bt, mult, by="site", name=None):
    out = bt.merge(mult, on=["origin", by], how="left")
    out["qstaff"] = out["yhat"] * out["mult"].fillna(1.0)   # 점 예측은 그대로, 인력만 보정
    out["model"] = name or bt["model"].iloc[0] + "+SL"
    return out.drop(columns="mult")


# ════════════════════════════════════════════════════════════
# 6. 비용과 지표 (개소 × 날짜 단위, 문을 연 날만)
# ════════════════════════════════════════════════════════════
def add_cost(bt, c_u=C_U, c_o=C_O, per_staff=PER_STAFF):
    bt = bt.copy()
    bt["staff"] = bt["qstaff"] / per_staff
    bt["need"] = bt["y"] / per_staff
    bt["short"] = (bt["need"] - bt["staff"]).clip(lower=0)
    bt["over"] = (bt["staff"] - bt["need"]).clip(lower=0)
    bt["cost"] = c_u * bt["short"] + c_o * bt["over"]
    bt["hit"] = (bt["need"] <= bt["staff"]).astype(int)
    return bt


def mase_denominators(panel, origins=ORIGINS, lead=1, start="2023-01-01") -> pd.DataFrame:
    """개소별 MASE 분모 Q_i: start ~ cutoff 의 문을 연 날에서 1주 전 같은 요일 값(lag 7)의 평균 절대 오차."""
    out = []
    for site, g in panel.groupby("site"):
        y = g.set_index("date")["y"]
        e = (y - y.shift(7, freq="D").reindex(y.index)).abs()
        for o in origins:
            cutoff = o - pd.Timedelta(days=lead)
            out.append((o, site, e.loc[start:cutoff].mean()))
    return pd.DataFrame(out, columns=["origin", "site", "qden"])


def week_type(bt) -> pd.Series:
    """원점 주(7일)에 공휴일·무료 개방일(규칙)이 하나라도 있으면 '특별일 있는 주'."""
    special = (bt["is_holiday"] == 1) | (bt["is_free_rule"] == 1)
    flag = special.groupby(bt["origin"]).transform("max")
    return np.where(flag, "특별일 있는 주", "특별일 없는 주")


def summarize(bt, by=None, open_only=True, qden=None):
    """개소 × 날짜 단위 지표. qden(mase_denominators 결과)을 주면 MASE 도 계산(원점·개소별 MASE 의 평균)."""
    # open_only=False: 휴관일 0 도 실제값으로 넣어 평가(실수 1번 측정용). 결측일은 항상 뺀다.
    d = bt[bt["status"] == "open"] if open_only else bt[bt["status"] != "missing"].assign(y=lambda x: x["total"])
    d = add_cost(d)
    d["ae"] = (d["y"] - d["yhat"]).abs()
    keys = ([by] if isinstance(by, str) else list(by or [])) + ["model"]
    g = d.groupby(keys)
    res = pd.DataFrame({"n": g.size()})
    if qden is not None:
        unit = keys + [k for k in ("origin", "site") if k not in keys]
        m = d.groupby(unit)["ae"].mean().reset_index().merge(qden, on=["origin", "site"])
        res["MASE"] = (m["ae"] / m["qden"]).groupby([m[k] for k in keys]).mean()
    res["WAPE"] = g["ae"].sum() / g["y"].sum()
    if {"q10", "q90"} <= set(d.columns) and d["q10"].notna().any():
        inside = (d["y"] >= d["q10"]) & (d["y"] <= d["q90"])
        res["포함률(80%)"] = inside.groupby([d[k] for k in keys]).mean()
    res["적중률"] = g["hit"].mean()
    res["부족비용"] = g.apply(lambda x: (C_U * x["short"]).sum(), include_groups=False)
    res["잉여비용"] = g.apply(lambda x: (C_O * x["over"]).sum(), include_groups=False)
    res["비용"] = g["cost"].sum()
    return res
