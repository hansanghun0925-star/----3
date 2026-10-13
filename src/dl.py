"""딥러닝 예측: 글로벌 MLP(분위수 손실) · Chronos-2(제로샷 / + 공변량).

수업(딥러닝 시계열) 설정을 따른다.
- 글로벌 MLP: 개소별 표준화, 입력 28일 → 출력 7일, 은닉층 64 × 2(ReLU), 원점 직전 56일 검증 + 조기 종료(5에폭), 시드 0~4
  · 손실만 MAE 대신 분위수(pinball) 손실 τ = 0.1·0.5·0.8·0.9. τ=0.5 는 MAE 와 같은 최적점(중앙값)이고,
    τ=0.8 은 인력 비용(C_u=60만, C_o=15만)을 그대로 최소화하는 분위수(임계비율)다.
  · log1p 공간에서 표준화·학습한다. 분위수는 단조 변환에 대해 보존되므로 expm1 으로 되돌려도 같은 분위수다.
- Chronos-2: 원점 전날까지 512일 입력, 7일을 한 번에 예측. 공변량은 df·future_df 양쪽에 넣은 미래 공변량만.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import hv

QS = [0.1, 0.5, 0.8, 0.9]
L_IN, H = 28, 7
VAL_DAYS = 56
TRAIN_START = pd.Timestamp("2023-01-01")  # 코로나 이후만 학습(2020~2022 수준 변화 회피)

# FVA 행 순서: 앞 행의 정보에 하나씩 더한다
FEATURE_SETS = {
    "past": [],
    "+dow": ["dow", "closed_wd"],
    "+holiday": ["dow", "closed_wd", "holiday"],
    "+lastwed": ["dow", "closed_wd", "holiday", "lastwed"],
    "+neighbor": ["dow", "closed_wd", "holiday", "lastwed", "neighbor"],
    "+rain_lag": ["dow", "closed_wd", "holiday", "lastwed", "neighbor", "rain_past"],
    "[완벽예보]+rain": ["dow", "closed_wd", "holiday", "lastwed", "neighbor", "rain_past", "rain_future"],
}


# ════════════════════════════════════════════════════════════
# 배열 준비: 개소 × 날짜
# ════════════════════════════════════════════════════════════
class Arrays:
    def __init__(self, panel: pd.DataFrame, cal: pd.DataFrame, weather: pd.DataFrame, sites: pd.DataFrame):
        self.sites = list(sites["site"])
        self.groups = dict(zip(sites["site"], sites["group"]))
        self.dates = pd.DatetimeIndex(cal["date"].sort_values())
        S, D = len(self.sites), len(self.dates)
        self.didx = {d: i for i, d in enumerate(self.dates)}

        p = panel.pivot(index="site", columns="date", values="y").reindex(index=self.sites, columns=self.dates)
        self.y = p.to_numpy(dtype="float64")                       # NaN = 휴관·결측·자료 기간 밖
        self.logy = np.log1p(self.y)

        c = cal.set_index("date").reindex(self.dates)
        dow = self.dates.dayofweek.to_numpy()
        self.dow = np.eye(7)[dow]                                   # D × 7
        self.holiday = c["is_holiday"].to_numpy(float)
        self.myeongjeol = c["is_myeongjeol"].to_numpy(float)
        self.lastwed = c["is_last_wednesday"].to_numpy(float)

        wd_name = np.array(list("월화수목금토일"))[dow]
        closed_wd = np.array([[wd_name[j] == cw for j in range(D)] for cw in sites["closed_weekday"]], float)
        self.closed_wd = closed_wd                                  # S × D: 정기 휴관 요일
        # 이웃 휴관: 같은 날 정기 휴관(공휴일 아님)인 다른 궁·종묘 수 / 4  — 달력으로 미리 아는 값
        palace = np.array([self.groups[s] in ("궁", "종묘") for s in self.sites])
        rule_closed = closed_wd * (1 - self.holiday)[None, :]
        tot = rule_closed[palace].sum(0)
        self.neighbor = np.where(palace[:, None], (tot[None, :] - rule_closed) / 4, 0.0)

        wpv = weather.assign(rain=(weather["precip_mm"] >= 5).astype(float)).pivot(
            index="site", columns="date", values="rain").reindex(index=self.sites, columns=self.dates)
        self.rain = wpv.fillna(0).to_numpy()                        # 2026 은 값 없음(0) — 지난 날 실측만 쓸 수 있음
        self.static = np.eye(S)                                     # 개소 원-핫

    # 한 윈도(개소 s, 예측 시작 인덱스 a)의 입력 벡터들을 한꺼번에 만든다
    def features(self, s_idx, a_idx, feats, mu, sd):
        s_idx, a_idx = np.asarray(s_idx), np.asarray(a_idx)
        past = a_idx[:, None] + np.arange(-L_IN, 0)[None, :]        # N × 28
        fut = a_idx[:, None] + np.arange(H)[None, :]                 # N × 7
        z = (self.logy[s_idx[:, None], past] - mu[s_idx, None]) / sd[s_idx, None]
        m = ~np.isnan(z)
        parts = [np.nan_to_num(z), m.astype(float), self.static[s_idx]]
        if "dow" in feats:
            parts.append(self.dow[fut].reshape(len(a_idx), -1))
        if "closed_wd" in feats:
            parts.append(self.closed_wd[s_idx[:, None], fut])
        if "holiday" in feats:
            parts += [self.holiday[fut], self.myeongjeol[fut], self.holiday[past], self.myeongjeol[past]]
        if "lastwed" in feats:
            parts += [self.lastwed[fut], self.lastwed[past]]
        if "neighbor" in feats:
            parts.append(self.neighbor[s_idx[:, None], fut])
        if "rain_past" in feats:
            parts.append(self.rain[s_idx[:, None], past])
        if "rain_future" in feats:
            parts.append(self.rain[s_idx[:, None], fut])
        x = np.concatenate(parts, axis=1).astype("float32")
        yt = (self.logy[s_idx[:, None], fut] - mu[s_idx, None]) / sd[s_idx, None]
        return x, yt.astype("float32")


# ════════════════════════════════════════════════════════════
# 글로벌 MLP
# ════════════════════════════════════════════════════════════
def _pinball(pred, y, qs):
    """pred: N×7×Q, y: N×7 (NaN = 휴관·결측 → 손실에서 뺀다)."""
    import torch
    mask = ~torch.isnan(y)
    yy = torch.nan_to_num(y)[..., None]
    q = torch.tensor(qs, dtype=pred.dtype)
    e = yy - pred
    loss = torch.maximum(q * e, (q - 1) * e) * mask[..., None]
    return loss.sum() / (mask.sum() * len(qs)).clamp(min=1)


def fit_predict_mlp(A: Arrays, origin, feats=(), seed=0, lead=1, hidden=64, lr=1e-3, batch=512,
                    max_epochs=200, patience=5, train_start=TRAIN_START, return_history=False):
    """원점 하나: 학습·검증·예측. cutoff = origin - lead. 반환: 개소 × 7일 예측 표."""
    import torch
    from torch import nn

    torch.manual_seed(seed); np.random.seed(seed)
    o = A.didx[pd.Timestamp(origin)]
    cutoff = o - lead
    val_lo = cutoff - VAL_DAYS + 1                     # 검증 구간: 확정일 직전 56일
    t0 = A.didx[train_start]

    # 개소별 표준화: 검증 구간 전(= 원점 57일 전)까지의 개방일로만
    hist = A.logy[:, t0:val_lo]
    mu = np.nanmean(hist, axis=1); sd = np.nanstd(hist, axis=1)
    sd = np.where(sd > 0, sd, 1.0)

    S = len(A.sites)
    # 학습 윈도: 예측 7일이 학습 구간 안 / 검증 윈도: 예측 7일이 검증 구간 안
    tr_a = np.arange(t0 + L_IN, val_lo - H + 1)
    va_a = np.arange(val_lo, cutoff - H + 2)
    def windows(anchors):
        ss, aa = np.meshgrid(np.arange(S), anchors, indexing="ij")
        ss, aa = ss.ravel(), aa.ravel()
        keep = ~np.all(np.isnan(A.logy[ss[:, None], aa[:, None] + np.arange(H)]), axis=1)
        return ss[keep], aa[keep]
    (trs, tra), (vas, vaa) = windows(tr_a), windows(va_a)
    Xtr, Ytr = map(torch.from_numpy, A.features(trs, tra, feats, mu, sd))
    Xva, Yva = map(torch.from_numpy, A.features(vas, vaa, feats, mu, sd))

    nq = len(QS)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, H * nq))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    fwd = lambda x: torch.sort(net(x).view(-1, H, nq), dim=-1).values   # 분위수 교차 방지

    best, best_state, bad, hist_log = np.inf, None, 0, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad(); loss = _pinball(fwd(Xtr[b]), Ytr[b], QS); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = _pinball(fwd(Xva), Yva, QS).item()
        hist_log.append(vl)
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)

    # 예측: 입력 28일은 cutoff 에서 끝난다. lead>1 이면 cutoff+1 ~ 을 예측해 원점부터 7일을 쓴다
    a_pred = cutoff + 1
    steps = o - a_pred                                   # lead=1 이면 0
    if steps + H > H:
        raise ValueError("lead > 1 은 출력 길이를 늘려야 한다(Day 3 에서 확장)")
    Xp, _ = A.features(np.arange(S), np.full(S, a_pred), feats, mu, sd)
    with torch.no_grad():
        zq = fwd(torch.from_numpy(Xp)).numpy()           # S × 7 × Q
    yq = np.expm1(zq * sd[:, None, None] + mu[:, None, None]).clip(min=0)
    dates = A.dates[o:o + H]
    rows = []
    for si, s in enumerate(A.sites):
        r = pd.DataFrame({"origin": pd.Timestamp(origin), "site": s, "date": dates})
        for qi, q in enumerate(QS):
            r[f"q{int(q*100)}"] = yq[si, :, qi]
        rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    out["yhat"] = out["q50"]
    out["epochs"] = len(hist_log) - bad
    return (out, hist_log) if return_history else out


def run_mlp(A, panel, origins=hv.ORIGINS, feats=(), seeds=(0,), name="MLP", **kw):
    """원점 × 시드 반복. 시드별 예측을 seed 열로 남긴다."""
    out = []
    for sd in seeds:
        for o in origins:
            f = fit_predict_mlp(A, o, feats=feats, seed=sd, **kw)
            f["seed"] = sd
            out.append(f)
    fc = pd.concat(out, ignore_index=True)
    fc = fc[fc.join(panel.set_index(["site", "date"])["status"], on=["site", "date"])["status"].notna()]
    return hv.attach(panel, fc, name)


# ════════════════════════════════════════════════════════════
# Chronos-2
# ════════════════════════════════════════════════════════════
CHRONOS_RELEASE = pd.Timestamp("2025-10-17")
COVS = ["is_holiday", "is_myeongjeol", "is_last_wednesday", "is_closed_weekday"]
_PIPE = None


def chronos_pipe():
    global _PIPE
    if _PIPE is None:
        import torch
        from chronos import BaseChronosPipeline
        _PIPE = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32)
    return _PIPE


def chronos_frames(panel, origin, covariates=False, ctx=512, lead=1):
    o = pd.Timestamp(origin)
    cutoff = o - pd.Timedelta(days=lead)
    past = panel[(panel["date"] > cutoff - pd.Timedelta(days=ctx)) & (panel["date"] <= cutoff)]
    cols = ["site", "date", "y"] + (COVS if covariates else [])
    df = past[cols].rename(columns={"site": "item_id", "date": "timestamp", "y": "target"})
    fut = None
    if covariates:
        days = pd.date_range(cutoff + pd.Timedelta(days=1), o + pd.Timedelta(days=hv.HORIZON - 1))
        fut = panel[panel["date"].isin(days)][["site", "date"] + COVS].rename(
            columns={"site": "item_id", "date": "timestamp"})
        assert "target" not in fut.columns
    return df, fut


def run_chronos(panel, origins=hv.ORIGINS, covariates=False, lead=1, name=None):
    pipe = chronos_pipe()
    qs = [round(0.1 * i, 1) for i in range(1, 10)]
    out = []
    for o in origins:
        df, fut = chronos_frames(panel, o, covariates, lead=lead)
        n_pred = hv.HORIZON + lead - 1
        fc = pipe.predict_df(df, future_df=fut, prediction_length=n_pred, quantile_levels=qs,
                             id_column="item_id", timestamp_column="timestamp", target="target")
        fc = fc.rename(columns={"item_id": "site", "timestamp": "date"})
        fc = fc[fc["date"] >= pd.Timestamp(o)].assign(origin=pd.Timestamp(o))
        f = fc[["origin", "site", "date"]].copy()
        for q in [0.1, 0.5, 0.8, 0.9]:
            f[f"q{int(q*100)}"] = fc[str(q)].clip(lower=0).values
        f["yhat"] = f["q50"]
        out.append(f)
    fc = pd.concat(out, ignore_index=True)
    return hv.attach(panel, fc, name or ("Chronos-2 + 공변량" if covariates else "Chronos-2 제로샷"))
