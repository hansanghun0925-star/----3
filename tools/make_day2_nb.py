"""02_day2_models.ipynb 생성 스크립트. 무거운 계산은 tools/run_day2.py 결과(outputs/day2/*.parquet)를 읽는다."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Day 2 — 글로벌 MLP · Chronos-2 · FVA 표 1차

| 항목 | 설정 |
|---|---|
| 원점 · 예측 기간 · 확정 | 2025-07-01부터 7일 간격 26개 · 7일 · 원점 전날 (Day 1 과 같음) |
| 1순위 지표 | **개소 단위 MASE** — 개소마다 7일 평균 절대 오차 ÷ 그 개소에서 1주 전 같은 요일 값(lag 7)이 2023-01-01~원점 전날 낸 평균 절대 오차, 원점·개소 평균 |
| 인력 | 분위수 모델은 q0.8 ÷ 2,000 (τ* = 60 ÷ (60+15)), 베이스라인은 규칙 그대로 |
| 글로벌 MLP | 입력 28일(log1p, 개소별 표준화) + 열림 표시 + 개소 원-핫 [+ 공변량] → 64 → 64 → 7일 × 분위수 4개(0.1·0.5·0.8·0.9), pinball 손실, Adam 1e-3, 배치 512, 검증 = 원점 직전 56일, 조기 종료 5에폭, 시드 0~4 |
| 학습 구간 | 2023-01-01 ~ 원점 57일 전 (표준화도 같은 구간) |
| Chronos-2 | 원점 전날까지 512일, 7일 한 번에, 분위수 9개. 공변량 = 공휴일·명절 연휴·마지막 수요일·정기 휴관 요일(df·future_df 양쪽) |

계산은 `python tools/run_day2.py` (CPU 8프로세스, 약 10분) 로 `outputs/day2/` 에 저장하고, 이 노트북은 그 결과를 읽어 표를 만듭니다.
""")
code("""
import sys; sys.path.insert(0, 'src')
import pandas as pd, numpy as np
import hv, dl
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', '{:,.3f}'.format)
from pathlib import Path
OUT = Path('outputs/day2')

v, w, c, s = hv.load()
assert hv.check_anchors(v, w, c)['일치'].all()
cal = hv.add_free_rule(hv.make_calendar())
panel = hv.build_panel(v, cal, s)
qden = hv.mase_denominators(panel)
""")

md("## 1. 베이스라인과 서비스 수준 보정 (FVA 1·2행) — MASE 추가")
code("""
m7 = hv.backtest(panel, hv.fc_mean7, name='mean7')
mult = hv.service_level_multiplier(panel, weeks=26, by='group')   # Day 1 검증 구간에서 고른 설정
base = {
    'B1 나이브': hv.backtest(panel, hv.fc_naive, name='B1 나이브'),
    'B2 계절 나이브(7일)': hv.backtest(panel, hv.fc_snaive7, name='B2 계절 나이브(7일)'),
    'F1 직전 7일 평균': m7.assign(model='F1 직전 7일 평균'),
    'F2 + 서비스 수준 보정': hv.apply_multiplier(m7, mult, by='group', name='F2 + 서비스 수준 보정'),
}
hv.summarize(pd.concat(base.values()), qden=qden)
""")

md("""
## 2. 글로벌 MLP — FVA 3행부터 (시드 0~4)

행마다 앞 행의 정보에 하나씩 더합니다. 값은 시드 5개의 평균이고, `범위` 는 시드 5개 값의 최대 − 최소입니다.
""")
code("""
raw = pd.read_parquet(OUT / 'mlp_raw.parquet')
VARIANT_LABEL = {
    'past': 'F3 MLP: 과거 관람객만',
    '+dow': 'F4 + 요일·정기 휴관 요일',
    '+holiday': 'F5 + 공휴일·명절 연휴',
    '+lastwed': 'F6 + 마지막 수요일',
    '+neighbor': 'F7 + 이웃 궁 휴관',
    '+rain_lag': 'F8 + 전날까지 비',
    '[완벽예보]+rain': '[완벽한 예보 가정] + 당일 비',
}
mlp = {}
for vk, lab in VARIANT_LABEL.items():
    r = raw[raw['variant'] == vk]
    mlp[lab] = {sd: hv.attach(panel, g.drop(columns=['variant', 'seed']), lab) for sd, g in r.groupby('seed')}
print('학습 에폭(최적) 분포:', raw.groupby('variant')['epochs'].describe()[['mean', 'min', 'max']].round(1).to_dict('index'))
""")
code("""
def seed_table(runs, by=None):
    per = pd.concat([hv.summarize(bt, by=by, qden=qden).assign(seed=sd) for sd, bt in runs.items()])
    keys = [by] if by else []
    g = per.reset_index().groupby(keys + ['model'])
    mean = g[['MASE', 'WAPE', '포함률(80%)', '적중률', '비용']].mean()
    mean['MASE 범위'] = g['MASE'].agg(lambda x: x.max() - x.min())
    mean['비용 범위'] = g['비용'].agg(lambda x: x.max() - x.min())
    return mean, per

rows, per_seed = [], {}
for lab, runs in mlp.items():
    m, per = seed_table(runs)
    rows.append(m); per_seed[lab] = per.set_index('seed')['비용']
mlp_tab = pd.concat(rows)
mlp_tab
""")

md("## 3. Chronos-2 — 제로샷 / + 공변량")
code("""
chz = pd.read_parquet(OUT / 'chronos_zs.parquet')
chc = pd.read_parquet(OUT / 'chronos_cov.parquet')
ch = pd.concat([chz, chc])
ch_tab = hv.summarize(ch, qden=qden)
# 공개일(2025-10-17) 이전 원점: 채점 7일이 공개일 전에 끝나면 사전학습 자료와 겹쳤을 수 있다
ch['공개일 기준'] = np.where(ch['origin'] + pd.Timedelta(days=6) < dl.CHRONOS_RELEASE, '공개일 전(겹침 가능)', '공개일 후')
display(ch_tab)
hv.summarize(ch, by='공개일 기준', qden=qden)[['n', 'MASE', 'WAPE', '포함률(80%)', '적중률', '비용']]
""")

md("""
## 4. FVA 표 1차 (요구사항 6)

1순위 지표 MASE, 비용(원), 앞 행 대비 비용 변화. MLP 행은 시드 5개 평균, `5시드 모두 앞 행보다 비용 ↓` 는 같은 시드끼리 비교한 결과입니다.
""")
code("""
b = hv.summarize(pd.concat([base['F1 직전 7일 평균'], base['F2 + 서비스 수준 보정']]), qden=qden)
b['MASE 범위'] = 0.0; b['비용 범위'] = 0.0
fva = pd.concat([b.reindex(columns=mlp_tab.columns), mlp_tab, ch_tab.reindex(columns=mlp_tab.columns)])  # 베이스라인은 분위수가 없어 포함률 빈칸
fva['앞 행 대비 비용 변화(원)'] = fva['비용'].diff()
order = list(fva.index)
labs = list(VARIANT_LABEL.values())
same_dir = {labs[i]: bool((per_seed[labs[i]] < per_seed[labs[i - 1]]).all()) for i in range(1, len(labs))}
fva['5시드 모두 앞 행보다 비용 ↓'] = pd.Series(same_dir)
fva.loc[ch_tab.index, '앞 행 대비 비용 변화(원)'] = np.nan
fva['직전 7일 평균 대비 절감(원)'] = fva.loc['F1 직전 7일 평균', '비용'] - fva['비용']
fva_out = fva.copy()
for col in ['비용', '비용 범위', '앞 행 대비 비용 변화(원)', '직전 7일 평균 대비 절감(원)']:
    fva_out[col] = fva_out[col].round(0)
fva_out.to_csv(OUT / 'fva_v1.csv', encoding='utf-8-sig')
fva_out
""")

md("""
## 5. 주 유형별 — 특별일(공휴일·무료 개방일 규칙) 있는 주 / 없는 주

수업 규칙: 평가표를 평균 하나로 정하지 않고 주 유형으로 나눕니다. 26개 원점 가운데 특별일 있는 주의 수도 함께 봅니다.
""")
code("""
def by_week(bt):
    bt = bt.assign(주유형=hv.week_type(bt))
    return hv.summarize(bt, by='주유형', qden=qden)

pick = ['F4 + 요일·정기 휴관 요일', 'F6 + 마지막 수요일', 'F8 + 전날까지 비']
frames = [base['B2 계절 나이브(7일)'], base['F1 직전 7일 평균'], base['F2 + 서비스 수준 보정']] + \\
         [pd.concat([bt.assign(seed=sd) for sd, bt in mlp[p].items()]) for p in pick] + [chz, chc]
wk = []
for f in frames:
    if 'seed' in f:
        t = pd.concat([by_week(g).assign(seed=sd) for sd, g in f.groupby('seed')]).reset_index().groupby(['주유형', 'model']).mean(numeric_only=True).drop(columns='seed')
    else:
        t = by_week(f)
    wk.append(t)
wk = pd.concat(wk)
print('원점 수:', hv.backtest(panel, hv.fc_naive).assign(주유형=lambda d: hv.week_type(d)).groupby('주유형')['origin'].nunique().to_dict())
wk[['MASE', 'WAPE', '포함률(80%)', '적중률', '비용']].unstack('주유형')
""")

md("## 6. 날짜 유형 · 개소 유형별 (주제 1: 무료 개방일)")
code("""
def day_type(d):
    return np.select([d['is_free_rule'] == 1, d['is_holiday'] == 1, d['is_offday'] == 1], ['무료개방(규칙)', '공휴일', '주말'], '평일')

def typed(bt, key):
    bt = bt.assign(날짜유형=day_type(bt))
    return hv.summarize(bt, by=key, qden=qden)[['WAPE', '적중률', '비용']]

best_mlp = pd.concat([bt.assign(seed=sd) for sd, bt in mlp['F6 + 마지막 수요일'].items()])
best_mlp = best_mlp[best_mlp['seed'] == 0]
cmp = pd.concat([base['F1 직전 7일 평균'], base['F2 + 서비스 수준 보정'], best_mlp, chc])
display(typed(cmp, '날짜유형').unstack('model'))
typed(cmp, 'group').unstack('model')
""")

md("### 원점별 비용과 계절 나이브보다 MASE 가 낮은 개소 비율")
code("""
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
d = hv.add_cost(cmp[cmp['status'] == 'open'])
po = d.groupby(['origin', 'model'])['cost'].sum().unstack() / 1e6
ax = po.plot(figsize=(11, 4), marker='o', ms=3)
ax.axvline(dl.CHRONOS_RELEASE, color='gray', ls=':', lw=1); ax.text(dl.CHRONOS_RELEASE, ax.get_ylim()[1] * .95, ' Chronos-2 공개일', fontsize=8)
ax.set_title('원점별 비용 (백만 원, 23개소 × 7일)'); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig(OUT / 'per_origin_cost.png', dpi=150)
po.round(1).to_csv(OUT / 'per_origin_cost.csv', encoding='utf-8-sig')

def site_mase(bt):
    x = bt[bt['status'] == 'open'].assign(ae=lambda d: (d['y'] - d['yhat']).abs())
    return x.groupby(['origin', 'site'])['ae'].mean()
sn = site_mase(base['B2 계절 나이브(7일)'])
beat = pd.DataFrame({m: (site_mase(g) < sn.reindex(site_mase(g).index)).groupby('origin').mean()
                     for m, g in cmp.groupby('model')})
beat.round(2)
""")

md("""
## Day 2 결론 · Day 3 할 일

**FVA (시드 5개 평균, 원점 26개 × 23개소, 기본 가정)**

- F1 직전 7일 평균 8.39억 → F2 서비스 수준 보정 7.68억 → F3 MLP(과거만) 6.97억 → **F5 + 공휴일·명절 5.57억(−1.35억, 5시드 모두 ↓)** → F7 + 이웃 휴관 **5.51억**(직전 7일 평균 대비 **−2.88억**)
- 가장 큰 한 걸음은 **공휴일·명절 연휴** 정보. MASE 0.989 → 0.914 (시드 범위 0.046 보다 큼)
- **요일·정기 휴관**(F4), **마지막 수요일**(F6, 전체 평균), **이웃 휴관**(F7)은 MASE 차이가 시드 범위 안 → 전체 평균으로는 「차이 없음」. 단 주 유형별로 보면 마지막 수요일은 **특별일 있는 주** MASE 1.531(F4) → 1.379(F6), 비용 4.10억 → 2.83억 — 효과가 특별일 주에만 몰려 전체 평균에 묻힘
- **비**: 전날까지 비(F8)는 비용 +0.39억(5시드 모두 ↓ 아님), 당일 비 완벽 예보도 개선 없음 → 날씨는 최종 모형에서 뺀다(격자 강수의 품질 한계와 일치)

**Chronos-2**

- 제로샷 6.27억, + 공변량 5.65억 — 최선 MLP(5.51억)와 비슷한 수준
- 공개일(2025-10-17) 전 원점에서는 + 공변량이 크게 좋고(MASE 0.892), 공개일 후에는 제로샷보다 나쁨(0.995 vs 0.934) → 공개일 전 점수는 사전학습 자료와 겹쳤을 수 있어 그대로 믿지 않는다
- 특별일 없는 주에는 제로샷이 MASE 0.685 로 가장 낮음(수업 결과와 같은 모양)

**적중률·포함률**: MLP 적중률 0.78~0.79(목표 0.80), 80% 포함률 0.76 — 위쪽으로 벗어나는 날이 조금 많음. 무료 개방일 적중률은 F6 0.71, Chronos-2 + 공변량 0.50

**Day 3**

1. 최종 후보: F7(MLP + 공휴일·명절·마지막 수요일·이웃 휴관), Chronos-2 + 공변량, 둘의 앙상블 — 공개일 후 원점·시드 범위로 판정
2. 근무표 확정 시점 2종(원점 전날 vs 1주 전) 비용 비교 — `dl.fit_predict_mlp` 출력 길이 확장 필요
3. 유형별 지표·적중률 확정, 인력이 가장 많이 모자란 10건 → 원인 확인(공지)
4. 마지막 수요일 효과가 왜 전체 평균에서 안 보이는지: 무료 개방일 손실 가중치 / 학습 구간을 2015~2019 궁 자료까지 늘리는 안

""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, ROOT / "02_day2_models.ipynb")
print("written")
