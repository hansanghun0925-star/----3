"""Day 2 계산: 글로벌 MLP(FVA 변형 7개 × 시드 0~4 × 원점 26개) + Chronos-2 2종 → outputs/day2/*.parquet

python tools/run_day2.py [--workers 8]
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs" / "day2"

_A = None


def _init():
    global _A
    import torch
    torch.set_num_threads(2)
    import hv, dl
    v, w, c, s = hv.load()
    cal = hv.add_free_rule(hv.make_calendar())
    p = hv.build_panel(v, cal, s)
    _A = dl.Arrays(p, cal, w, s)


def _job(args):
    import dl
    variant, seed, origin = args
    f = dl.fit_predict_mlp(_A, origin, feats=dl.FEATURE_SETS[variant], seed=seed)
    f["seed"], f["variant"] = seed, variant
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    import hv, dl

    jobs = [(vname, sd, o) for vname in dl.FEATURE_SETS for sd in range(5) for o in hv.ORIGINS]
    t = time.time()
    res = []
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        for i, r in enumerate(ex.map(_job, jobs, chunksize=4)):
            res.append(r)
            if (i + 1) % 52 == 0:
                print(f"MLP {i + 1}/{len(jobs)}  {time.time() - t:.0f}s", flush=True)
    pd.concat(res, ignore_index=True).to_parquet(OUT / "mlp_raw.parquet")
    print("MLP done", round(time.time() - t), "s", flush=True)

    v, w, c, s = hv.load()
    cal = hv.add_free_rule(hv.make_calendar())
    p = hv.build_panel(v, cal, s)
    for cov in (False, True):
        t = time.time()
        bt = dl.run_chronos(p, covariates=cov)
        bt.to_parquet(OUT / f"chronos_{'cov' if cov else 'zs'}.parquet")
        print("chronos", cov, round(time.time() - t), "s", flush=True)


if __name__ == "__main__":
    main()
