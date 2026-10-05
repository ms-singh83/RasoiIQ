"""Backtest TabPFN vs the same-weekday baseline and write reports/results.{json,md}.

Every number in the README comes from this script. Run:
    .venv/bin/python scripts/evaluate.py              # your orders.csv or the sample
    .venv/bin/python scripts/evaluate.py --seeds 5    # also 5 freshly generated sample datasets
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402,F401  (sets TABPFN_DISABLE_TELEMETRY)
from app.config import default_dataset  # noqa: E402
from app.data import load_orders  # noqa: E402
from app.features import build_features  # noqa: E402
from app.forecast import HOLDOUT_DAYS, REFIT_EVERY, TabPFNForecaster, backtest  # noqa: E402
from scripts.generate_sample import generate  # noqa: E402


def evaluate(raw: bytes | str, label: str) -> dict:
    data = load_orders(raw)
    feats = build_features(data.daily, data.items, data.special_dates)
    t0 = time.perf_counter()
    bt = backtest(feats, TabPFNForecaster())
    bt.pop("rows")
    bt.update(label=label, seconds=round(time.perf_counter() - t0, 1),
              days=(data.end - data.start).days + 1, items=len(data.items))
    return bt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=0, help="extra synthetic datasets to test")
    ap.add_argument("--seed-start", type=int, default=1000)
    a = ap.parse_args()

    path, is_sample = default_dataset()
    runs = [evaluate(path.read_bytes(), f"{path.name}{' (sample)' if is_sample else ''}")]
    for s in range(1, a.seeds + 1):
        seed = a.seed_start + s
        df = generate(date(2026, 10, 3), 90, seed=seed)
        runs.append(evaluate(df.to_csv(index=False), f"sample seed {seed}"))

    out = {"python": platform.python_version(), "platform": platform.platform(),
           "holdout_days": HOLDOUT_DAYS, "refit_every": REFIT_EVERY, "runs": runs}
    import tabpfn, torch  # noqa: E401
    out.update(tabpfn=tabpfn.__version__, torch=torch.__version__)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "results.json").write_text(json.dumps(out, indent=1))

    lines = ["| Dataset | Days x dishes | Baseline MAE | TabPFN MAE | Baseline WAPE | TabPFN WAPE | TabPFN vs baseline |",
             "|---|---|---|---|---|---|---|"]
    for r in runs:
        b, t = r["baseline"], r["tabpfn"]
        gain = (1 - t["mae"] / b["mae"]) * 100
        lines.append(f"| {r['label']} | {r['days']} x {r['items']} | {b['mae']:.2f} | {t['mae']:.2f} | "
                     f"{b['wape']:.1f}% | {t['wape']:.1f}% | {gain:+.1f}% |")
    first = runs[0]
    lines += ["", f"Per dish ({first['label']}), MAE in plates:", "",
              "| Dish | Avg sold/day | Baseline MAE | TabPFN MAE |", "|---|---|---|---|"]
    for k, v in first["per_item"].items():
        lines.append(f"| {k} | {v['actual_avg']:.1f} | {v['baseline_mae']:.2f} | {v['tabpfn_mae']:.2f} |")
    lines += ["", f"Holdout: last {HOLDOUT_DAYS} days, one-day-ahead, TabPFN refit every {REFIT_EVERY} days. "
              f"tabpfn {out['tabpfn']}, torch {out['torch']}, Python {out['python']}, {out['platform']}."]
    md = "\n".join(lines) + "\n"
    (ROOT / "reports" / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
