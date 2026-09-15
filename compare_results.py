"""Print a side-by-side comparison of eval result CSVs.

Usage:
    python compare_results.py 009.csv 010.csv test_6k.csv
Scans results/<model>/<file> for each known model dir and tabulates the
judge aggregates so baseline / V1 / V2 can be compared at a glance.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

MODELS = ["baseline", "noshfinetunedV1", "noshfinetunedV2"]


def agg(path: Path) -> dict:
    df = pd.read_csv(path)
    judged = df[df["overall"].notna()] if "overall" in df.columns else df.iloc[0:0]
    out = {
        "n": len(df),
        "exact": round(df["exact_match"].mean(), 3) if "exact_match" in df else None,
        "syntax": round(df["syntax_valid"].mean(), 3) if "syntax_valid" in df else None,
    }
    if len(judged):
        out.update({
            "overall": round(judged["overall"].mean(), 3),
            "pct>=4": round((judged["overall"] >= 4).mean(), 3),
            "tool": round(judged["tool"].mean(), 3),
            "safety": round(judged["safety"].mean(), 3),
        })
    return out


def main():
    files = sys.argv[1:] or ["009.csv", "010.csv", "test_6k.csv"]
    for f in files:
        print(f"\n=== {f} ===")
        rows = []
        for m in MODELS:
            p = RESULTS / m / f
            if p.exists():
                d = agg(p)
                d["model"] = m
                rows.append(d)
        if not rows:
            print("  (no results)")
            continue
        cols = ["model", "n", "overall", "pct>=4", "exact", "syntax", "tool", "safety"]
        tbl = pd.DataFrame(rows)[ [c for c in cols if c in rows[0] or c == "model"] ]
        print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
