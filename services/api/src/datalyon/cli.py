"""Run the ML engine straight through on a file (no LLM, automatic approvals).

uv run python -m datalyon.cli ../../data/samples/customer_churn.xlsx "Predict which customers will churn" --budget 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from .ml.ingest import read_file, sniff_file_type
from .ml.pipeline import run_all


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("goal")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--budget", type=int, default=60, help="search time budget in seconds")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    path = Path(a.file)
    sheets = read_file(path, sniff_file_type(path.name, path.read_bytes()[:16]), 500_000, 300)
    sheet = next((s for s in sheets if s.name == a.sheet), max(sheets, key=lambda s: len(s.df)))
    t0 = time.perf_counter()

    def progress(kind: str, p: dict[str, Any]) -> None:
        if kind == "run" and not a.json:
            r = p["run"]
            print(f"  round {r['round']}  {r['candidate']:<28} {r['mean']:.4f} ± {r['std']:.4f}{'  (tuned)' if r['tuned'] else ''}")

    res = run_all(sheet.df, a.goal, budget_s=a.budget, progress=progress)
    out: dict[str, Any] = {
        "sheet": sheet.name,
        "problem_type": res.detection.problem_type,
        "target": res.detection.target,
        "metric": res.strategy.metric_label,
        "algorithm": res.algorithm,
        "metrics": [{k: m[k] for k in ("name", "display", "baseline")} for m in res.metrics],
        "findings": [f"{f.severity}: {f.title}" for f in res.findings],
        "excluded": res.excluded,
        "stopped": res.stopped,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if a.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"\n{out['problem_type']} on `{out['target']}` → {out['algorithm']} ({out['seconds']}s)")
        for m in out["metrics"]:
            print(f"  {m['name']:<24} {m['display']:>10}   baseline {m['baseline']}")
        print("critic:", *out["findings"], sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
