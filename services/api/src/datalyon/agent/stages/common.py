"""Helpers shared by the stages."""

from __future__ import annotations

import time
from typing import Any

from ...ml import forecasting as fc
from ...ml.metrics import LABELS, LOWER_IS_BETTER, display
from ...ml.pipeline import candidates_for
from ...ml.preprocessing import clean_rows
from ...ml.registry import Tracker
from ...ml.strategy import FREQ_INFO
from ...ml.training import TabularTask, prepare_task
from ...ml.types import Strategy, TrainConfig
from ..context import StageContext

STAGE_PROGRESS = {"data": 3, "detect": 6, "strategy": 10, "prep": 15, "train": 20, "critic": 62, "final": 85, "publish": 95}


class ResumeFromError(Exception):
    """In-memory prerequisites are missing (e.g. after a server restart): restart from an earlier stage."""

    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


def problem_label(plan: dict[str, Any], n_classes: int | None = None, n_series: int | None = None) -> str:
    pt = plan["problem_type"]
    if pt == "binary":
        return "Binary classification"
    if pt == "multiclass":
        return f"Multiclass classification{f' ({n_classes} classes)' if n_classes else ''}"
    if pt == "regression":
        return "Regression"
    return f"Time-series forecasting{f' · {n_series} series' if n_series and n_series > 1 else ''}"


def strategy_of(ctx: StageContext) -> Strategy:
    s = dict(ctx.state["strategy"])
    s["extras"] = [tuple(x) for x in s.get("extras", [])]
    return Strategy(**s)


def tracker(ctx: StageContext) -> Tracker:
    if "tracker" not in ctx.mem:
        ctx.mem["tracker"] = Tracker(
            ctx.settings.mlflow_enabled, ctx.settings.mlruns_uri, f"datalyon-{ctx.state.get('model_id', ctx.session_id)}"
        )
    t: Tracker = ctx.mem["tracker"]
    return t


def ensure_task(ctx: StageContext) -> TabularTask:
    """The prepared tabular task (rebuilt deterministically from the checkpoint when needed)."""
    if "task" not in ctx.mem:
        plan = ctx.state["plan"]
        clean, _ = clean_rows(ctx.df(), plan["target"], dedupe=ctx.state.get("dedupe", True))
        ctx.mem["clean"] = clean
        ctx.mem["task"] = prepare_task(
            clean,
            ctx.profile(),
            plan["problem_type"],
            plan["target"],
            strategy_of(ctx),
            [],
            plan.get("time_column"),
            plan.get("positive_class"),
            ctx.settings.random_state,
        )
    task: TabularTask = ctx.mem["task"]
    return task


def ensure_candidates(ctx: StageContext) -> dict[str, Any]:
    if "cands" not in ctx.mem:
        cands, skipped = candidates_for(ensure_task(ctx), ctx.n_jobs)
        ctx.mem["cands"], ctx.mem["skipped"] = cands, skipped
    c: dict[str, Any] = ctx.mem["cands"]
    return c


def ensure_fc_task(ctx: StageContext) -> fc.ForecastTask:
    if "fc_task" not in ctx.mem:
        plan = ctx.state["plan"]
        ctx.mem["fc_task"] = fc.prepare_forecast(
            ctx.df(),
            plan["target"],
            plan["time_column"],
            plan.get("series_column"),
            plan["freq"],
            plan["horizon"],
            ctx.state["strategy"]["metric"],
            ctx.settings.random_state,
        )
    t: fc.ForecastTask = ctx.mem["fc_task"]
    return t


def train_config(ctx: StageContext, key: str = "cfg") -> TrainConfig:
    return TrainConfig.from_dict(ctx.state.get(key) or {})


def unit_label(plan: dict[str, Any]) -> str:
    return str(FREQ_INFO[plan["freq"]]["unit"]) if plan.get("freq") else "period"


class Leaderboard:
    """Turns engine progress callbacks into live `card.update` and `experiment` events."""

    def __init__(
        self,
        ctx: StageContext,
        round_: int,
        candidates: list[tuple[str, str, str]],
        metric: str,
        prev: dict[str, float] | None = None,
        progress_span: tuple[int, int] = (20, 60),
    ) -> None:
        self.ctx, self.round, self.metric = ctx, round_, metric
        self.rows: dict[str, dict[str, Any]] = {
            key: {
                "key": key,
                "name": name,
                "family": family,
                "status": "queued",
                "progress": 0,
                "score": None,
                "display": None,
                "std": None,
                "tuned": False,
                "prev": (prev or {}).get(name),
                "prev_display": display(metric, (prev or {}).get(name)) if prev and name in prev else None,
                "interim": None,
                "seconds": None,
                "note": None,
            }
            for key, name, family in candidates
        }
        self.trials = 0
        self.tuning: str | None = None
        self.done = False
        self.span = progress_span
        self.best_run: str | None = None
        self._last = 0.0
        self.card_id = ctx.card("leaderboard", self.data())

    def data(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "metric": self.metric,
            "metric_label": LABELS[self.metric],
            "better": "lower" if self.metric in LOWER_IS_BETTER else "higher",
            "rows": list(self.rows.values()),
            "trials": self.trials,
            "tuning": self.tuning,
            "done": self.done,
        }

    def _push(self, force: bool = False) -> None:
        now = time.perf_counter()
        if force or now - self._last > 0.3:
            self._last = now
            self.ctx.card_update(self.card_id, self.data())
            finished = sum(1 for r in self.rows.values() if r["status"] in ("done", "skipped", "failed"))
            lo, hi = self.span
            pct = lo + (hi - lo) * finished / max(1, len(self.rows))
            self.ctx.model_update(progress=int(pct))

    def progress(self, kind: str, p: dict[str, Any]) -> None:
        self.ctx.check_cancel()
        key = p.get("key")
        row = self.rows.get(key) if key else None
        if kind == "start" and row:
            row.update(status="running", progress=2)
        elif kind == "fold" and row:
            row["progress"] = int(100 * p["fold"] / p["folds"]) if row["status"] == "running" else row["progress"]
            row["interim"] = display(self.metric, p.get("partial"))
        elif kind in ("run", "tuned") and row:
            run = p["run"]
            better = row["score"] is None or (run["mean"] < row["score"] if self.metric in LOWER_IS_BETTER else run["mean"] > row["score"])
            if better:
                row.update(
                    score=run["mean"],
                    display=display(self.metric, run["mean"]),
                    std=run["std"],
                    seconds=run["fit_seconds"],
                    tuned=run["tuned"],
                )
            row.update(status="done", progress=100, interim=None)
            if kind == "tuned":
                self.tuning = None
                row["note"] = f"tuned with {p.get('trials', 0)} trials" + ("" if p.get("improved") else ", defaults kept")
            self.ctx.emit(
                "experiment",
                run_id=run["run_id"],
                model=run["candidate"],
                round=self.round,
                score=run["mean"],
                display=display(self.metric, run["mean"]),
                best=False,
                tuned=run["tuned"],
            )
            self._push(force=True)
            return
        elif kind == "tune_start" and row:
            self.tuning = row["name"]
            row["note"] = f"tuning ({p.get('budget_s', 0):.0f}s budget)"
        elif kind == "trial":
            self.trials += 1
        elif kind == "skip" and row:
            row.update(status="skipped", note=p.get("reason"))
        elif kind == "failed" and row:
            row.update(status="failed", note=str(p.get("error", ""))[:120])
        self._push(force=kind in ("start", "skip", "failed", "tune_start"))

    def finish(self) -> dict[str, Any] | None:
        self.done = True
        self.tuning = None
        scored = [r for r in self.rows.values() if r["score"] is not None and r["family"] != "Baseline"]
        best = None
        if scored:
            best = min(scored, key=lambda r: r["score"]) if self.metric in LOWER_IS_BETTER else max(scored, key=lambda r: r["score"])
            best["best"] = True
        self._push(force=True)
        return best
