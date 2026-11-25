"""Stage 1: get the file (upload ask), choose the sheet, read and profile it."""

from __future__ import annotations

from typing import Any

from ...core.errors import AppError
from ...ml.metrics import fmt_num
from ..context import StageContext
from ..privacy import safe_summary, summary_text


def run(ctx: StageContext) -> None:
    if not ctx.state.get("greeted"):
        ctx.say(
            f"Hi {ctx.settings.display_name}! I'm your AI data scientist. Your goal: **{ctx.goal}**. "
            "I'll read your data, frame the problem, train and critique candidate models, and explain every decision I make."
        )
        ctx.state["greeted"] = True
    while not ctx.state.get("dataset_id"):
        r = ctx.ask("upload", "Upload the Excel (.xlsx, .xls) or CSV file that holds your data. You can drag it into the chat.")
        try:
            ctx.datasets.get(str(r["value"]))
        except AppError:
            ctx.say("I couldn't find that upload. Please try uploading the file again.")
            continue
        ctx.state["dataset_id"] = str(r["value"])
        ctx.set_status("running", dataset_id=ctx.state["dataset_id"])
    ds = ctx.datasets.get(ctx.state["dataset_id"])
    if ds.status != "ready":
        sheets = sorted(ds.sheets, key=lambda s: -s["rows"])
        r = ctx.ask(
            "sheet",
            f"**{ds.file_name}** has {len(sheets)} sheets. Which one holds the data for this model?",
            options=[{"label": s["name"], "value": s["name"], "hint": f"{s['rows']:,} rows · {s['cols']} columns"} for s in sheets],
            default=sheets[0]["name"],
        )
        try:
            ds = ctx.datasets.select_sheet(ds.id, str(r["value"]))
        except AppError as e:
            ctx.say(f"That didn't work: {e.message}")
            raise
    ctx.checkpoint()
    df = ctx.df()
    profile = ctx.profile()
    info: dict[str, Any] = next((s for s in ds.sheets if s["name"] == ds.sheet), {"notes": []})
    with ctx.tool(f'ingest.read("{ds.file_name}")') as t:
        t.log(f"{ds.file_type.upper()} file, {ds.size_bytes / 1024:,.0f} KB, {len(ds.sheets)} sheet(s)")
        t.log(f"using sheet '{ds.sheet}': {ds.rows:,} rows × {ds.cols} columns")
        for note in info.get("notes", []):
            t.log(note)
    with ctx.tool("data_profiler.run(sheet)") as t:
        kinds: dict[str, int] = {}
        for c in profile["columns"]:
            kinds[c["role"]] = kinds.get(c["role"], 0) + 1
        t.log("column roles: " + ", ".join(f"{v} {k.lower()}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
        t.log(f"{profile['missing_pct']}% of cells empty · {profile['duplicates']:,} duplicate rows")
        for r_ in profile["quality_reasons"]:
            t.log(f"quality −{r_['penalty']}: {r_['detail']}")
        pii = [c["name"] for c in profile["columns"] if c.get("pii")]
        if pii:
            t.log(f"personal data detected and masked for the LLM: {', '.join(pii)}")
    ctx.mem["summary"] = summary_text(safe_summary(df, profile, ctx.settings.llm_sample_rows))
    preview_rows = [[str(v)[:30] if v == v and v is not None else "" for v in row] for row in df.head(6).itertuples(index=False, name=None)]
    ctx.card(
        "profile",
        {
            "file_name": ds.file_name,
            "sheet": ds.sheet,
            "rows": profile["rows"],
            "cols": profile["cols"],
            "quality": profile["quality"],
            "quality_reasons": profile["quality_reasons"],
            "period": profile["period"],
            "duplicates": profile["duplicates"],
            "missing_pct": profile["missing_pct"],
            "columns": [
                {k: c.get(k) for k in ("name", "kind", "role", "missing_pct", "unique", "pii", "top_values", "min", "max", "median")}
                for c in profile["columns"][:40]
            ],
            "issues": profile["issues"][:12],
            "notes": profile["notes"],
            "preview": {"columns": [str(c) for c in df.columns], "rows": preview_rows},
        },
    )
    ctx.blueprint("Data", f"{ds.file_name} · {ds.sheet}")
    ctx.blueprint("Rows", f"{profile['rows']:,} rows × {profile['cols']} columns")
    issues = [i for i in profile["issues"] if i["severity"] in ("high", "medium")]
    ctx.say(
        f"I read **{profile['rows']:,} rows** and {profile['cols']} columns"
        + (f" covering {profile['period']}" if profile["period"] else "")
        + f". Data quality is **{profile['quality']}/100**"
        + (f": the main issues are {', '.join(i['detail'] for i in issues[:3])}." if issues else ", with no major issues.")
        + (f" Only {fmt_num(profile['rows'])} rows is small, so scores will be rough." if profile["rows"] < 300 else "")
    )
