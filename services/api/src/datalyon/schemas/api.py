"""Request/response models for the internal /api/v1 endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody
    request_id: str | None = None


class LlmStatusOut(BaseModel):
    provider: Literal["mock", "gemini"]
    configured_provider: str
    model: str | None = None
    note: str | None = None


class LimitsOut(BaseModel):
    max_upload_mb: int
    max_rows: int
    max_columns: int
    train_time_budget_s: int


class HealthOut(BaseModel):
    status: Literal["ok"]
    version: str
    display_name: str
    llm: LlmStatusOut
    limits: LimitsOut


class Page[T](BaseModel):
    items: list[T]
    total: int
    limit: int
    offset: int


# ---- datasets ---------------------------------------------------------------------------


class SheetInfo(BaseModel):
    name: str
    rows: int
    cols: int
    header_row: int | None = None


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    file_name: str
    file_type: str
    size_bytes: int
    status: str
    sheets: list[SheetInfo]
    sheet: str | None = None
    header_row: int | None = None
    rows: int | None = None
    cols: int | None = None
    quality: int | None = None
    needs_sheet: bool = False
    used_by: list[str] = Field(default_factory=list)
    created_at: datetime


class SheetChoiceIn(BaseModel):
    sheet: str
    header_row: int | None = Field(None, ge=0, description="0-based row index of the header; detected automatically when omitted")


class PreviewOut(BaseModel):
    columns: list[str]
    rows: list[list[str]]
    total_rows: int


# ---- sessions ---------------------------------------------------------------------------


class SessionCreateIn(BaseModel):
    goal: str = Field(..., min_length=3, max_length=2000, examples=["Predict which customers will churn next month"])
    dataset_id: str | None = None


class SessionOut(BaseModel):
    id: str
    goal: str
    status: str
    dataset_id: str | None
    model_id: str | None
    stage_key: str | None
    stage_index: int
    error: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class ReplyIn(BaseModel):
    ask_id: str
    value: Any = None
    text: str | None = Field(None, max_length=2000)


class ReplyOut(BaseModel):
    accepted: bool
    message: str | None = None


# ---- models -----------------------------------------------------------------------------


class MetricOut(BaseModel):
    name: str
    display: str
    value: float | None = None


class ModelSummaryOut(BaseModel):
    id: str
    name: str
    slug: str
    problem_type: str
    problem_label: str
    objective: str
    status: str
    progress: int
    stage: str | None
    target: str | None
    algorithm: str | None
    latest_version: int
    primary_metric: MetricOut | None
    baseline_metric: MetricOut | None
    dataset_id: str | None
    dataset_name: str | None = None
    session_id: str | None
    prediction_count: int
    error: str | None
    created_at: datetime
    updated_at: datetime


class ModelVersionOut(BaseModel):
    version: int
    created_at: datetime
    data_hash: str
    primary_metric: MetricOut | None
    algorithm: str | None
    note: str | None = None


class ModelDetailOut(ModelSummaryOut):
    card: dict[str, Any] | None = None
    versions: list[ModelVersionOut] = Field(default_factory=list)


class InputField(BaseModel):
    name: str
    label: str
    help: str = ""
    kind: Literal["number", "integer", "category", "boolean", "date"]
    required: bool = False
    importance: float = 0.0
    default: Any = None
    min: Any = None
    max: Any = None
    p05: float | None = None
    p95: float | None = None
    median: float | None = None
    step: Any = None
    options: list[dict[str, Any]] | None = None
    allow_other: bool = False


class InputSchemaOut(BaseModel):
    model_id: str
    version: int
    problem_type: str
    target: str | None
    fields: list[InputField]
    examples: list[dict[str, Any]] = Field(default_factory=list)
    forecast: dict[str, Any] | None = None


class PredictIn(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1, max_length=1000)
    series: str | None = Field(None, description="Forecasting only: series to forecast, or null for the total")
    horizon: int | None = Field(None, ge=1, le=1000, description="Forecasting only")
    explain: bool = True


class Driver(BaseModel):
    feature: str
    label: str
    value: Any = None
    effect: float
    direction: Literal["up", "down"]


class PredictionRow(BaseModel):
    prediction: Any
    display: str
    probabilities: dict[str, float] | None = None
    risk_band: str | None = None
    interval: list[float] | None = None
    drivers: list[Driver] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PredictOut(BaseModel):
    model_id: str
    version: int
    problem_type: str
    rows: list[PredictionRow] = Field(default_factory=list)
    forecast: dict[str, Any] | None = None
    explainer: str | None = None


class BatchJobOut(BaseModel):
    id: str
    model_id: str
    status: str
    file_name: str
    rows: int | None
    summary: dict[str, Any] | None
    error: str | None
    created_at: datetime
    download_url: str | None = None


class ActivityOut(BaseModel):
    id: int
    type: str
    text: str
    model_id: str | None
    dataset_id: str | None
    session_id: str | None
    created_at: datetime
