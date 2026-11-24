"""ORM models. JSON columns hold structured documents (profiles, cards, event payloads)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utcnow


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    file_name: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(16))  # xlsx | xls | csv
    size_bytes: Mapped[int] = mapped_column(Integer)
    file_hash: Mapped[str] = mapped_column(String(64))
    file_path: Mapped[str] = mapped_column(Text)
    sheets: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    sheet: Mapped[str | None] = mapped_column(String(255), nullable=True)
    header_row: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="uploaded")  # uploaded | ready
    rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cols: Mapped[int | None] = mapped_column(Integer, nullable=True)
    data_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    data_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    profile: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    goal: Mapped[str] = mapped_column(Text)
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running | waiting | done | failed | cancelled
    stage_key: Mapped[str | None] = mapped_column(String(32), nullable=True)
    stage_index: Mapped[int] = mapped_column(Integer, default=0)
    state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # checkpoint so a run can resume or retry
    error: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class SessionEvent(Base):
    __tablename__ = "session_events"
    __table_args__ = (UniqueConstraint("session_id", "seq", name="uq_event_seq"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("agent_sessions.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Model(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    slug: Mapped[str] = mapped_column(String(64))
    problem_type: Mapped[str] = mapped_column(String(32))
    problem_label: Mapped[str] = mapped_column(String(64))
    objective: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="draft")  # draft | training | ready | failed
    progress: Mapped[int] = mapped_column(Integer, default=0)
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    target: Mapped[str | None] = mapped_column(String(255), nullable=True)
    algorithm: Mapped[str | None] = mapped_column(String(128), nullable=True)
    latest_version: Mapped[int] = mapped_column(Integer, default=0)
    primary_metric: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    baseline_metric: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    prediction_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("model_id", "version", name="uq_model_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    artifact_path: Mapped[str] = mapped_column(Text)
    card: Mapped[dict[str, Any]] = mapped_column(JSON)
    input_schema: Mapped[dict[str, Any]] = mapped_column(JSON)
    data_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class PredictionJob(Base):
    __tablename__ = "prediction_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued | running | done | failed
    file_name: Mapped[str] = mapped_column(String(255))
    input_path: Mapped[str] = mapped_column(Text)
    output_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Activity(Base):
    __tablename__ = "activity"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String(32))
    text: Mapped[str] = mapped_column(Text)
    model_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dataset_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
