"""Agent sessions: one session = one agent run. Events stream over Server-Sent Events with Last-Event-ID replay."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...agent.orchestrator import TERMINAL, AgentService
from ...db.models import AgentSession
from ...schemas.api import ErrorEnvelope, ReplyIn, ReplyOut, SessionCreateIn, SessionOut
from ..deps import get_agent

router = APIRouter(prefix="/sessions", tags=["sessions"], responses={404: {"model": ErrorEnvelope}, 409: {"model": ErrorEnvelope}})


class MessageIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


def _out(row: AgentSession) -> SessionOut:
    return SessionOut(
        id=row.id,
        goal=row.goal,
        status=row.status,
        dataset_id=row.dataset_id,
        model_id=row.model_id,
        stage_key=row.stage_key,
        stage_index=row.stage_index,
        error=row.error,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.post("", response_model=SessionOut, status_code=201, summary="Start an agent run for a goal (and optionally an uploaded dataset)")
def create_session(body: SessionCreateIn, agent: AgentService = Depends(get_agent)) -> SessionOut:
    return _out(agent.create(body.goal, body.dataset_id))


@router.get("", response_model=list[SessionOut], summary="Recent agent runs")
def list_sessions(limit: int = Query(20, ge=1, le=100), agent: AgentService = Depends(get_agent)) -> list[SessionOut]:
    return [_out(r) for r in agent.list(limit)]


@router.get("/{session_id}", response_model=SessionOut)
def get_session(session_id: str, agent: AgentService = Depends(get_agent)) -> SessionOut:
    return _out(agent.get(session_id))


@router.get(
    "/{session_id}/events",
    summary="Server-Sent Events stream of agent events",
    description="Each `data:` line is a JSON envelope `{seq, type, at, payload}` (see docs/agent-events.md). "
    "Send `Last-Event-ID` (or `?after=`) to replay only newer events after a reconnect. An `end` event closes a finished run.",
    response_class=StreamingResponse,
)
async def stream_events(
    session_id: str,
    request: Request,
    after: int | None = Query(None, ge=0),
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    agent: AgentService = Depends(get_agent),
) -> StreamingResponse:
    await run_in_threadpool(agent.get, session_id)
    try:
        start = int(last_event_id) if last_event_id else (after or 0)
    except ValueError:
        start = after or 0

    async def gen() -> AsyncIterator[str]:
        last = start
        last_check = 0.0
        last_ping = time.monotonic()
        yield "retry: 1500\n\n"
        while True:
            if await request.is_disconnected():
                return
            events: list[dict[str, Any]] = await run_in_threadpool(agent.bus.after, session_id, last)
            for ev in events:
                last = ev["seq"]
                yield f"id: {ev['seq']}\nevent: message\ndata: {json.dumps(ev)}\n\n"
            if events:
                continue
            now = time.monotonic()
            if now - last_check > 1.0:
                last_check = now
                status = await run_in_threadpool(agent.status, session_id)
                if status in TERMINAL and session_id not in agent.runs:
                    tail = await run_in_threadpool(agent.bus.after, session_id, last)
                    if not tail:
                        yield f"event: end\ndata: {json.dumps({'status': status})}\n\n"
                        return
            if now - last_ping > 15:
                last_ping = now
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    )


@router.post("/{session_id}/reply", response_model=ReplyOut, summary="Answer the pending question or approval")
def reply(session_id: str, body: ReplyIn, agent: AgentService = Depends(get_agent)) -> ReplyOut:
    agent.reply(session_id, body.ask_id, body.value, body.text)
    return ReplyOut(accepted=True)


@router.post("/{session_id}/message", response_model=ReplyOut, summary="Free text: answers a pending question, or asks the agent something")
def message(session_id: str, body: MessageIn, agent: AgentService = Depends(get_agent)) -> ReplyOut:
    return ReplyOut(accepted=True, message=agent.message(session_id, body.text))


@router.post("/{session_id}/cancel", response_model=SessionOut, summary="Cancel a running agent run")
def cancel(session_id: str, agent: AgentService = Depends(get_agent)) -> SessionOut:
    return _out(agent.cancel(session_id))


@router.post("/{session_id}/retry", response_model=SessionOut, summary="Retry a failed or cancelled run from the stage where it stopped")
def retry(session_id: str, agent: AgentService = Depends(get_agent)) -> SessionOut:
    return _out(agent.retry(session_id))
