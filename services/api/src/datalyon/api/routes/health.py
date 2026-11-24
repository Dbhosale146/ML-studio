from fastapi import APIRouter, Depends

from ...core.config import Settings
from ...schemas.api import HealthOut, LimitsOut, LlmStatusOut
from ..deps import get_settings_dep

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut, summary="Service health, LLM mode and limits")
def health(settings: Settings = Depends(get_settings_dep)) -> HealthOut:
    return HealthOut(
        status="ok",
        version=settings.version,
        display_name=settings.display_name,
        llm=LlmStatusOut(
            provider="gemini" if settings.llm_enabled else "mock",
            model=settings.gemini_model if settings.llm_enabled else None,
            configured_provider=settings.llm_provider,
            note=None
            if settings.llm_enabled
            else (
                "GEMINI_API_KEY is empty, so the rule-based MockLLM is used."
                if settings.llm_provider == "gemini"
                else "Rule-based MockLLM (no API key needed)."
            ),
        ),
        limits=LimitsOut(
            max_upload_mb=settings.max_upload_mb,
            max_rows=settings.max_rows,
            max_columns=settings.max_columns,
            train_time_budget_s=settings.train_time_budget_s,
        ),
    )
