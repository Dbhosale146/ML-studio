from fastapi import APIRouter, Query, Request

from ...db.repositories.activity import list_activity
from ...schemas.api import ActivityOut

router = APIRouter(tags=["activity"])


@router.get("/activity", response_model=list[ActivityOut], summary="Recent real events for the dashboard")
def activity(request: Request, limit: int = Query(30, ge=1, le=200)) -> list[ActivityOut]:
    return [ActivityOut.model_validate(a, from_attributes=True) for a in list_activity(request.app.state.db, limit)]
