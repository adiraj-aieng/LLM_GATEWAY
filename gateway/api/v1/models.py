"""``GET /v1/models`` — the logical model catalog.

``/healthz`` lives in ``gateway/api/v1/__init__.py``'s sibling router (mounted at the app
root), because a liveness probe that requires a ``/v1`` prefix is a liveness probe that
breaks the moment you version the API.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from gateway.api import deps

router = APIRouter(tags=["meta"])


@router.get("/models")
async def list_models(
    request: Request,
    registry: Annotated[Any, Depends(deps.get_registry)],
    router_: Annotated[Any, Depends(deps.get_router)],
):
    """Logical model catalog, including how ``auto`` will resolve.

    Clients use this to discover that ``auto`` exists; the ``gateway`` object on each
    entry is the router's public face — it shows the tier and the current router mode.
    """
    models = registry.catalog()
    for entry in models:
        if entry["id"] == "auto":
            entry["gateway"]["router_mode"] = router_.mode
    return JSONResponse({"object": "list", "data": models})
