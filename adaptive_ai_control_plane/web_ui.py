"""Nexus-Q — interactive web cockpit (FastAPI + WebSocket)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import adaptive_ai_control_plane.settings  # noqa: F401  # load .env before app

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from adaptive_ai_control_plane.main import iter_simulation_events

UI_DIR = Path(__file__).resolve().parent / "ui"

app = FastAPI(title="Nexus-Q Cockpit", version="1.0.0")

_MAX_PROMPT_LEN = 16000
_MAX_LINE_PROMPTS = 200


def _parse_user_prompts_from_message(msg: dict[str, Any]) -> Optional[List[str]]:
    """Build prompt list from WebSocket JSON, or None to use the corpus."""
    mode = str(msg.get("prompt_mode", "single")).lower()
    raw = msg.get("user_prompt")
    if raw is None:
        raw = msg.get("prompt")
    if not isinstance(raw, str):
        raw = str(raw or "")
    text = raw.strip()
    if not text:
        return None
    if mode == "lines":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return [ln[:_MAX_PROMPT_LEN] for ln in lines[:_MAX_LINE_PROMPTS]]
    return [text[:_MAX_PROMPT_LEN]]


if UI_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(UI_DIR / "index.html"))


@app.websocket("/ws/simulate")
async def simulate_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        msg = await websocket.receive_json()
        n_requests = int(msg.get("n_requests", 50))
        daily_budget = float(msg.get("daily_budget", 1.0))
        step_delay_ms = float(msg.get("step_delay_ms", 0))
        n_requests = max(1, min(n_requests, 500))
        daily_budget = max(0.01, min(daily_budget, 1000.0))

        user_prompts = _parse_user_prompts_from_message(msg)

        for event in iter_simulation_events(
            n_requests,
            daily_budget,
            user_prompts=user_prompts,
        ):
            await websocket.send_json(event)
            if step_delay_ms > 0 and event.get("type") == "step":
                await asyncio.sleep(step_delay_ms / 1000.0)
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def main() -> None:
    import uvicorn

    port = int(os.environ.get("NEXUS_Q_WEB_PORT", "8765"))
    host = os.environ.get("NEXUS_Q_WEB_HOST", "127.0.0.1")

    uvicorn.run(
        "adaptive_ai_control_plane.web_ui:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
