from __future__ import annotations

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from backend.event_bus import BUS, serialize_event
from backend.health import health_snapshot
from database.db_manager import get_all_cameras, get_system_events, init_database

app = FastAPI(title="DriveShieldX Local Backend", version="1.0.0")
init_database()


@app.get("/healthz")
def healthz():
    return health_snapshot()


@app.get("/cameras")
def cameras():
    return get_all_cameras()


@app.get("/events")
def events():
    return get_system_events(100)


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await websocket.accept()
    queue = await BUS.subscribe()
    try:
        for event in BUS.history()[-25:]:
            await websocket.send_text(serialize_event(event))
        while True:
            event = await queue.get()
            await websocket.send_text(serialize_event(event))
    except WebSocketDisconnect:
        BUS.unsubscribe(queue)
    except asyncio.CancelledError:
        BUS.unsubscribe(queue)
        raise
    except Exception:
        BUS.unsubscribe(queue)
        await websocket.close()
