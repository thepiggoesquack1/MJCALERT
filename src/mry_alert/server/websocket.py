from __future__ import annotations

import asyncio

from fastapi import WebSocket

from mry_alert.models import AlertEvent, PendingDestinationEvent
from mry_alert.operational_logging import DeliveryResult


class WebSocketManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    @property
    def count(self) -> int:
        return len(self._connections)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, event: AlertEvent | PendingDestinationEvent) -> DeliveryResult:
        stale: list[WebSocket] = []
        connections = tuple(self._connections)
        delivered = 0
        for websocket in connections:
            try:
                await websocket.send_text(event.model_dump_json())
                delivered += 1
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            await self.disconnect(websocket)
        return DeliveryResult(
            connected=len(connections),
            delivered=delivered,
            failed=len(stale),
        )
