from __future__ import annotations

import asyncio
import json
import logging
import secrets
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status

from mry_alert.config import AppConfig
from mry_alert.models import (
    AlertEvent,
    EventAcknowledgementRequest,
    MatchStatus,
    NotificationRecord,
    PendingDestinationEvent,
    SessionEventRecord,
    SessionHistoryResponse,
)
from mry_alert.notifications.ntfy import NtfyPublisher
from mry_alert.operational_logging import DeliveryLogTracker, log_notification_delivery
from mry_alert.server.audio_ingest import AudioWebSocketService, LiveAudioIngestService
from mry_alert.server.event_store import EventStore, NotificationHistoryStore
from mry_alert.server.runtime_status import RuntimeStatus, utc_now
from mry_alert.server.session_history import (
    SessionEventHistory,
    record_detection_event,
    record_transcript,
)
from mry_alert.server.websocket import WebSocketManager

logger = logging.getLogger(__name__)


def load_or_create_token(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    path.write_text(token + "\n", encoding="utf-8")
    return token


def create_app(
    config: AppConfig | None = None,
    *,
    token: str | None = None,
    event_path: Path | None = None,
    audio_service: AudioWebSocketService | None = None,
    ntfy_publisher: NtfyPublisher | None = None,
) -> FastAPI:
    settings = config or AppConfig()
    pairing_token = token or load_or_create_token(settings.server.pairing_token_file)
    store = EventStore(event_path or Path("data/events.jsonl"))
    notification_store = NotificationHistoryStore(store.path.with_name("notifications.jsonl"))
    sockets = WebSocketManager()
    app = FastAPI(title="MRY Jet Center Alert", version="0.1.0")
    app.state.pairing_token = pairing_token
    app.state.store = store
    app.state.notification_store = notification_store
    app.state.sockets = sockets
    session_id = str(uuid4())
    session_history = SessionEventHistory(settings.event_history.maximum_events)
    app.state.session_id = session_id
    app.state.session_history = session_history
    app.state.input_mode = "idle"
    delivery_log_tracker = DeliveryLogTracker()
    runtime = RuntimeStatus(settings.speech.model, settings.adsb.provider)
    app.state.runtime_status = runtime
    ntfy = ntfy_publisher or NtfyPublisher(settings.ntfy)
    app.state.ntfy_publisher = ntfy

    async def publish_alert(event: AlertEvent | PendingDestinationEvent) -> None:
        session_history.append(record_detection_event(event))
        if isinstance(event, AlertEvent):
            store.append(event)
        result = await sockets.broadcast(event)
        if isinstance(event, AlertEvent):
            chrome_result = (
                "not_connected"
                if result.connected == 0
                else "failed"
                if result.delivered == 0
                else "partial"
                if result.failed
                else "awaiting_extension_ack"
            )
            session_history.update_delivery(event.event_id, chrome=chrome_result)
            notification_store.append(
                NotificationRecord(
                    event_id=event.event_id,
                    timestamp=event.timestamp,
                    sent_at=utc_now(),
                    destination=event.destination,
                    registration=event.registration,
                    spoken_callsign=event.spoken_callsign,
                    aircraft_type=event.aircraft_type,
                    operator_name=event.operator_name,
                    aircraft_type_code=event.aircraft_type_code,
                    aircraft_type_name=event.aircraft_type_name,
                    manufacturer=event.manufacturer,
                    model=event.model,
                    aircraft_category=event.aircraft_category,
                    aircraft_type_source=event.aircraft_type_source,
                    aircraft_type_confidence=event.aircraft_type_confidence,
                    confidence=event.confidence,
                    status=event.status,
                    confirmation_status=event.confirmation_status,
                    transcript_excerpt=event.transcript_excerpt,
                    match_reasons=event.match_reasons,
                    test=event.test,
                    connected_clients=result.connected,
                    delivered_clients=result.delivered,
                    failed_clients=result.failed,
                )
            )
            log_notification_delivery(event, result, settings.logging, delivery_log_tracker)
            runtime.last_notification_at = utc_now()
            runtime.last_notification_success = result.delivered > 0
            runtime.last_notification_delivered = result.delivered
            if settings.ntfy.enabled:
                runtime.ntfy_last_attempt_at = utc_now()
                try:
                    ntfy_result = await ntfy.send(event)
                    runtime.ntfy_last_success = ntfy_result.delivered
                    runtime.ntfy_last_error = ntfy_result.error
                    session_history.update_delivery(
                        event.event_id,
                        ntfy="delivered" if ntfy_result.delivered else "failed",
                    )
                    if ntfy_result.delivered:
                        logger.info("ntfy notification delivered for event %s", event.event_id)
                    else:
                        logger.error(
                            "ntfy notification failed for event %s: %s",
                            event.event_id,
                            ntfy_result.error or "unknown delivery error",
                        )
                except Exception as exc:
                    runtime.ntfy_last_success = False
                    runtime.ntfy_last_error = str(exc)
                    logger.error(
                        "ntfy notification failed for event %s: %s",
                        event.event_id,
                        exc,
                    )
                    session_history.update_delivery(event.event_id, ntfy="failed")
            else:
                session_history.update_delivery(event.event_id, ntfy="disabled")

    async def publish_lifecycle(event: AlertEvent | PendingDestinationEvent) -> None:
        """Record state changes that must not create a desktop/push notification."""
        session_history.append(record_detection_event(event))

    live_audio = audio_service or LiveAudioIngestService(
        settings,
        publish_alert,
        runtime_status=runtime,
        outcome_observer=lambda event: session_history.append(record_transcript(event)),
        lifecycle_publish=publish_lifecycle,
    )
    app.state.live_audio = live_audio

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/status")
    async def api_status() -> dict[str, object]:
        base = {
            "monitoring": app.state.input_mode != "idle",
            "input_mode": app.state.input_mode,
            "speech_model": settings.speech.model,
            "speech_provider": settings.speech.provider,
            "alert_sensitivity": settings.detection.alert_sensitivity,
            "adsb_provider": settings.adsb.provider,
            "connected_extensions": sockets.count,
            "liveatc_enabled": settings.liveatc.enabled,
            "live_audio_status": live_audio.status,
            "ntfy_enabled": settings.ntfy.enabled,
            "ntfy_server_url": settings.ntfy.server_url,
            "ntfy_topic": settings.ntfy.topic,
            "ntfy_authorization_configured": bool(settings.ntfy.authorization),
            "ntfy_subscription_url": ntfy.subscription_url,
            "session_id": session_id,
        }
        base.update(
            runtime.snapshot(
                extension_clients=sockets.count,
                audio_status=live_audio.status,
                input_mode=app.state.input_mode,
            )
        )
        provider = getattr(live_audio, "_provider", None)
        last_success = getattr(provider, "last_success_at", None)
        last_error = getattr(provider, "last_error", None)
        runtime.adsb_error = str(last_error) if last_error else None
        runtime.adsb_ok = last_success is not None and last_error is None
        if last_success is not None:
            runtime.adsb_last_success_at = last_success
        cached = getattr(provider, "_cache", [])
        runtime.adsb_fresh_candidates = len(cached)
        traffic_filter = getattr(provider, "traffic_filter", None)
        tracker = getattr(getattr(live_audio, "_engine", None), "matcher", None)
        correlator = getattr(tracker, "correlator", None)
        contacts = getattr(getattr(correlator, "tracker", None), "contacts", {})
        runtime.adsb_tracked_aircraft = len(contacts)
        base.update(
            runtime.snapshot(
                extension_clients=sockets.count,
                audio_status=live_audio.status,
                input_mode=app.state.input_mode,
            )
        )
        if traffic_filter is not None:
            base.update(traffic_filter.snapshot())
        return base

    @app.get("/api/events", response_model=list[AlertEvent])
    async def events() -> list[AlertEvent]:
        return store.recent()

    @app.get("/api/notifications", response_model=list[NotificationRecord])
    async def notifications() -> list[NotificationRecord]:
        return notification_store.recent()

    @app.get("/api/event-history", response_model=SessionHistoryResponse)
    async def event_history(
        x_pairing_token: str | None = Header(default=None),
    ) -> SessionHistoryResponse:
        if not x_pairing_token or not secrets.compare_digest(x_pairing_token, pairing_token):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid pairing token")
        return SessionHistoryResponse(session_id=session_id, events=session_history.recent())

    @app.post(
        "/api/event-history/{event_id}/acknowledgement",
        response_model=SessionEventRecord,
    )
    async def acknowledge_event(
        event_id: str,
        request: EventAcknowledgementRequest,
        x_pairing_token: str | None = Header(default=None),
    ) -> SessionEventRecord:
        if not x_pairing_token or not secrets.compare_digest(x_pairing_token, pairing_token):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid pairing token")
        record = session_history.acknowledge(
            event_id,
            request.acknowledgement,
            utc_now(),
        )
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session event not found")
        return record

    @app.post("/api/test-alert", response_model=AlertEvent)
    async def test_alert(x_pairing_token: str | None = Header(default=None)) -> AlertEvent:
        if not x_pairing_token or not secrets.compare_digest(x_pairing_token, pairing_token):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid pairing token")
        event = AlertEvent(
            event_id=str(uuid4()),
            timestamp=datetime.now(UTC),
            destination="Monterey Jet Center",
            registration="N123AB",
            spoken_callsign="Test November one two three alpha bravo",
            aircraft_type="Test aircraft",
            aircraft_type_code="TEST",
            aircraft_type_name="Test aircraft type",
            manufacturer="Test manufacturer",
            model="Test model",
            aircraft_category="test",
            aircraft_type_source="test_data",
            aircraft_type_confidence=1.0,
            confidence=1.0,
            status=MatchStatus.CONFIRMED,
            transcript_excerpt="This is a test notification generated locally.",
            match_reasons=["clearly labeled local test alert"],
            test=True,
        )
        await publish_alert(event)
        return event

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket, token: str = "") -> None:
        if not token or not secrets.compare_digest(token, pairing_token):
            await websocket.accept()
            await websocket.close(code=4403, reason="Pairing token rejected")
            return
        await sockets.connect(websocket)
        try:
            while True:
                try:
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=25)
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("type") == "notification_delivery_ack":
                        event_id = payload.get("event_id")
                        delivery = payload.get("status")
                        allowed = {
                            "delivered",
                            "failed",
                            "suppressed",
                            "duplicate_ignored",
                            "pending_not_notified",
                        }
                        if isinstance(event_id, str) and delivery in allowed:
                            session_history.update_delivery(event_id, chrome=delivery)
                except TimeoutError:
                    await websocket.send_json(
                        {"type": "heartbeat", "timestamp": datetime.now(UTC).isoformat()}
                    )
        except WebSocketDisconnect:
            pass
        finally:
            await sockets.disconnect(websocket)

    @app.websocket(settings.liveatc.audio_websocket_path)
    async def audio_websocket_endpoint(websocket: WebSocket, token: str = "") -> None:
        if not token or not secrets.compare_digest(token, pairing_token):
            await websocket.accept()
            await websocket.close(code=4403, reason="Pairing token rejected")
            return
        if not settings.liveatc.enabled:
            await websocket.close(code=1008, reason="LiveATC integration is disabled")
            return
        app.state.input_mode = "liveatc_web_audio"
        try:
            await live_audio.handle(websocket)
        finally:
            app.state.input_mode = "idle"

    return app
