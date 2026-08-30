import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from mry_alert.config import AppConfig, LiveAtcConfig
from mry_alert.server.app import create_app


class FakeAudioService:
    status = "idle"

    async def handle(self, websocket: Any) -> None:
        await websocket.accept()
        self.status = "monitoring"
        await websocket.send_json({"type": "status", "status": self.status})
        data = await websocket.receive_bytes()
        await websocket.send_json({"type": "received", "bytes": len(data)})
        self.status = "idle"
        await websocket.close()


class FakeAdsbProvider:
    def __init__(self) -> None:
        self.last_success_at = datetime.now(UTC)
        self.last_error: str | None = "temporary upstream failure"
        self._cache: list[object] = []


class FakeAudioServiceWithProvider(FakeAudioService):
    def __init__(self) -> None:
        self._provider = FakeAdsbProvider()


def test_health_status_and_authentication(tmp_path: Path) -> None:
    client = TestClient(create_app(token="correct", event_path=tmp_path / "events.jsonl"))
    assert client.get("/health").json() == {"status": "ok"}
    api_status = client.get("/api/status").json()
    assert api_status["connected_extensions"] == 0
    assert api_status["liveatc_enabled"] is True
    assert api_status["live_audio_status"] == "idle"
    assert api_status["server_running"] is True
    assert api_status["extension_clients"] == 0
    assert api_status["audio_recently_received"] is False
    assert api_status["speech_model"] == "small.en"
    assert api_status["traffic_filter_enabled"] is True
    assert api_status["filtered_airline_count"] == 0
    assert api_status["allowed_override_count"] == 0
    assert api_status["unknown_operator_count"] == 0
    assert "pairing_token" not in api_status
    assert client.post("/api/test-alert", headers={"X-Pairing-Token": "wrong"}).status_code == 401
    assert client.get("/api/event-history").status_code == 401
    assert api_status["session_id"]


def test_valid_websocket_receives_test_event(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="mry_alert.operations")
    client = TestClient(create_app(token="correct", event_path=tmp_path / "events.jsonl"))
    with client.websocket_connect("/ws?token=correct") as websocket:
        response = client.post("/api/test-alert", headers={"X-Pairing-Token": "correct"})
        assert response.status_code == 200
        received = websocket.receive_json()
        assert received["test"] is True
        assert received["registration"] == "N123AB"
        assert received["aircraft_type_name"] == "Test aircraft type"
        websocket.send_json(
            {
                "type": "notification_delivery_ack",
                "event_id": received["event_id"],
                "status": "delivered",
            }
        )
    assert "NOTIFICATION SENT" in caplog.text
    assert "Extension:    Sent to 1 connected client" in caplog.text
    assert len(client.get("/api/events").json()) == 1
    notifications = client.get("/api/notifications").json()
    assert len(notifications) == 1
    assert notifications[0]["registration"] == "N123AB"
    assert notifications[0]["delivered_clients"] == 1
    assert notifications[0]["connected_clients"] == 1
    assert notifications[0]["test"] is True
    session = client.get(
        "/api/event-history", headers={"X-Pairing-Token": "correct"}
    ).json()
    assert session["events"][0]["transition_type"] == "confirmed"
    assert session["events"][0]["chrome_delivery_result"] == "delivered"
    assert session["events"][0]["test"] is True


def test_zero_client_delivery_log_never_exposes_pairing_token(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="mry_alert.operations")
    secret = "pairing-secret-must-not-be-logged"
    client = TestClient(create_app(token=secret, event_path=tmp_path / "events.jsonl"))
    response = client.post("/api/test-alert", headers={"X-Pairing-Token": secret})
    assert response.status_code == 200
    assert "NOTIFICATION NOT DELIVERED" in caplog.text
    assert secret not in caplog.text
    assert "\\x00" not in caplog.text


def test_notification_history_persists_across_backend_restarts(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    client = TestClient(create_app(token="correct", event_path=event_path))
    response = client.post("/api/test-alert", headers={"X-Pairing-Token": "correct"})
    assert response.status_code == 200
    history = client.get("/api/notifications").json()
    assert history[0]["delivered_clients"] == 0
    assert history[0]["connected_clients"] == 0

    restarted = TestClient(create_app(token="correct", event_path=event_path))
    reloaded = restarted.get("/api/notifications").json()
    assert len(reloaded) == 1
    assert reloaded[0]["event_id"] == response.json()["event_id"]


def test_session_history_survives_reconnect_but_clears_on_restart(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    app = create_app(token="correct", event_path=path)
    client = TestClient(app)
    client.post("/api/test-alert", headers={"X-Pairing-Token": "correct"})
    first = client.get(
        "/api/event-history", headers={"X-Pairing-Token": "correct"}
    ).json()
    reconnect = TestClient(app).get(
        "/api/event-history", headers={"X-Pairing-Token": "correct"}
    ).json()
    restarted = TestClient(create_app(token="correct", event_path=path)).get(
        "/api/event-history", headers={"X-Pairing-Token": "correct"}
    ).json()

    assert reconnect == first
    assert restarted["session_id"] != first["session_id"]
    assert restarted["events"] == []


def test_session_event_acknowledgement_is_authenticated_and_nonpersistent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    client = TestClient(create_app(token="correct", event_path=path))
    event_id = client.post(
        "/api/test-alert", headers={"X-Pairing-Token": "correct"}
    ).json()["event_id"]
    endpoint = f"/api/event-history/{event_id}/acknowledgement"

    assert client.post(endpoint, json={"acknowledgement": "seen"}).status_code == 401
    response = client.post(
        endpoint,
        headers={"X-Pairing-Token": "correct"},
        json={"acknowledgement": "aircraft_arrived"},
    )
    assert response.status_code == 200
    assert response.json()["operator_acknowledgement"] == "aircraft_arrived"
    assert response.json()["acknowledged_at"] is not None

    restarted = TestClient(create_app(token="correct", event_path=path))
    history = restarted.get(
        "/api/event-history", headers={"X-Pairing-Token": "correct"}
    ).json()
    assert history["events"] == []


def test_invalid_websocket_is_rejected(tmp_path: Path) -> None:
    client = TestClient(create_app(token="correct", event_path=tmp_path / "events.jsonl"))
    with (
        client.websocket_connect("/ws?token=wrong") as websocket,
        pytest.raises(WebSocketDisconnect) as rejected,
    ):
        websocket.receive_json()
    assert rejected.value.code == 4403


def test_adsb_runtime_error_clears_after_provider_recovers(tmp_path: Path) -> None:
    service = FakeAudioServiceWithProvider()
    client = TestClient(
        create_app(
            token="correct",
            event_path=tmp_path / "events.jsonl",
            audio_service=service,
        )
    )
    failed = client.get("/api/status").json()
    assert failed["adsb_ok"] is False
    assert failed["adsb_error"] == "temporary upstream failure"

    service._provider.last_error = None
    recovered = client.get("/api/status").json()
    assert recovered["adsb_ok"] is True
    assert recovered["adsb_error"] is None


def test_authorized_audio_websocket_accepts_binary_pcm(tmp_path: Path) -> None:
    service = FakeAudioService()
    client = TestClient(
        create_app(
            token="correct",
            event_path=tmp_path / "events.jsonl",
            audio_service=service,
        )
    )
    with client.websocket_connect("/ws/audio?token=correct") as websocket:
        assert websocket.receive_json() == {"type": "status", "status": "monitoring"}
        websocket.send_bytes(b"\0" * 960)
        assert websocket.receive_json() == {"type": "received", "bytes": 960}


def test_audio_websocket_rejects_invalid_token_and_disabled_config(tmp_path: Path) -> None:
    client = TestClient(create_app(token="correct", event_path=tmp_path / "events.jsonl"))
    with (
        client.websocket_connect("/ws/audio?token=wrong") as websocket,
        pytest.raises(WebSocketDisconnect) as rejected,
    ):
        websocket.receive_json()
    assert rejected.value.code == 4403

    disabled = AppConfig(liveatc=LiveAtcConfig(enabled=False))
    disabled_client = TestClient(
        create_app(disabled, token="correct", event_path=tmp_path / "disabled.jsonl")
    )
    with (
        pytest.raises(WebSocketDisconnect),
        disabled_client.websocket_connect("/ws/audio?token=correct"),
    ):
        pass
