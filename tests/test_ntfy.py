from __future__ import annotations

from datetime import UTC, datetime
from urllib.request import Request

from fastapi.testclient import TestClient

from mry_alert.config import AppConfig, NtfyConfig
from mry_alert.models import (
    AlertEvent,
    IdentificationSource,
    MatchStatus,
)
from mry_alert.notifications.ntfy import NtfyPublisher
from mry_alert.server.app import create_app


def alert() -> AlertEvent:
    return AlertEvent(
        event_id="ntfy-test",
        timestamp=datetime(2026, 7, 30, 19, 30, tzinfo=UTC),
        destination="Monterey Jet Center",
        registration="N123AB",
        spoken_callsign="November one two three alpha bravo",
        aircraft_type="Cessna Citation CJ3",
        aircraft_type_name="Cessna Citation CJ3",
        confidence=0.91,
        status=MatchStatus.CONFIRMED,
        transcript_excerpt="taxi to Monterey Jet Center",
        identification_source=IdentificationSource.ADSB_CORRELATION,
    )


async def test_ntfy_sends_complete_alert_with_optional_authorization() -> None:
    captured: list[tuple[Request, float]] = []

    def transport(request: Request, timeout: float) -> None:
        captured.append((request, timeout))

    publisher = NtfyPublisher(
        NtfyConfig(
            enabled=True,
            server_url="https://ntfy.sh/",
            topic="mry_private_123",
            authorization="Bearer secret",
        ),
        transport=transport,
    )

    result = await publisher.send(alert())

    assert result.delivered is True
    request, timeout = captured[0]
    assert request.full_url == "https://ntfy.sh/mry_private_123"
    assert request.get_header("Authorization") == "Bearer secret"
    body = (request.data or b"").decode()
    assert request.get_header("Title") == "N123AB - Cessna Citation CJ3"
    assert "Going to Monterey Jet Center" in body
    assert "Confidence: 91%" in body
    assert "ADS-B source: adsb_correlation" in body
    assert "2026-07-30T19:30:00+00:00" in body
    assert timeout == 5


async def test_ntfy_failure_is_returned_without_raising() -> None:
    def transport(_request: Request, _timeout: float) -> None:
        raise TimeoutError("ntfy unavailable")

    publisher = NtfyPublisher(
        NtfyConfig(enabled=True, topic="mry_private"),
        transport=transport,
    )

    result = await publisher.send(alert())

    assert result.delivered is False
    assert result.error == "ntfy unavailable"


def test_ntfy_failure_does_not_interrupt_normal_alert_pipeline(tmp_path) -> None:
    def transport(_request: Request, _timeout: float) -> None:
        raise OSError("push server offline")

    config = AppConfig()
    config.ntfy = NtfyConfig(enabled=True, topic="mry_private")
    publisher = NtfyPublisher(config.ntfy, transport=transport)
    app = create_app(
        config,
        token="pairing-token",
        event_path=tmp_path / "events.jsonl",
        ntfy_publisher=publisher,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/test-alert",
            headers={"X-Pairing-Token": "pairing-token"},
        )
        status = client.get("/api/status").json()

    assert response.status_code == 200
    assert status["ntfy_enabled"] is True
    assert status["ntfy_topic"] == "mry_private"
    assert status["ntfy_authorization_configured"] is False
    assert status["ntfy_last_success"] is False
    assert status["ntfy_last_error"] == "push server offline"


async def test_disabled_ntfy_never_calls_transport() -> None:
    called = False

    def transport(_request: Request, _timeout: float) -> None:
        nonlocal called
        called = True

    result = await NtfyPublisher(NtfyConfig(), transport=transport).send(alert())

    assert result.delivered is False
    assert called is False
