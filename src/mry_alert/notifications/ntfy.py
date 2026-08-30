from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from urllib.request import Request, urlopen

from mry_alert.config import NtfyConfig
from mry_alert.models import AlertEvent


@dataclass(frozen=True)
class NtfyDeliveryResult:
    delivered: bool
    error: str | None = None


Transport = Callable[[Request, float], None]


def _send_request(request: Request, timeout: float) -> None:
    with urlopen(request, timeout=timeout):
        return


class NtfyPublisher:
    def __init__(
        self,
        config: NtfyConfig,
        *,
        transport: Transport = _send_request,
        timeout_seconds: float = 5,
    ) -> None:
        self.config = config
        self._transport = transport
        self._timeout_seconds = timeout_seconds

    @property
    def subscription_url(self) -> str | None:
        if not self.config.topic:
            return None
        return f"{self.config.server_url}/{self.config.topic}"

    async def send(self, event: AlertEvent) -> NtfyDeliveryResult:
        if not self.config.enabled:
            return NtfyDeliveryResult(False, "ntfy is disabled")
        assert self.subscription_url is not None
        aircraft = event.registration or event.spoken_callsign or "Unknown aircraft"
        aircraft_type = (
            event.aircraft_type_name or event.aircraft_type or "Aircraft type unknown"
        )
        source = event.identification_source.value
        body = (
            f"Going to {event.corrected_destination or event.destination}\n"
            f"Confidence: {event.confidence:.0%}\n"
            f"ADS-B: {event.status.value}\n"
            f"ADS-B source: {source}\n"
            f"Timestamp: {event.timestamp.isoformat()}"
        )
        headers = {
            "Content-Type": "text/plain; charset=utf-8",
            # urllib serializes HTTP header values as Latin-1. Keep this title
            # ASCII-safe so an em dash cannot fail before the request is sent.
            "Title": f"{aircraft} - {aircraft_type}",
            "Tags": "airplane",
        }
        if self.config.authorization:
            headers["Authorization"] = self.config.authorization
        request = Request(
            self.subscription_url,
            data=body.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            await asyncio.to_thread(self._transport, request, self._timeout_seconds)
        except Exception as exc:
            return NtfyDeliveryResult(False, str(exc))
        return NtfyDeliveryResult(True)
