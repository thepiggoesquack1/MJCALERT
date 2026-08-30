from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import UTC, datetime, timedelta

import httpx

from mry_alert.config import AdsbConfig, AirportConfig
from mry_alert.models import NearbyAircraft

logger = logging.getLogger(__name__)


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return None
    return None


class AdsbLolNearbyAircraftProvider:
    def __init__(
        self, airport: AirportConfig, config: AdsbConfig, client: httpx.AsyncClient | None = None
    ) -> None:
        self.airport = airport
        self.config = config
        self._client = client
        self._cache: list[NearbyAircraft] = []
        self._last_poll = 0.0
        self._failures = 0
        self.last_success_at: datetime | None = None
        self.last_error: str | None = None

    async def nearby(self) -> list[NearbyAircraft]:
        now = time.monotonic()
        if now - self._last_poll < self.config.polling_interval_seconds:
            return self._cache
        if self._failures:
            backoff = min(60.0, 2.0**self._failures)
            if now - self._last_poll < backoff:
                return self._cache
        self._last_poll = now
        primary_url, fallback_url = self._endpoint_urls()
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            headers=self._request_headers(),
        )
        try:
            response = await self._get_with_route_fallback(
                client, primary_url, fallback_url
            )
            items = response.json().get("ac", [])
            fetched_at = datetime.now(UTC)
            self._cache = [self._parse(item, fetched_at) for item in items]
            self._failures = 0
            self.last_success_at = fetched_at
            self.last_error = None
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            self._failures += 1
            self.last_error = str(exc)
            logger.warning("ADS-B nearby lookup failed; using cached result: %s", exc)
        finally:
            if owns_client:
                await client.aclose()
        return self._cache

    def _endpoint_urls(self) -> tuple[str, str | None]:
        primary_url = (
            f"{self.config.base_url.rstrip('/')}/v2/lat/{self.airport.latitude}"
            f"/lon/{self.airport.longitude}/dist/{self.airport.adsb_radius_nm}"
        )
        fallback_url = (
            f"{self.config.base_url.rstrip('/')}/v2/point/{self.airport.latitude}"
            f"/{self.airport.longitude}/{self.airport.adsb_radius_nm}"
        )
        return primary_url, fallback_url

    def _request_headers(self) -> dict[str, str]:
        return {"User-Agent": "MRY-Jet-Center-Alert/0.1 (local advisory tool)"}

    @staticmethod
    async def _get_with_route_fallback(
        client: httpx.AsyncClient, primary_url: str, fallback_url: str | None
    ) -> httpx.Response:
        try:
            response = await client.get(primary_url)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500 or fallback_url is None:
                raise
            logger.info(
                "ADS-B primary route returned HTTP %d; retrying documented point route",
                exc.response.status_code,
            )
        response = await client.get(fallback_url)
        response.raise_for_status()
        return response

    @staticmethod
    def _parse(item: dict[str, object], fetched_at: datetime | None = None) -> NearbyAircraft:
        altitude = item.get("alt_baro")
        on_ground = altitude == "ground"
        seconds_since_seen = _optional_float(item.get("seen"))
        flight = str(item["flight"]).strip().upper() if item.get("flight") else None
        flight_prefix_match = re.match(r"([A-Z]{3})", flight or "")
        provided_designator = item.get("icao") or item.get("icao_designator")
        icao_designator = (
            str(provided_designator).strip().upper()
            if provided_designator
            else flight_prefix_match.group(1)
            if flight_prefix_match
            else None
        )
        source_timestamp = (
            fetched_at - timedelta(seconds=seconds_since_seen)
            if fetched_at is not None and seconds_since_seen is not None
            else None
        )
        type_code_value = item.get("t") or item.get("type_code") or item.get("icao_type")
        type_code = str(type_code_value).strip().upper() if type_code_value else None
        manufacturer_value = item.get("manufacturer") or item.get("mfr")
        manufacturer = str(manufacturer_value).strip() if manufacturer_value else None
        model_value = item.get("model") or item.get("aircraft_model")
        model = str(model_value).strip() if model_value else None
        type_name_value = item.get("aircraft_type_name") or item.get("type_name")
        type_name = str(type_name_value).strip() if type_name_value else None
        category_value = (
            item.get("category") or item.get("categoryDescription") or item.get("desc")
        )
        category = str(category_value).strip() if category_value else None
        descriptive_type = " ".join(part for part in (manufacturer, model) if part).strip()
        display_type = descriptive_type or type_name or type_code
        type_confidence = 1.0 if descriptive_type or type_name else 0.95 if type_code else 0.0
        return NearbyAircraft(
            hex=str(item.get("hex", "unknown")),
            registration=str(item["r"]).strip().upper() if item.get("r") else None,
            flight=flight,
            operator_name=(
                str(item.get("ownOp") or item.get("operator")).strip()
                if item.get("ownOp") or item.get("operator")
                else None
            ),
            icao_designator=icao_designator,
            iata_code=(str(item["iata"]).strip().upper() if item.get("iata") else None),
            aircraft_type=display_type,
            aircraft_type_code=type_code,
            aircraft_type_name=type_name or descriptive_type or None,
            manufacturer=manufacturer,
            model=model,
            aircraft_category=category,
            aircraft_type_source="adsb_provider" if display_type or category else "unknown",
            aircraft_type_confidence=type_confidence,
            latitude=_optional_float(item.get("lat")),
            longitude=_optional_float(item.get("lon")),
            altitude=altitude if isinstance(altitude, (float, str)) else None,
            on_ground=on_ground,
            ground_speed=_optional_float(item.get("gs")),
            seconds_since_seen=seconds_since_seen,
            track=_optional_float(item.get("track")),
            vertical_rate=_optional_float(item.get("baro_rate")),
            squawk=str(item["squawk"]) if item.get("squawk") else None,
            source_timestamp=source_timestamp,
            distance_nm=_optional_float(item.get("dst")),
        )

    async def wait_after_failure(self) -> None:
        """Available to long-running pollers; regular calls remain non-blocking."""
        await asyncio.sleep(min(60.0, 2.0**self._failures))
