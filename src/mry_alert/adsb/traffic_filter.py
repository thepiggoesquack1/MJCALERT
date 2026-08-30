from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum

from mry_alert.adsb.base import NearbyAircraftProvider
from mry_alert.config import TrafficFilterConfig
from mry_alert.models import NearbyAircraft

logger = logging.getLogger("mry_alert.operations")


class TrafficDecision(StrEnum):
    ALLOWED = "allowed"
    ALLOWED_OVERRIDE = "allowed_override"
    IGNORED = "ignored_scheduled_airline"
    UNKNOWN = "unknown_operator_allowed"


@dataclass(frozen=True)
class TrafficFilterResult:
    decision: TrafficDecision
    reason: str

    @property
    def allowed(self) -> bool:
        return self.decision != TrafficDecision.IGNORED


@dataclass(frozen=True)
class _ScheduledOperator:
    icao: str
    iata: str
    name: str
    callsign: str


# Deliberately limited to clearly scheduled passenger airlines. Private, charter,
# fractional, medevac, military, and uncertain identities are not inferred here.
SCHEDULED_OPERATORS = (
    _ScheduledOperator("UAL", "UA", "united airlines", "UNITED"),
    _ScheduledOperator("ASA", "AS", "alaska airlines", "ALASKA"),
    _ScheduledOperator("AAL", "AA", "american airlines", "AMERICAN"),
    _ScheduledOperator("DAL", "DL", "delta air lines", "DELTA"),
    _ScheduledOperator("SWA", "WN", "southwest airlines", "SOUTHWEST"),
    _ScheduledOperator("JBU", "B6", "jetblue airways", "JETBLUE"),
    _ScheduledOperator("NKS", "NK", "spirit airlines", "SPIRIT"),
    _ScheduledOperator("FFT", "F9", "frontier airlines", "FRONTIER"),
    _ScheduledOperator("AAY", "G4", "allegiant air", "ALLEGIANT"),
    _ScheduledOperator("HAL", "HA", "hawaiian airlines", "HAWAIIAN"),
    _ScheduledOperator("SKW", "OO", "skywest airlines", "SKYWEST"),
    _ScheduledOperator("RPA", "YX", "republic airways", "BRICKYARD"),
    _ScheduledOperator("ENY", "MQ", "envoy air", "ENVOY"),
    _ScheduledOperator("JIA", "OH", "psa airlines", "BLUE STREAK"),
    _ScheduledOperator("PDT", "PT", "piedmont airlines", "PIEDMONT"),
    _ScheduledOperator("ACA", "AC", "air canada", "AIR CANADA"),
    _ScheduledOperator("BAW", "BA", "british airways", "SPEEDBIRD"),
    _ScheduledOperator("DLH", "LH", "lufthansa", "LUFTHANSA"),
    _ScheduledOperator("AFR", "AF", "air france", "AIRFRANS"),
    _ScheduledOperator("KLM", "KL", "klm", "KLM"),
    _ScheduledOperator("VIR", "VS", "virgin atlantic", "VIRGIN"),
    _ScheduledOperator("UAE", "EK", "emirates", "EMIRATES"),
    _ScheduledOperator("QTR", "QR", "qatar airways", "QATARI"),
    _ScheduledOperator("SIA", "SQ", "singapore airlines", "SINGAPORE"),
    _ScheduledOperator("CPA", "CX", "cathay pacific", "CATHAY"),
    _ScheduledOperator("QFA", "QF", "qantas", "QANTAS"),
    _ScheduledOperator("ANZ", "NZ", "air new zealand", "NEW ZEALAND"),
)

BUILTIN_ICAO = {item.icao for item in SCHEDULED_OPERATORS}
BUILTIN_IATA = {item.iata for item in SCHEDULED_OPERATORS}
BUILTIN_NAMES = {item.name for item in SCHEDULED_OPERATORS}
BUILTIN_CALLSIGNS = {item.callsign for item in SCHEDULED_OPERATORS}
NUMBER_WORDS = {
    "ZERO",
    "OH",
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "NINER",
}


def _normalized(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip() if value else ""


def _prefix(value: str | None) -> str:
    normalized = _normalized(value).replace(" ", "")
    match = re.match(r"([A-Z]{2,})", normalized)
    return match.group(1) if match else ""


class TrafficFilter:
    def __init__(self, config: TrafficFilterConfig) -> None:
        self.config = config
        self._filtered: set[str] = set()
        self._overrides: set[str] = set()
        self._unknown: set[str] = set()
        self.recent_decision = "none"
        self.recent_reason = "No aircraft evaluated"

    @property
    def filtered_airline_count(self) -> int:
        return len(self._filtered)

    @property
    def allowed_override_count(self) -> int:
        return len(self._overrides)

    @property
    def unknown_operator_count(self) -> int:
        return len(self._unknown)

    def snapshot(self) -> dict[str, object]:
        return {
            "traffic_filter_enabled": self.config.enabled,
            "filtered_airline_count": self.filtered_airline_count,
            "allowed_override_count": self.allowed_override_count,
            "unknown_operator_count": self.unknown_operator_count,
            "recent_traffic_filter_decision": self.recent_decision,
            "recent_traffic_filter_reason": self.recent_reason,
        }

    def _record(
        self,
        key: str,
        result: TrafficFilterResult,
        *,
        aircraft: str,
        operator: str,
    ) -> TrafficFilterResult:
        self.recent_decision = result.decision.value
        self.recent_reason = result.reason
        if result.decision == TrafficDecision.IGNORED:
            first = key not in self._filtered
            self._filtered.add(key)
            if first and self.config.log_filtered_aircraft:
                logger.info(
                    "TRAFFIC FILTER\nAircraft: %s\nOperator: %s\nDecision: "
                    "ignored scheduled airline\nReason: %s",
                    aircraft or "unknown",
                    operator or "unknown",
                    result.reason,
                )
        elif result.decision == TrafficDecision.ALLOWED_OVERRIDE:
            first = key not in self._overrides
            self._overrides.add(key)
            if first and self.config.log_filtered_aircraft:
                logger.info(
                    "TRAFFIC FILTER\nAircraft: %s\nOperator: %s\nDecision: "
                    "allowed override\nReason: %s",
                    aircraft or "unknown",
                    operator or "unknown",
                    result.reason,
                )
        elif result.decision == TrafficDecision.UNKNOWN:
            self._unknown.add(key)
        return result

    def evaluate_aircraft(self, aircraft: NearbyAircraft) -> TrafficFilterResult:
        if not self.config.enabled:
            return TrafficFilterResult(TrafficDecision.ALLOWED, "traffic filter disabled")
        flight = _normalized(aircraft.flight)
        flight_prefix = _prefix(aircraft.flight)
        operator = _normalized(aircraft.operator_name)
        icao = _normalized(aircraft.icao_designator)
        iata = _normalized(aircraft.iata_code)
        key = aircraft.hex or aircraft.registration or flight or "unknown"

        allowed_names = {
            _normalized(value) for value in [*self.config.allowed_operator_overrides, "JSX"]
        }
        allowed_icao = {
            _normalized(value) for value in [*self.config.allowed_icao_designators, "JSX"]
        }
        allowed_iata = {_normalized(value) for value in self.config.allowed_iata_codes}
        if (
            icao in allowed_icao
            or flight_prefix in allowed_icao
            or iata in allowed_iata
            or any(value and value in operator for value in allowed_names)
            or flight_prefix == "JSX"
        ):
            return self._record(
                key,
                TrafficFilterResult(TrafficDecision.ALLOWED_OVERRIDE, "JSX is explicitly allowed"),
                aircraft=aircraft.flight or aircraft.registration or aircraft.hex,
                operator=aircraft.operator_name or aircraft.icao_designator or "JSX",
            )

        ignored_names = {_normalized(value) for value in self.config.ignored_operator_names}
        ignored_icao = {_normalized(value) for value in self.config.ignored_icao_designators}
        ignored_iata = {_normalized(value) for value in self.config.ignored_iata_codes}
        ignored_prefixes = {
            _normalized(value).replace(" ", "") for value in self.config.ignored_callsign_prefixes
        }
        explicit_reason: str | None = None
        if icao and icao in ignored_icao:
            explicit_reason = f"matched ignored ICAO designator {icao}"
        elif iata and iata in ignored_iata:
            explicit_reason = f"matched ignored IATA code {iata}"
        elif flight_prefix and flight_prefix in ignored_prefixes:
            explicit_reason = f"matched ignored ADS-B flight prefix {flight_prefix}"
        elif any(value and value in operator for value in ignored_names):
            explicit_reason = "matched ignored operator name"
        if explicit_reason:
            return self._record(
                key,
                TrafficFilterResult(TrafficDecision.IGNORED, explicit_reason),
                aircraft=aircraft.flight or aircraft.registration or aircraft.hex,
                operator=aircraft.operator_name or aircraft.icao_designator or "unknown",
            )

        heuristic_reason: str | None = None
        if self.config.ignore_scheduled_airlines:
            if icao in BUILTIN_ICAO:
                heuristic_reason = f"matched scheduled-airline ICAO designator {icao}"
            elif iata in BUILTIN_IATA:
                heuristic_reason = f"matched scheduled-airline IATA code {iata}"
            elif flight_prefix in BUILTIN_ICAO:
                heuristic_reason = f"matched scheduled-airline ADS-B flight prefix {flight_prefix}"
            elif flight_prefix in BUILTIN_CALLSIGNS:
                heuristic_reason = f"matched scheduled-airline callsign prefix {flight_prefix}"
            elif any(_normalized(name) in operator for name in BUILTIN_NAMES):
                heuristic_reason = "matched known scheduled-airline operator name"
        if heuristic_reason:
            return self._record(
                key,
                TrafficFilterResult(TrafficDecision.IGNORED, heuristic_reason),
                aircraft=aircraft.flight or aircraft.registration or aircraft.hex,
                operator=aircraft.operator_name or aircraft.icao_designator or "unknown",
            )
        if not any((operator, icao, iata, flight_prefix)):
            return self._record(
                key,
                TrafficFilterResult(
                    TrafficDecision.UNKNOWN,
                    "operator identity incomplete; default allow",
                ),
                aircraft=aircraft.registration or aircraft.hex,
                operator="unknown",
            )
        return TrafficFilterResult(TrafficDecision.ALLOWED, "no airline ignore rule matched")

    def evaluate_transcript(self, text: str) -> TrafficFilterResult:
        if not self.config.enabled:
            return TrafficFilterResult(TrafficDecision.ALLOWED, "traffic filter disabled")
        normalized = _normalized(text)
        tokens = normalized.split()
        configured_allow = {
            _normalized(value) for value in [*self.config.allowed_operator_overrides, "JSX"]
        }
        configured_ignore = {_normalized(value) for value in self.config.ignored_callsign_prefixes}
        callsigns = BUILTIN_CALLSIGNS if self.config.ignore_scheduled_airlines else set()
        for index, token in enumerate(tokens):
            nearby = tokens[index + 1 : index + 5]
            token_prefix = _prefix(token)
            first_suffix_token = nearby[0] if nearby else ""
            has_number = (
                first_suffix_token in NUMBER_WORDS
                or bool(re.search(r"\d", first_suffix_token))
                or bool(re.search(r"\d", token))
            )
            if not has_number:
                continue
            if (
                token == "JSX"
                or token_prefix == "JSX"
                or any(allowed and normalized.startswith(allowed) for allowed in configured_allow)
            ):
                return self._record(
                    f"transcript:{' '.join(tokens[index : index + 5])}",
                    TrafficFilterResult(
                        TrafficDecision.ALLOWED_OVERRIDE, "JSX is explicitly allowed"
                    ),
                    aircraft=" ".join(tokens[index : index + 5]),
                    operator="JSX",
                )
            ignored_icao = BUILTIN_ICAO if self.config.ignore_scheduled_airlines else set()
            if token in configured_ignore or token in callsigns or token_prefix in ignored_icao:
                return self._record(
                    f"transcript:{' '.join(tokens[index : index + 5])}",
                    TrafficFilterResult(
                        TrafficDecision.IGNORED,
                        "matched clearly identified airline callsign prefix "
                        f"{token_prefix or token}",
                    ),
                    aircraft=" ".join(tokens[index : index + 5]),
                    operator=token,
                )
        return TrafficFilterResult(
            TrafficDecision.ALLOWED, "no clearly identified airline callsign"
        )

    def filter(self, aircraft: list[NearbyAircraft]) -> list[NearbyAircraft]:
        return [item for item in aircraft if self.evaluate_aircraft(item).allowed]


class TrafficFilteringProvider:
    def __init__(self, source: NearbyAircraftProvider, traffic_filter: TrafficFilter) -> None:
        self.source = source
        self.traffic_filter = traffic_filter
        self._cache: list[NearbyAircraft] = []

    @property
    def last_success_at(self) -> object:
        return getattr(self.source, "last_success_at", None)

    @property
    def last_error(self) -> object:
        return getattr(self.source, "last_error", None)

    async def nearby(self) -> list[NearbyAircraft]:
        self._cache = self.traffic_filter.filter(await self.source.nearby())
        return self._cache
