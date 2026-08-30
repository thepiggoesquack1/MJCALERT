from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from rapidfuzz.fuzz import partial_ratio

from mry_alert.adsb.tracker import AdsbCorrelator
from mry_alert.config import (
    AdsbConfig,
    AdsbDecisionConfig,
    AdsbGeofencesConfig,
    AdsbMatchingConfig,
    AdsbScoringConfig,
    AdsbTrackingConfig,
    AirportConfig,
    DetectionConfig,
)
from mry_alert.detection.callsign import DIGITS, PHONETIC
from mry_alert.models import (
    AircraftMatch,
    IdentificationSource,
    MatchStatus,
    NearbyAircraft,
    SpokenCallsign,
)


@dataclass
class _Score:
    aircraft: NearbyAircraft
    value: float
    reasons: list[str]
    source: IdentificationSource


class AircraftMatcher:
    def __init__(
        self,
        detection: DetectionConfig,
        adsb: AdsbConfig,
        matching: AdsbMatchingConfig | None = None,
        airport: AirportConfig | None = None,
        tracking: AdsbTrackingConfig | None = None,
        geofences: AdsbGeofencesConfig | None = None,
        scoring: AdsbScoringConfig | None = None,
        decision: AdsbDecisionConfig | None = None,
    ) -> None:
        self.detection = detection
        self.adsb = adsb
        self.matching = matching or AdsbMatchingConfig(
            maximum_seen_seconds=adsb.maximum_seen_seconds,
            maximum_ground_speed_knots=adsb.maximum_taxi_speed_knots,
            ambiguity_margin=detection.ambiguity_margin,
        )
        self.correlator = (
            AdsbCorrelator(
                airport,
                tracking,
                geofences or AdsbGeofencesConfig(),
                self.matching,
                scoring or AdsbScoringConfig(),
                decision or AdsbDecisionConfig(),
            )
            if airport and tracking and tracking.enabled
            else None
        )

    def observe(self, aircraft: list[NearbyAircraft]) -> None:
        """Feed background ADS-B observations into movement history without matching."""
        if self.correlator is not None:
            self.correlator.tracker.update(aircraft)

    def _plausible(self, aircraft: NearbyAircraft) -> bool:
        altitude = aircraft.altitude
        low = (
            aircraft.on_ground
            or altitude == "ground"
            or (
                isinstance(altitude, (int, float))
                and altitude <= self.matching.maximum_low_altitude_feet
            )
        )
        if not low:
            return False
        if (
            aircraft.ground_speed is not None
            and aircraft.ground_speed > self.matching.maximum_ground_speed_knots
        ):
            return False
        if self.matching.require_recent_position and (
            aircraft.seconds_since_seen is None
            or aircraft.seconds_since_seen > self.matching.maximum_seen_seconds
        ):
            return False
        return not (
            aircraft.distance_nm is not None and aircraft.distance_nm > self.matching.radius_nm
        )

    @staticmethod
    def _registration_variants(registration: str) -> list[str]:
        value = registration.upper().replace("-", "")
        if not value.startswith("N") or len(value) < 3:
            return []
        digit_words = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
        }
        phonetic_words = {value: key for key, value in PHONETIC.items() if key != "alfa"}
        words = [
            digit_words.get(character, phonetic_words.get(character, character.lower()))
            for character in value[1:]
        ]
        suffix = " ".join(words)
        return [f"november {suffix}", suffix, f"citation {suffix}"]

    def _movement_reasons(self, aircraft: NearbyAircraft) -> tuple[float, list[str]]:
        value = 0.0
        reasons: list[str] = []
        if aircraft.on_ground:
            value += 0.1
            reasons.append("aircraft reported on ground")
        if aircraft.ground_speed is not None:
            value += 0.07
            reasons.append(f"ground speed {aircraft.ground_speed:g} kt is consistent with taxiing")
        if aircraft.seconds_since_seen is not None:
            value += 0.06
            reasons.append(f"aircraft seen {aircraft.seconds_since_seen:g} seconds ago")
        return value, reasons

    def _identity_score(
        self,
        spoken: SpokenCallsign,
        aircraft: NearbyAircraft,
        suffix_count: int,
    ) -> _Score | None:
        registration = (aircraft.registration or "").upper().replace("-", "")
        flight = (aircraft.flight or "").upper().replace(" ", "")
        value = 0.0
        reasons: list[str] = []
        source = IdentificationSource.UNRESOLVED
        if spoken.full_registration and registration == spoken.full_registration:
            value = 0.78
            source = IdentificationSource.SPOKEN_FULL_REGISTRATION
            reasons.append("full registration matched ADS-B registration")
        elif spoken.full_registration and flight == spoken.full_registration:
            value = 0.7
            source = IdentificationSource.SPOKEN_CALLSIGN_ADSB_MATCH
            reasons.append("full spoken registration matched ADS-B flight callsign")
        elif spoken.suffix and (
            registration.endswith(spoken.suffix) or flight.endswith(spoken.suffix)
        ):
            value = 0.62 if suffix_count == 1 else 0.46
            source = IdentificationSource.UNIQUE_SUFFIX_ADSB_MATCH
            reasons.append(
                "unique suffix matched nearby registration"
                if suffix_count == 1
                else "suffix matched multiple nearby aircraft"
            )
        else:
            return None
        movement, movement_reasons = self._movement_reasons(aircraft)
        if (
            spoken.aircraft_type_prefix
            and aircraft.aircraft_type
            and spoken.aircraft_type_prefix.lower() in aircraft.aircraft_type.lower()
        ):
            movement += 0.04
            movement_reasons.append("spoken aircraft type agreed with ADS-B metadata")
        return _Score(aircraft, min(1.0, value + movement), reasons + movement_reasons, source)

    def _fuzzy_scores(self, raw_text: str, aircraft: list[NearbyAircraft]) -> list[_Score]:
        if not self.matching.fuzzy_callsign_matching:
            return []
        known_tokens = set(DIGITS) | set(PHONETIC)
        if len([token for token in raw_text.split() if token in known_tokens]) < 3:
            return []
        scores: list[_Score] = []
        for item in aircraft:
            variants = self._registration_variants(item.registration or "")
            fuzzy = max((partial_ratio(raw_text, variant) for variant in variants), default=0)
            if fuzzy < self.matching.fuzzy_minimum_score:
                continue
            movement, movement_reasons = self._movement_reasons(item)
            scores.append(
                _Score(
                    item,
                    min(0.89, fuzzy / 100 * 0.72 + movement),
                    [f"fuzzy spoken-form score {fuzzy:.0f}", *movement_reasons],
                    IdentificationSource.FUZZY_ADSB_RECOVERY,
                )
            )
        return scores

    def match(
        self,
        spoken: SpokenCallsign | None,
        aircraft: list[NearbyAircraft],
        raw_callsign_text: str | None = None,
    ) -> AircraftMatch:
        if self.correlator is not None:
            return self.correlator.correlate(spoken, aircraft)
        snapshot = datetime.now(UTC)
        plausible = [item for item in aircraft if self._plausible(item)]
        suffix = spoken.full_registration or spoken.suffix or "" if spoken else ""
        suffix_count = sum(
            1
            for item in plausible
            if suffix and (item.registration or "").upper().replace("-", "").endswith(suffix)
        )
        scores = [
            score
            for item in plausible
            if spoken and (score := self._identity_score(spoken, item, suffix_count))
        ]
        if not scores and raw_callsign_text:
            scores = self._fuzzy_scores(raw_callsign_text, plausible)
        scores.sort(key=lambda item: item.value, reverse=True)
        candidate_scores = [
            f"{score.aircraft.registration or score.aircraft.hex} score={score.value:.2f}: "
            + "; ".join(score.reasons)
            for score in scores
        ]
        if scores:
            top = scores[0]
            alternatives = [
                item.aircraft
                for item in scores[1:]
                if top.value - item.value < self.matching.ambiguity_margin
            ]
            if alternatives:
                return AircraftMatch(
                    confidence=top.value,
                    status=MatchStatus.AMBIGUOUS,
                    match_reasons=[*top.reasons, "candidate scores within ambiguity margin"],
                    alternative_candidates=[top.aircraft, *alternatives],
                    raw_callsign_text=raw_callsign_text,
                    normalized_callsign=spoken.normalized_form if spoken else None,
                    candidate_scores=candidate_scores,
                    adsb_snapshot_timestamp=snapshot,
                )
            confirmed = bool(
                spoken
                and spoken.full_registration
                and top.source != IdentificationSource.FUZZY_ADSB_RECOVERY
                and top.value >= 0.8
            )
            return AircraftMatch(
                aircraft=top.aircraft,
                registration=top.aircraft.registration,
                confidence=top.value,
                status=MatchStatus.CONFIRMED if confirmed else MatchStatus.LIKELY,
                match_reasons=top.reasons,
                raw_callsign_text=raw_callsign_text,
                normalized_callsign=spoken.normalized_form if spoken else None,
                identification_source=top.source,
                candidate_scores=candidate_scores,
                adsb_snapshot_timestamp=snapshot,
            )
        if spoken and spoken.full_registration:
            return AircraftMatch(
                registration=spoken.full_registration,
                confidence=0.82,
                status=MatchStatus.LIKELY,
                match_reasons=["full spoken registration retained without ADS-B confirmation"],
                raw_callsign_text=raw_callsign_text,
                normalized_callsign=spoken.normalized_form,
                identification_source=IdentificationSource.SPOKEN_FULL_REGISTRATION,
                adsb_snapshot_timestamp=snapshot,
            )
        raw_tokens = (raw_callsign_text or "").split()
        known_count = len([token for token in raw_tokens if token in set(DIGITS) | set(PHONETIC)])
        unsafe_short_fragment = 0 < known_count < 3 and len(raw_tokens) <= 3
        if (
            self.matching.allow_unique_ground_candidate_fallback
            and len(plausible) == 1
            and not unsafe_short_fragment
            and spoken is None
        ):
            candidate = plausible[0]
            return AircraftMatch(
                aircraft=candidate,
                registration=candidate.registration,
                confidence=self.matching.unique_candidate_minimum_confidence,
                status=MatchStatus.LIKELY,
                match_reasons=[
                    "exactly one recent low-speed ground aircraft was plausible",
                    "proximity alone was not treated as confirmation",
                ],
                raw_callsign_text=raw_callsign_text,
                identification_source=IdentificationSource.UNIQUE_GROUND_CANDIDATE,
                adsb_snapshot_timestamp=snapshot,
            )
        return AircraftMatch(
            confidence=0,
            status=MatchStatus.AMBIGUOUS if len(plausible) > 1 else MatchStatus.UNRESOLVED,
            match_reasons=[
                "multiple plausible ground aircraft without usable callsign"
                if len(plausible) > 1
                else "no ADS-B identity match"
            ],
            alternative_candidates=plausible if len(plausible) > 1 else [],
            raw_callsign_text=raw_callsign_text,
            adsb_snapshot_timestamp=snapshot,
        )
