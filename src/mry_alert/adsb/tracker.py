from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from rapidfuzz.fuzz import ratio

from mry_alert.config import (
    AdsbDecisionConfig,
    AdsbGeofencesConfig,
    AdsbMatchingConfig,
    AdsbScoringConfig,
    AdsbTrackingConfig,
    AirportConfig,
    CircleGeofenceConfig,
    GeofenceConfig,
    PolygonGeofenceConfig,
)
from mry_alert.models import (
    AdsbDecisionState,
    AircraftMatch,
    IdentificationSource,
    MatchStatus,
    NearbyAircraft,
    SpokenCallsign,
)


class MovementState(StrEnum):
    AIRBORNE_INBOUND = "airborne_inbound"
    RECENTLY_LANDED = "recently_landed"
    TAXIING = "taxiing"
    STOPPED_ON_AIRPORT = "stopped_on_airport"
    MOVING_TOWARD_JET_CENTER = "moving_toward_jet_center"
    MOVING_AWAY_FROM_JET_CENTER = "moving_away_from_jet_center"
    PARKED_AT_JET_CENTER = "parked_at_jet_center"
    DEPARTING = "departing"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    aircraft: NearbyAircraft


@dataclass
class TrackedContact:
    hex: str
    first_seen: datetime
    last_seen: datetime
    observations: list[Observation] = field(default_factory=list)


@dataclass(frozen=True)
class CandidateAnalysis:
    aircraft: NearbyAircraft
    state: MovementState
    score: float
    score_breakdown: list[str]
    current_distance_nm: float | None
    previous_distance_nm: float | None
    distance_trend_nm: float | None
    moving_toward_destination: bool
    trend_confidence: float
    recently_landed: bool
    recently_landed_confidence: float
    age_seconds: float
    inside_fbo_geofence: bool
    fbo_geofence_names: list[str]
    was_stationary_at_fbo: bool


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_nm = 3440.065
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return radius_nm * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _inside(aircraft: NearbyAircraft, geofence: GeofenceConfig | None) -> bool:
    if geofence is None or aircraft.latitude is None or aircraft.longitude is None:
        return False
    if isinstance(geofence, CircleGeofenceConfig):
        return bool(
            geofence.latitude is not None
            and geofence.longitude is not None
            and haversine_nm(
                aircraft.latitude,
                aircraft.longitude,
                geofence.latitude,
                geofence.longitude,
            )
            <= geofence.radius_nm
        )
    assert isinstance(geofence, PolygonGeofenceConfig)
    latitude = aircraft.latitude
    longitude = aircraft.longitude
    inside = False
    coordinates = geofence.coordinates
    for index, (lat_a, lon_a) in enumerate(coordinates):
        lat_b, lon_b = coordinates[index - 1]
        if (lon_a > longitude) != (lon_b > longitude):
            crossing_latitude = (lat_b - lat_a) * (longitude - lon_a) / (lon_b - lon_a) + lat_a
            if latitude < crossing_latitude:
                inside = not inside
    return inside


def _center(geofence: GeofenceConfig | None) -> tuple[float, float] | None:
    if isinstance(geofence, CircleGeofenceConfig):
        if geofence.latitude is None or geofence.longitude is None:
            return None
        return geofence.latitude, geofence.longitude
    if isinstance(geofence, PolygonGeofenceConfig):
        points = geofence.coordinates
        if len(points) > 1 and points[0] == points[-1]:
            points = points[:-1]
        return (
            sum(latitude for latitude, _ in points) / len(points),
            sum(longitude for _, longitude in points) / len(points),
        )
    return None


class AdsbContactTracker:
    def __init__(self, config: AdsbTrackingConfig) -> None:
        self.config = config
        self.contacts: dict[str, TrackedContact] = {}

    def update(
        self, aircraft: list[NearbyAircraft], now: datetime | None = None
    ) -> list[TrackedContact]:
        now = now or datetime.now(UTC)
        history_cutoff = now - timedelta(seconds=self.config.history_seconds)
        for item in aircraft:
            observed = item.source_timestamp or now - timedelta(
                seconds=item.seconds_since_seen or 0
            )
            contact = self.contacts.setdefault(
                item.hex, TrackedContact(item.hex, observed, observed)
            )
            contact.last_seen = max(contact.last_seen, observed)
            if not contact.observations or contact.observations[-1].timestamp != observed:
                contact.observations.append(Observation(observed, item))
            contact.observations = [
                point for point in contact.observations if point.timestamp >= history_cutoff
            ]
        purge_cutoff = now - timedelta(seconds=self.config.purge_after_seconds)
        self.contacts = {
            key: contact
            for key, contact in self.contacts.items()
            if contact.last_seen >= purge_cutoff
        }
        return list(self.contacts.values())


class AdsbCorrelator:
    def __init__(
        self,
        airport: AirportConfig,
        tracking: AdsbTrackingConfig,
        geofences: AdsbGeofencesConfig,
        matching: AdsbMatchingConfig,
        scoring: AdsbScoringConfig,
        decision: AdsbDecisionConfig,
    ) -> None:
        self.tracker = AdsbContactTracker(tracking)
        self.tracking = tracking
        self.geofences = geofences.model_copy(deep=True)
        if self.geofences.kmry_airport.latitude is None:
            self.geofences.kmry_airport.latitude = airport.latitude
            self.geofences.kmry_airport.longitude = airport.longitude
        self.matching = matching
        self.scoring = scoring
        self.decision = decision

    def _analyze(
        self, contact: TrackedContact, spoken: SpokenCallsign | None, now: datetime
    ) -> CandidateAnalysis:
        points = contact.observations
        aircraft = points[-1].aircraft
        age = max(0.0, (now - contact.last_seen).total_seconds())
        jet = self.geofences.monterey_jet_center
        fbo_geofences = {
            "Monterey Jet Center": jet,
            **self.geofences.fbo_geofences,
        }
        fbo_names = [
            name for name, geofence in fbo_geofences.items() if _inside(aircraft, geofence)
        ]
        motion_geofence = (
            fbo_geofences[fbo_names[0]] if fbo_names else self.geofences.monterey_jet_center
        )
        center = _center(motion_geofence)
        distances = (
            [
                haversine_nm(
                    point.aircraft.latitude,
                    point.aircraft.longitude,
                    center[0],
                    center[1],
                )
                for point in points
                if point.aircraft.latitude is not None and point.aircraft.longitude is not None
            ]
            if center
            else []
        )
        current_position_complete = aircraft.latitude is not None and aircraft.longitude is not None
        enough_track_points = len(distances) >= self.matching.minimum_track_points
        current_distance = distances[-1] if current_position_complete and distances else None
        previous_distance = (
            distances[-2]
            if current_position_complete and enough_track_points and len(distances) >= 2
            else None
        )
        trend = (
            current_distance - previous_distance
            if current_distance is not None and previous_distance is not None
            else None
        )
        toward = trend is not None and trend < -0.01
        away = trend is not None and trend > 0.01
        trend_confidence = (
            min(1.0, len(distances) / max(2, self.matching.minimum_track_points))
            if current_position_complete and enough_track_points
            else 0.0
        )
        previously_airborne = any(
            not point.aircraft.on_ground
            and isinstance(point.aircraft.altitude, (int, float))
            and point.aircraft.altitude >= self.matching.airborne_altitude_threshold_ft
            for point in points[:-1]
        )
        recently_landed = aircraft.on_ground and previously_airborne
        landing_confidence = 0.9 if recently_landed and len(points) >= 2 else 0.0
        stale = age > self.tracking.stale_after_seconds
        inside_airport = _inside(aircraft, self.geofences.kmry_airport)
        near_jet = _inside(aircraft, jet)
        was_stationary_at_fbo = any(
            any(_inside(point.aircraft, geofence) for geofence in fbo_geofences.values())
            and (
                point.aircraft.ground_speed is None
                or point.aircraft.ground_speed <= self.matching.stopped_speed_max_knots
            )
            for point in points[:-1]
        )
        speed = aircraft.ground_speed
        taxiing = bool(
            speed is not None
            and self.matching.taxi_speed_min_knots <= speed <= self.matching.taxi_speed_max_knots
        )
        airborne = (
            not aircraft.on_ground
            and isinstance(aircraft.altitude, (int, float))
            and aircraft.altitude > self.matching.airborne_altitude_threshold_ft
        )
        if stale:
            state = MovementState.STALE
        elif near_jet and (speed is None or speed <= self.matching.stopped_speed_max_knots):
            state = MovementState.PARKED_AT_JET_CENTER
        elif recently_landed:
            state = MovementState.RECENTLY_LANDED
        elif toward:
            state = MovementState.MOVING_TOWARD_JET_CENTER
        elif away:
            state = MovementState.MOVING_AWAY_FROM_JET_CENTER
        elif taxiing and inside_airport:
            state = MovementState.TAXIING
        elif (
            inside_airport and speed is not None and speed <= self.matching.stopped_speed_max_knots
        ):
            state = MovementState.STOPPED_ON_AIRPORT
        elif airborne and (aircraft.vertical_rate or 0) > 100:
            state = MovementState.DEPARTING
        elif airborne:
            state = MovementState.AIRBORNE_INBOUND
        else:
            state = MovementState.UNKNOWN
        score = 0.0
        breakdown: list[str] = []

        def add(label: str, applies: bool, weight: float) -> None:
            nonlocal score
            if applies:
                score += weight
                breakdown.append(f"{label}: {weight:+g}")

        add("inside_airport", inside_airport, self.scoring.inside_airport)
        add("on_ground", aircraft.on_ground, self.scoring.on_ground)
        add("recently_landed", recently_landed, self.scoring.recently_landed)
        add("taxi_speed", taxiing, self.scoring.taxi_speed)
        add("moving_toward_jet_center", toward, self.scoring.moving_toward_jet_center)
        add("near_jet_center", near_jet, self.scoring.near_jet_center)
        add("recent", age <= self.decision.maximum_candidate_age_seconds, self.scoring.recent)
        add("stale", stale, self.scoring.stale)
        add(
            "airborne_climbing",
            airborne and (aircraft.vertical_rate or 0) > 100,
            self.scoring.airborne_climbing,
        )
        add("departing", state == MovementState.DEPARTING, self.scoring.departing)
        add("moving_away", away, self.scoring.moving_away)
        incomplete = aircraft.registration is None
        add("incomplete", incomplete, self.scoring.incomplete)
        registration = (aircraft.registration or "").replace("-", "").upper()
        flight = (aircraft.flight or "").replace(" ", "").upper()
        if spoken:
            exact = bool(
                spoken.full_registration and spoken.full_registration in {registration, flight}
            )
            suffix_exact = bool(
                spoken.suffix
                and (registration.endswith(spoken.suffix) or flight.endswith(spoken.suffix))
            )
            fuzzy = bool(
                spoken.normalized_form and ratio(spoken.normalized_form, registration) >= 85
            )
            flight_fuzzy = bool(
                spoken.normalized_form and ratio(spoken.normalized_form, flight) >= 85
            )
            add("spoken_registration_exact", exact, self.scoring.spoken_registration_exact)
            add(
                "spoken_registration_suffix",
                suffix_exact and not exact,
                self.scoring.spoken_registration_exact,
            )
            add(
                "spoken_registration_fuzzy",
                fuzzy and not exact and not suffix_exact,
                self.scoring.spoken_registration_fuzzy,
            )
            add(
                "spoken_callsign_fuzzy",
                flight_fuzzy and not exact and not suffix_exact,
                self.scoring.spoken_callsign_fuzzy,
            )
        return CandidateAnalysis(
            aircraft,
            state,
            score,
            breakdown,
            current_distance,
            previous_distance,
            trend,
            toward,
            trend_confidence,
            recently_landed,
            landing_confidence,
            age,
            bool(fbo_names),
            fbo_names,
            was_stationary_at_fbo,
        )

    def correlate(
        self,
        spoken: SpokenCallsign | None,
        aircraft: list[NearbyAircraft],
        now: datetime | None = None,
    ) -> AircraftMatch:
        now = now or datetime.now(UTC)
        contacts = self.tracker.update(aircraft, now)
        analyses = [
            self._analyze(contact, spoken, now) for contact in contacts if contact.observations
        ]
        registrations = [a.aircraft.registration for a in analyses if a.aircraft.registration]
        duplicates = {value for value in registrations if registrations.count(value) > 1}
        if duplicates:
            analyses = [
                CandidateAnalysis(
                    **{
                        **analysis.__dict__,
                        "score": analysis.score
                        + (
                            self.scoring.duplicate_registration
                            if analysis.aircraft.registration in duplicates
                            else 0
                        ),
                        "score_breakdown": [
                            *analysis.score_breakdown,
                            "duplicate_registration: conflict",
                        ],
                    }
                )
                for analysis in analyses
            ]
        analyses.sort(key=lambda item: item.score, reverse=True)
        candidate_lines = [
            f"{item.aircraft.registration or item.aircraft.hex} score={item.score:.1f} "
            f"state={item.state.value} age={item.age_seconds:.1f}s: "
            + "; ".join(item.score_breakdown)
            for item in analyses
        ]
        if not analyses:
            return AircraftMatch(
                confidence=0,
                status=MatchStatus.UNRESOLVED,
                adsb_decision=AdsbDecisionState.NO_CANDIDATE,
                match_reasons=["no ADS-B candidate"],
                candidate_scores=[],
            )
        winner = analyses[0]
        second = analyses[1].score if len(analyses) > 1 else 0.0
        margin = winner.score - second
        stale = winner.age_seconds > self.decision.maximum_candidate_age_seconds
        plausible_state = winner.aircraft.on_ground or winner.recently_landed
        speech_conflict = bool(
            spoken
            and spoken.full_registration
            and winner.aircraft.registration
            and spoken.full_registration != winner.aircraft.registration.replace("-", "").upper()
        )
        suffix_matches = [
            item
            for item in analyses
            if spoken
            and spoken.suffix
            and (
                (item.aircraft.registration or "").replace("-", "").upper().endswith(spoken.suffix)
                or (item.aircraft.flight or "").replace(" ", "").upper().endswith(spoken.suffix)
            )
        ]
        unique_suffix_match = len(suffix_matches) == 1 and suffix_matches[0] is winner
        speech_identity_supported = bool(
            spoken
            and (
                unique_suffix_match
                or (
                    spoken.full_registration
                    and spoken.full_registration
                    in {
                        (winner.aircraft.registration or "").replace("-", "").upper(),
                        (winner.aircraft.flight or "").replace(" ", "").upper(),
                    }
                )
            )
        )
        reasons = [
            f"winning score {winner.score:.1f}",
            f"second-best score {second:.1f}",
            f"winning margin {margin:.1f}",
            f"movement state {winner.state.value}",
        ]
        if unique_suffix_match:
            reasons.append("unique suffix matched nearby registration")
        if speech_conflict:
            decision = AdsbDecisionState.REJECTED
            reasons.append("speech_adsb_conflict")
        elif stale:
            decision = AdsbDecisionState.REJECTED
            reasons.append("stale_candidate_rejected")
        elif self.decision.require_on_ground_or_recently_landed and not plausible_state:
            decision = AdsbDecisionState.REJECTED
            reasons.append("not_on_ground_or_recently_landed")
        elif winner.score < self.decision.minimum_score:
            decision = AdsbDecisionState.INSUFFICIENT_DATA
            reasons.append("winning_score_below_minimum")
        elif not speech_identity_supported and winner.state in {
            MovementState.PARKED_AT_JET_CENTER,
            MovementState.STOPPED_ON_AIRPORT,
        }:
            decision = AdsbDecisionState.INSUFFICIENT_DATA
            reasons.append(
                "stationary ADS-B candidate was not assigned without matching callsign evidence"
            )
        elif margin < self.decision.minimum_margin and not unique_suffix_match:
            decision = AdsbDecisionState.AMBIGUOUS
            reasons.append("winning_margin_below_minimum")
        else:
            decision = AdsbDecisionState.CONFIRMED
            reasons.append(
                "unique spoken suffix identified the ADS-B aircraft"
                if unique_suffix_match
                else "one ADS-B candidate clearly won"
            )
        confirmed = decision == AdsbDecisionState.CONFIRMED
        return AircraftMatch(
            aircraft=winner.aircraft if confirmed else None,
            registration=winner.aircraft.registration if confirmed else None,
            confidence=0.92 if confirmed else 0,
            status=MatchStatus.CONFIRMED
            if confirmed
            else (
                MatchStatus.AMBIGUOUS
                if decision == AdsbDecisionState.AMBIGUOUS
                else MatchStatus.UNRESOLVED
            ),
            match_reasons=reasons,
            alternative_candidates=[item.aircraft for item in analyses[:3]]
            if not confirmed
            else [],
            identification_source=IdentificationSource.ADSB_CORRELATION
            if confirmed
            else IdentificationSource.UNRESOLVED,
            candidate_scores=candidate_lines,
            adsb_snapshot_timestamp=now,
            adsb_decision=decision,
            winning_score=winner.score,
            second_best_score=second,
            winning_margin=margin,
            movement_state=winner.state.value,
            inside_fbo_geofence=winner.inside_fbo_geofence,
            fbo_geofence_names=winner.fbo_geofence_names,
            was_stationary_at_fbo=winner.was_stationary_at_fbo,
            moving_away_from_destination=winner.state == MovementState.MOVING_AWAY_FROM_JET_CENTER,
            recently_landed=winner.recently_landed,
        )
