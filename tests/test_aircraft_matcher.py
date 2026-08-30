from mry_alert.config import AdsbConfig, DetectionConfig
from mry_alert.detection.callsign import parse_callsign
from mry_alert.detection.matcher import AircraftMatcher
from mry_alert.models import MatchStatus, NearbyAircraft


def aircraft(registration: str, speed: float = 5, seen: float = 2) -> NearbyAircraft:
    return NearbyAircraft(
        hex=registration,
        registration=registration,
        on_ground=True,
        ground_speed=speed,
        seconds_since_seen=seen,
    )


def test_full_registration_is_confirmed() -> None:
    spoken = parse_callsign("november one two three alpha bravo")
    assert spoken
    result = AircraftMatcher(DetectionConfig(), AdsbConfig()).match(spoken, [aircraft("N123AB")])
    assert result.status == MatchStatus.CONFIRMED
    assert result.confidence >= 0.8
    assert "full registration matched ADS-B registration" in result.match_reasons


def test_unique_suffix_is_likely() -> None:
    spoken = parse_callsign("citation three alpha bravo")
    assert spoken
    result = AircraftMatcher(DetectionConfig(), AdsbConfig()).match(
        spoken, [aircraft("N123AB"), aircraft("N555ZZ")]
    )
    assert result.status == MatchStatus.LIKELY
    assert result.aircraft and result.aircraft.registration == "N123AB"
    assert "unique suffix matched nearby registration" in result.match_reasons


def test_equal_suffix_matches_are_ambiguous() -> None:
    spoken = parse_callsign("citation three alpha bravo")
    assert spoken
    result = AircraftMatcher(DetectionConfig(), AdsbConfig()).match(
        spoken, [aircraft("N123AB"), aircraft("N923AB")]
    )
    assert result.status == MatchStatus.AMBIGUOUS
    assert result.aircraft is None
    assert len(result.alternative_candidates) == 2


def test_proximity_without_identity_is_unresolved() -> None:
    spoken = parse_callsign("citation three alpha bravo")
    assert spoken
    result = AircraftMatcher(DetectionConfig(), AdsbConfig()).match(spoken, [aircraft("N888ZZ")])
    assert result.status == MatchStatus.UNRESOLVED


def test_full_spoken_registration_survives_adsb_unavailability() -> None:
    spoken = parse_callsign("november eight two five sierra papa")
    assert spoken
    result = AircraftMatcher(DetectionConfig(), AdsbConfig()).match(spoken, [])
    assert result.registration == "N825SP"
    assert result.status == MatchStatus.LIKELY
    assert result.identification_source.value == "spoken_full_registration"


def test_fuzzy_damaged_callsign_is_likely_but_not_confirmed() -> None:
    result = AircraftMatcher(DetectionConfig(), AdsbConfig()).match(
        None,
        [aircraft("N825SP")],
        "eight two five cereal papa",
    )
    assert result.registration == "N825SP"
    assert result.status == MatchStatus.LIKELY
    assert result.identification_source.value == "fuzzy_adsb_recovery"


def test_unsafe_short_fragment_does_not_identify_aircraft() -> None:
    result = AircraftMatcher(DetectionConfig(), AdsbConfig()).match(
        None, [aircraft("N825SP")], "papa"
    )
    assert result.status == MatchStatus.UNRESOLVED
    assert result.registration is None


def test_no_callsign_unique_ground_candidate_is_likely_only() -> None:
    result = AircraftMatcher(DetectionConfig(), AdsbConfig()).match(
        None, [aircraft("N825SP")], "alpha echo to the jet center"
    )
    assert result.status == MatchStatus.LIKELY
    assert result.identification_source.value == "unique_ground_candidate"


def test_no_callsign_multiple_ground_candidates_are_ambiguous() -> None:
    result = AircraftMatcher(DetectionConfig(), AdsbConfig()).match(
        None,
        [aircraft("N825SP"), aircraft("N441QS")],
        "alpha echo to the jet center",
    )
    assert result.status == MatchStatus.AMBIGUOUS
    assert result.registration is None
