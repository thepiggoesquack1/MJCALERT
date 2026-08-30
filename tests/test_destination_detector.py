import pytest

from mry_alert.config import AppConfig, DestinationConfig, IntentDetectionConfig
from mry_alert.detection.destination import detect_correction, detect_destination, parse_destination


def test_exact_destination_with_taxi_is_strong() -> None:
    result = detect_destination("request taxi to monterey jet center", DestinationConfig())
    assert result.detected
    assert result.confidence >= 0.9
    assert len(result.reasons) == 2


def test_bare_destination_is_not_an_event_without_context() -> None:
    result = detect_destination("monterey jet center", DestinationConfig())
    assert not result.detected


def test_parking_response_is_strong() -> None:
    result = detect_destination(
        "monterey jet center", DestinationConfig(), responding_to_parking=True
    )
    assert result.detected and result.confidence >= 0.9


def test_latest_explicit_known_destination_wins() -> None:
    result = parse_destination(
        "taxi to monterey jet center, correction del monte aviation", AppConfig().destinations
    )
    assert result and result.canonical_name == "Del Monte Aviation"


def test_correction_requires_marker_or_redirect_context() -> None:
    destination = parse_destination("make that del monte", AppConfig().destinations)
    result = detect_correction("make that del monte", destination, "Monterey Jet Center")
    assert result.detected and result.corrected_destination == "Del Monte Aviation"


def test_unrelated_bare_no_is_not_a_correction() -> None:
    result = detect_correction("no reported traffic", None, "Monterey Jet Center")
    assert not result.detected


def test_unambiguous_disregard_cancels_without_new_destination() -> None:
    result = detect_correction("disregard that request", None, "Monterey Jet Center")
    assert result.detected and result.corrected_destination is None


@pytest.mark.parametrize(
    "text",
    [
        "request taxi to monterey jet center",
        "taxi to the jet center",
        "parking monterey jet",
        "going to the jet center",
        "headed for monterey jet",
        "request jet center",
    ],
)
def test_recall_oriented_jet_center_requests_are_accepted(text: str) -> None:
    result = detect_destination(
        text,
        DestinationConfig(),
        authorized_ground_source=True,
        intent_config=IntentDetectionConfig(),
    )
    assert result.detected
    assert result.confidence >= 0.82


@pytest.mark.parametrize(
    "text",
    [
        "jet center via alpha echo",
        "via alpha echo to the jet center",
        "monitor ground taxi via alpha echo to jet center",
        "turn left at foxtrot then proceed to the jet center",
    ],
)
def test_partial_phrase_with_strong_route_context_is_accepted(text: str) -> None:
    result = detect_destination(
        text,
        DestinationConfig(),
        authorized_ground_source=True,
        intent_config=IntentDetectionConfig(),
    )
    assert result.detected
    assert result.intent_category.value in {
        "ground_route_to_destination",
        "explicit_taxi_request",
    }
    assert result.route_cues


def test_weak_jet_center_phrase_without_route_remains_unresolved_evidence() -> None:
    result = detect_destination(
        "jet center",
        DestinationConfig(),
        authorized_ground_source=True,
        intent_config=IntentDetectionConfig(),
    )
    assert not result.detected
    assert result.exact_destination
    assert result.intent_category.value == "weak_destination_mention"


def test_adsb_context_is_not_part_of_destination_detection() -> None:
    result = detect_destination(
        "november one two three alpha bravo holding short",
        DestinationConfig(),
        authorized_ground_source=True,
        intent_config=IntentDetectionConfig(),
    )
    assert not result.detected
    assert result.confidence == 0
