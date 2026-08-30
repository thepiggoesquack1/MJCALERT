import pytest

from mry_alert.detection.callsign import parse_callsign


@pytest.mark.parametrize(
    "text",
    [
        "november one two three alpha bravo request taxi",
        "n one two three alpha bravo taxi",
    ],
)
def test_full_n_number(text: str) -> None:
    parsed = parse_callsign(text)
    assert parsed is not None
    assert parsed.full_registration == "N123AB"
    assert parsed.parse_confidence >= 0.9


def test_type_prefixed_and_bare_suffix() -> None:
    typed = parse_callsign("citation three alpha bravo say parking")
    bare = parse_callsign("three alpha bravo")
    assert typed and typed.suffix == "3AB" and typed.aircraft_type_prefix == "Citation"
    assert bare and bare.suffix == "3AB"
    assert typed.full_registration is None


def test_does_not_parse_random_number_phrase() -> None:
    assert parse_callsign("taxi runway two eight") is None
