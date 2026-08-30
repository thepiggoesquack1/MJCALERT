from mry_alert.detection.callsign import parse_callsign
from mry_alert.detection.speaker_role import infer_speaker_role
from mry_alert.models import SpeakerRole


def infer(text: str, *, prompted: bool = False):
    return infer_speaker_role(
        text,
        parse_callsign(text),
        responding_to_destination_prompt=prompted,
    )


def test_controller_command_structure_is_inferred_not_identified() -> None:
    result = infer("november one two three alpha bravo taxi to monterey jet center")
    assert result.role == SpeakerRole.CONTROLLER
    assert 0.8 <= result.confidence < 1.0
    assert "taxi command without request language" in result.reasons
    assert "transmission begins by addressing a callsign" in result.reasons


def test_direct_pilot_request_is_inferred_from_request_language() -> None:
    result = infer(
        "monterey ground november one two three alpha bravo request taxi to monterey jet center"
    )
    assert result.role == SpeakerRole.PILOT
    assert result.confidence >= 0.8
    assert "first-person taxi request" in result.reasons


def test_wed_like_to_taxi_is_inferred_as_pilot_language() -> None:
    result = infer(
        "monterey ground four six five charlie golf at del monte "
        "we d like to taxi over to monterey jet center"
    )
    assert result.role == SpeakerRole.PILOT
    assert "first-person destination statement" in result.reasons


def test_youre_going_to_with_continue_instruction_is_controller_language() -> None:
    result = infer(
        "continue down the runway you re going to monterey jet center "
        "i ll call your exit"
    )
    assert result.role == SpeakerRole.CONTROLLER
    assert "continue command" in result.reasons


def test_prompted_short_destination_response_is_likely_pilot() -> None:
    result = infer("monterey jet center", prompted=True)
    assert result.role == SpeakerRole.PILOT
    assert result.confidence >= 0.8


def test_ambiguous_short_transmission_stays_unknown() -> None:
    result = infer("monterey jet center")
    assert result.role == SpeakerRole.UNKNOWN
    assert result.confidence == 0.0


def test_short_callsign_only_transmission_is_a_likely_readback() -> None:
    result = infer("november one two three alpha bravo")
    assert result.role == SpeakerRole.PILOT
    assert "short callsign-only readback" in result.reasons


def test_callsign_readback_at_end_supports_pilot_role() -> None:
    result = infer("holding short november one two three alpha bravo")
    assert result.role == SpeakerRole.PILOT
    assert "callsign readback appears at the end" in result.reasons


def test_multiple_addressed_aircraft_support_controller_role() -> None:
    result = infer(
        "november one two three alpha bravo hold short "
        "november four five six delta echo taxi runway one zero"
    )
    assert result.role == SpeakerRole.CONTROLLER
    assert "multiple aircraft callsigns addressed in sequence" in result.reasons


def test_callsign_near_beginning_supports_controller_address_structure() -> None:
    result = infer("and november one two three alpha bravo taxi to monterey jet center")
    assert result.role == SpeakerRole.CONTROLLER
    assert result.confidence >= 0.8
    assert "clearly addresses a callsign near the beginning" in " ".join(result.reasons)
