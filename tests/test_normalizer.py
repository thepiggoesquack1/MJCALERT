from mry_alert.detection.normalizer import normalize_transcript


def test_normalizes_recognition_variants_without_changing_original() -> None:
    original = "Monterrey Jett Center, X-ray nine er, ALFA!"
    assert normalize_transcript(original) == "monterey jet center xray niner alpha"
    assert original.startswith("Monterrey")


def test_number_becomes_november_only_before_spoken_digits() -> None:
    assert normalize_transcript("number one two three alpha bravo").startswith("november one")
    assert normalize_transcript("number of aircraft") == "number of aircraft"
