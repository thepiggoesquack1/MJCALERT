from pathlib import Path

from mry_alert.config import AppConfig


def test_authorized_liveatc_scope_matches_sample_configuration() -> None:
    defaults = AppConfig()
    sample = AppConfig.load(Path("config.example.yaml"))

    assert defaults.liveatc.enabled
    assert sample.liveatc.enabled
    assert sample.liveatc.authorized_player_url == (
        "https://www.liveatc.net/hlisten.php?mount=kmry&icao=kmry"
    )
    assert sample.liveatc.authorized_mount == "kmry"
    assert sample.liveatc.authorized_icao == "kmry"
    assert sample.liveatc.audio_websocket_path == "/ws/audio"


def test_example_config_contains_recall_and_traffic_filter_schema() -> None:
    sample = AppConfig.load(Path("config.example.yaml"))

    assert sample.traffic_filter.enabled
    assert sample.traffic_filter.ignore_scheduled_airlines
    assert sample.traffic_filter.allowed_icao_designators == ["JSX"]
    assert sample.intent_detection.destination_phrase_threshold == 0.82
    assert sample.intent_detection.route_context_threshold == 0.72
    assert sample.intent_detection.allow_partial_jet_center_match
