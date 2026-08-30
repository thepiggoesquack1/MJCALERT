from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import time
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import uvicorn

from mry_alert.adsb.factory import create_nearby_aircraft_provider
from mry_alert.adsb.mock import MockNearbyAircraftProvider
from mry_alert.audio.devices import list_audio_devices
from mry_alert.audio.file_source import WaveFileAudioSource
from mry_alert.audio.microphone_source import MicrophoneAudioSource
from mry_alert.audio.segmenter import VadSegmenter
from mry_alert.config import AppConfig
from mry_alert.detection.engine import DetectionEngine
from mry_alert.logging_config import configure_logging
from mry_alert.models import AlertEvent, NearbyAircraft, TranscriptEvent
from mry_alert.transcription.benchmark import benchmark_model, load_segmented_wave
from mry_alert.transcription.benchmark_process import run_benchmark_subprocess
from mry_alert.transcription.evaluation import load_expected_json, load_expected_transcript
from mry_alert.transcription.faster_whisper import FasterWhisperTranscriber
from mry_alert.transcription.runtime import diagnose_speech_runtime

logger = logging.getLogger(__name__)


def _config(path: str | None) -> AppConfig:
    return AppConfig.load(Path(path)) if path else AppConfig()


async def simulate(path: Path, config: AppConfig) -> int:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    aircraft = [NearbyAircraft.model_validate(item) for item in data.get("aircraft", [])]
    alerts: list[AlertEvent] = []

    async def publish(event: object) -> None:
        if isinstance(event, AlertEvent):
            alerts.append(event)
            print(event.model_dump_json(indent=2))

    engine = DetectionEngine(config, MockNearbyAircraftProvider(aircraft), publisher=publish)
    base = datetime.now(UTC)
    for index, item in enumerate(data.get("transcripts", [])):
        timestamp = base + timedelta(seconds=float(item.get("offset_seconds", index)))
        event = TranscriptEvent(
            event_id=str(uuid4()),
            timestamp=timestamp,
            text=str(item["text"]),
            source="simulation",
        )
        await engine.process(event)
    await engine.flush_pending()
    if not alerts:
        print("No alert produced.")
    return 0


async def _transcribe_source(
    source: WaveFileAudioSource | MicrophoneAudioSource, config: AppConfig
) -> None:
    transcriber = FasterWhisperTranscriber(config.speech, config.audio.preprocessing)
    segmenter = VadSegmenter(config.audio)
    provider = create_nearby_aircraft_provider(config.airport, config.adsb)

    async def publish(event: object) -> None:
        if isinstance(event, AlertEvent):
            print(event.model_dump_json(indent=2))

    engine = DetectionEngine(config, provider, publisher=publish)

    async def process_transmission(pcm: bytes) -> None:
        if config.audio.save_debug_audio:
            directory = config.audio.debug_audio_directory
            directory.mkdir(parents=True, exist_ok=True)
            debug_path = directory / f"transmission-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4()}.wav"
            with wave.open(str(debug_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(config.audio.sample_rate)
                output.writeframes(pcm)
            logger.info("Saved configured debug audio to %s", debug_path)
        transcription = transcriber.transcribe(pcm, config.audio.sample_rate)
        text = transcription.text
        confidence = transcription.language_probability
        print(text)
        event = TranscriptEvent(
            event_id=str(uuid4()),
            timestamp=datetime.now(UTC),
            text=text,
            source=source.__class__.__name__,
            duration_seconds=len(pcm) / 2 / config.audio.sample_rate,
            transcription_confidence=confidence,
        )
        await engine.process(event)

    async for frame in source.frames():
        pcm = segmenter.add_frame(frame)
        if pcm:
            await process_transmission(pcm)
    if final_pcm := segmenter.flush():
        await process_transmission(final_pcm)
    await engine.flush_pending()


def _load_adsb_fixture(path: Path | None) -> list[NearbyAircraft]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = datetime.now(UTC)
    observations: list[NearbyAircraft] = []
    for item in payload.get("observations", []):
        values = dict(item)
        offset = float(values.pop("timestamp_offset_seconds", 0))
        if "altitude_ft" in values:
            values["altitude"] = values.pop("altitude_ft")
        if "ground_speed_knots" in values:
            values["ground_speed"] = values.pop("ground_speed_knots")
        values["source_timestamp"] = base + timedelta(seconds=offset)
        observations.append(NearbyAircraft.model_validate(values))
    return observations


async def replay(
    path: Path, config: AppConfig, model: str | None, adsb_fixture: Path | None = None
) -> int:
    speech = config.speech_for_model(model or config.speech.model)
    transcriber = FasterWhisperTranscriber(speech, config.audio.preprocessing)
    source = WaveFileAudioSource(path, config.audio.frame_duration_ms)
    segmenter = VadSegmenter(config.audio)
    results: list[dict[str, object]] = []
    fixture_aircraft = _load_adsb_fixture(adsb_fixture)
    detector_events: list[object] = []

    async def publish(event: object) -> None:
        detector_events.append(event)

    engine = DetectionEngine(
        config, MockNearbyAircraftProvider(fixture_aircraft), publisher=publish
    )
    audio_elapsed = 0.0

    async def process(pcm: bytes, end_seconds: float) -> None:
        duration = len(pcm) / 2 / config.audio.sample_rate
        started = time.perf_counter()
        result = transcriber.transcribe(pcm, config.audio.sample_rate)
        elapsed = time.perf_counter() - started
        event = TranscriptEvent(
            event_id=str(uuid4()),
            timestamp=datetime.now(UTC),
            text=result.text,
            source=config.liveatc.source_label,
            duration_seconds=duration,
        )
        detector_result = await engine.process(event)
        results.append(
            {
                "start_seconds": round(max(0.0, end_seconds - duration), 3),
                "end_seconds": round(end_seconds, 3),
                "processing_seconds": round(elapsed, 3),
                "real_time_factor": round(elapsed / duration, 3) if duration else None,
                "transcript": result.text,
                "decoder_confidence": result.quality,
                "average_log_probability": result.average_log_probability,
                "no_speech_probability": result.no_speech_probability,
                "normalized_transcript": event.normalized_text,
                "detector_decision": event.detection_decision.value,
                "adsb_identification": event.identified_registration,
                "notification_would_be_sent": detector_result is not None,
                "notification_sent": False,
            }
        )

    async for frame in source.frames():
        audio_elapsed += len(frame) / 2 / config.audio.sample_rate
        if transmission := segmenter.add_frame(frame):
            await process(transmission, audio_elapsed)
    if transmission := segmenter.flush():
        await process(transmission, audio_elapsed)
    await engine.flush_pending()
    await engine.close()
    print(
        json.dumps(
            {
                "model": speech.model,
                "audio": str(path),
                "audio_duration_seconds": round(audio_elapsed, 3),
                "prompted_nearby_registrations": [],
                "segments": results,
                "adsb_fixture": str(adsb_fixture) if adsb_fixture else None,
                "notifications_sent": 0,
            },
            indent=2,
        )
    )
    return 0


def _comparison_table(reports: list[dict[str, Any]]) -> str:
    header = (
        f"{'Model':<58} {'RTF':>7} {'Class':<24} {'Callsign':<12} "
        f"{'Destination':<22} {'Hallucinations':>14}"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        if report.get("status") != "success":
            status = str(report.get("status", "failed")).upper()
            lines.append(f"{report['model']:<58} {status:>11} {str(report.get('error', ''))[:60]}")
            continue
        segments = report["segments"]
        callsign = next(
            (item["callsign_candidates"][0] for item in segments if item["callsign_candidates"]),
            "-",
        )
        destination = next(
            (item["detected_destination"] for item in segments if item["detected_destination"]),
            "-",
        )
        hallucinations = sum(item["hallucination_filter"]["trimmed"] for item in segments)
        lines.append(
            f"{report['model']:<58} {report['real_time_factor']:>7.2f} "
            f"{report['performance_classification']:<24} {callsign:<12} "
            f"{destination:<22} {hallucinations:>14}"
        )
    return "\n".join(lines)


async def compare_models(
    path: Path,
    models: list[str],
    config: AppConfig,
    expected_path: Path | None,
    send_notifications: bool,
) -> int:
    logger.info("Benchmark timing: loading and segmenting audio %s", path)
    loading_started = time.perf_counter()
    audio_duration, segments = await load_segmented_wave(path, config)
    logger.info(
        "Benchmark timing: audio loading produced %d segments in %.3f seconds",
        len(segments),
        time.perf_counter() - loading_started,
    )
    expected = load_expected_transcript(expected_path)
    reports: list[dict[str, Any]] = []
    for model in models:
        request = {
            "model": model,
            "segments": [base64.b64encode(segment).decode("ascii") for segment in segments],
            "audio_duration": audio_duration,
            "config": config.model_dump(mode="json"),
            "expected_transcript": expected,
            "send_notifications": send_notifications,
        }
        report = run_benchmark_subprocess(request, config.benchmark.model_timeout_seconds)
        reports.append(report)
        if report.get("status") == "interrupted":
            print(json.dumps({"audio": str(path), "models": reports}, indent=2))
            return 130
    print(json.dumps({"audio": str(path), "models": reports}, indent=2))
    print("\nMODEL COMPARISON")
    print(_comparison_table(reports))
    if send_notifications:
        logger.warning(
            "Comparison evaluated notification decisions, but no Chrome client is connected "
            "to this offline command; zero notifications were delivered"
        )
    return 0


async def benchmark_worker(request_path: Path, output_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    config = AppConfig.model_validate(request["config"])
    segments = [base64.b64decode(value) for value in request["segments"]]
    try:
        report = await benchmark_model(
            str(request["model"]),
            segments,
            float(request["audio_duration"]),
            config,
            request.get("expected_transcript"),
            send_notifications=bool(request.get("send_notifications", False)),
        )
    except (RuntimeError, OSError, ValueError) as exc:
        report = {
            "model": str(request["model"]),
            "status": "failed",
            "error": str(exc),
            "segments": [],
        }
    output_path.write_text(json.dumps(report), encoding="utf-8")
    return 0


def _dataset_cases(dataset: Path) -> list[tuple[Path, Path | None, Path | None]]:
    cases: list[tuple[Path, Path | None, Path | None]] = []
    for audio in sorted(dataset.glob("*/audio.wav")):
        transcript = audio.with_name("expected.txt")
        metadata = audio.with_name("expected.json")
        cases.append(
            (
                audio,
                transcript if transcript.exists() else None,
                metadata if metadata.exists() else None,
            )
        )
    if not cases:
        raise ValueError(f"No */audio.wav evaluation cases found under {dataset}")
    return cases


async def evaluate_models(dataset: Path, models: list[str], config: AppConfig) -> int:
    cases = _dataset_cases(dataset)
    aggregate: list[dict[str, Any]] = []
    for model in models:
        reports: list[dict[str, Any]] = []
        true_positive = false_positive = false_negative = 0
        wrong_aircraft_alerts = ambiguous_cases = no_candidate_cases = 0
        winning_margins: list[float] = []
        entity_keys = (
            "callsign_accuracy",
            "operator_callsign_accuracy",
            "destination_accuracy",
            "runway_accuracy",
            "taxiway_accuracy",
            "intent_accuracy",
            "correction_accuracy",
        )
        entity_totals = dict.fromkeys(entity_keys, 0.0)
        entity_counts = dict.fromkeys(entity_keys, 0)
        for audio, transcript_path, metadata_path in cases:
            duration, segments = await load_segmented_wave(audio, config)
            expected_text = load_expected_transcript(transcript_path)
            try:
                report = await benchmark_model(model, segments, duration, config, expected_text)
            except (RuntimeError, OSError, ValueError) as exc:
                report = {
                    "model": model,
                    "error": str(exc),
                    "segments": [],
                    "real_time_factor": 0.0,
                }
            report["case"] = audio.parent.name
            metadata = load_expected_json(metadata_path) if metadata_path else {}
            predicted_notify = any(
                item["notification_would_be_sent"] for item in report["segments"]
            )
            expected_notify = bool(metadata.get("should_notify", False))
            true_positive += int(predicted_notify and expected_notify)
            false_positive += int(predicted_notify and not expected_notify)
            false_negative += int(not predicted_notify and expected_notify)
            decisions = [item.get("adsb_decision") for item in report["segments"]]
            ambiguous_cases += int("ambiguous" in decisions)
            no_candidate_cases += int("no_candidate" in decisions)
            winning_margins.extend(
                float(item["adsb_winning_margin"])
                for item in report["segments"]
                if item.get("adsb_winning_margin") is not None
            )
            predicted_registration = next(
                (
                    item.get("adsb_identification")
                    for item in report["segments"]
                    if item.get("adsb_identification")
                ),
                None,
            )
            wrong_aircraft_alerts += int(
                predicted_notify
                and bool(metadata.get("registration"))
                and predicted_registration != metadata.get("registration")
            )
            json_metrics = _detector_expectation_metrics(report["segments"], metadata)
            report["expected_json_metrics"] = json_metrics
            for key, value in json_metrics.items():
                entity_totals[key] += value
                entity_counts[key] += 1
            reports.append(report)
        successful = [report for report in reports if not report.get("error")]
        aggregate.append(
            {
                "model": model,
                "cases": reports,
                "average_word_error_rate": _mean_metric(reports, "word_error_rate"),
                "callsign_accuracy": entity_totals["callsign_accuracy"]
                / max(1, entity_counts["callsign_accuracy"]),
                "operator_callsign_accuracy": entity_totals["operator_callsign_accuracy"]
                / max(1, entity_counts["operator_callsign_accuracy"]),
                "destination_accuracy": entity_totals["destination_accuracy"]
                / max(1, entity_counts["destination_accuracy"]),
                "runway_accuracy": entity_totals["runway_accuracy"]
                / max(1, entity_counts["runway_accuracy"]),
                "taxiway_accuracy": entity_totals["taxiway_accuracy"]
                / max(1, entity_counts["taxiway_accuracy"]),
                "intent_accuracy": entity_totals["intent_accuracy"]
                / max(1, entity_counts["intent_accuracy"]),
                "correction_accuracy": entity_totals["correction_accuracy"]
                / max(1, entity_counts["correction_accuracy"]),
                "alert_precision": true_positive / max(1, true_positive + false_positive),
                "alert_recall": true_positive / max(1, true_positive + false_negative),
                "wrong_aircraft_confirmed_alerts": wrong_aircraft_alerts,
                "false_destination_alerts": false_positive,
                "missed_jet_center_arrivals": false_negative,
                "ambiguous_case_rate": ambiguous_cases / len(reports),
                "no_candidate_rate": no_candidate_cases / len(reports),
                "average_decision_latency_seconds": None,
                "average_winning_margin": sum(winning_margins) / len(winning_margins)
                if winning_margins
                else None,
                "average_real_time_factor": (
                    sum(float(r["real_time_factor"]) for r in successful) / len(successful)
                    if successful
                    else None
                ),
                "hallucination_rejection_count": sum(
                    int(item["hallucination_filter"]["trimmed"])
                    for report in reports
                    for item in report["segments"]
                ),
            }
        )
    print(json.dumps({"dataset": str(dataset), "models": aggregate}, indent=2))
    return 0


def _detector_expectation_metrics(
    segments: list[dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, float]:
    entity_rows = [item["aviation_entities"] for item in segments]
    observed: dict[str, object | None] = {
        "registration": next(
            (item["callsign_candidates"][0] for item in segments if item["callsign_candidates"]),
            None,
        ),
        "destination": next(
            (item["detected_destination"] for item in segments if item["detected_destination"]),
            None,
        ),
        "intent": next(
            (item["detected_intent"] for item in segments if item["detected_intent"] != "none"),
            None,
        ),
        "correction": any(item["correction_handling"] is not None for item in segments),
        "operator_callsigns": next(
            (row["operator_callsigns"] for row in entity_rows if row["operator_callsigns"]), []
        ),
        "runways": next((row["runways"] for row in entity_rows if row["runways"]), []),
        "taxiways": next((row["taxiways"] for row in entity_rows if row["taxiways"]), []),
    }
    mapping = {
        "registration": "callsign_accuracy",
        "operator_callsigns": "operator_callsign_accuracy",
        "destination": "destination_accuracy",
        "runways": "runway_accuracy",
        "taxiways": "taxiway_accuracy",
        "intent": "intent_accuracy",
        "correction": "correction_accuracy",
    }
    return {
        score_name: float(observed[field] == metadata[field])
        for field, score_name in mapping.items()
        if field in metadata
    }


def _mean_metric(reports: list[dict[str, Any]], name: str) -> float | None:
    values = [
        float(metrics[name])
        for report in reports
        if isinstance((metrics := report.get("expected_transcript_metrics")), dict)
        and name in metrics
    ]
    return sum(values) / len(values) if values else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mry-alert")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--config")
    simulation = sub.add_parser("simulate")
    simulation.add_argument("--fixture", required=True)
    simulation.add_argument("--config")
    transcribe = sub.add_parser("transcribe-file")
    transcribe.add_argument("path")
    transcribe.add_argument("--config")
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--audio", required=True)
    replay_parser.add_argument("--model")
    replay_parser.add_argument("--config")
    replay_parser.add_argument("--adsb-fixture")
    check = sub.add_parser("check-speech-runtime")
    check.add_argument("--config")
    compare = sub.add_parser("compare-models")
    compare.add_argument("--audio", required=True)
    compare.add_argument("--models", nargs="+", required=True)
    compare.add_argument("--expected-transcript")
    compare.add_argument("--send-notifications", action="store_true")
    compare.add_argument("--config")
    evaluate = sub.add_parser("evaluate-models")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--models", nargs="+", required=True)
    evaluate.add_argument("--config")
    worker = sub.add_parser("_benchmark-worker", help=argparse.SUPPRESS)
    worker.add_argument("--request", required=True)
    worker.add_argument("--output", required=True)
    sub.add_parser("list-audio-devices")
    microphone = sub.add_parser("monitor-microphone")
    microphone.add_argument("--device")
    microphone.add_argument("--config")
    train_classifier = sub.add_parser("train-audio-classifier")
    train_classifier.add_argument("--dataset", required=True)
    train_classifier.add_argument("--output", required=True)
    train_classifier.add_argument("--config")
    evaluate_classifier = sub.add_parser("evaluate-audio-classifier")
    evaluate_classifier.add_argument("--dataset", required=True)
    evaluate_classifier.add_argument("--model", required=True)
    evaluate_classifier.add_argument("--config")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    try:
        if args.command == "_benchmark-worker":
            return asyncio.run(benchmark_worker(Path(args.request), Path(args.output)))
        if args.command == "simulate":
            return asyncio.run(simulate(Path(args.fixture), _config(args.config)))
        if args.command == "serve":
            config = _config(args.config)
            if config.server.host not in {"127.0.0.1", "localhost", "::1"}:
                logger.warning("Server is configured beyond localhost; ensure this is deliberate")
            from mry_alert.server.app import create_app

            print(f"Pairing token file: {config.server.pairing_token_file}")
            uvicorn.run(create_app(config), host=config.server.host, port=config.server.port)
            return 0
        if args.command == "list-audio-devices":
            print(list_audio_devices())
            return 0
        config = _config(args.config)
        if args.command == "train-audio-classifier":
            from mry_alert.audio_classifier.training import train_classifier

            report = train_classifier(
                Path(args.dataset),
                Path(args.output),
                config.audio_augmentation,
            )
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "evaluate-audio-classifier":
            from mry_alert.audio_classifier.training import evaluate_classifier

            report = evaluate_classifier(Path(args.dataset), Path(args.model))
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "check-speech-runtime":
            diagnostic = diagnose_speech_runtime(config)
            print(json.dumps(diagnostic.to_dict(), indent=2))
            return 0 if diagnostic.initialized else 1
        if args.command == "compare-models":
            return asyncio.run(
                compare_models(
                    Path(args.audio),
                    args.models,
                    config,
                    Path(args.expected_transcript) if args.expected_transcript else None,
                    args.send_notifications,
                )
            )
        if args.command == "evaluate-models":
            return asyncio.run(evaluate_models(Path(args.dataset), args.models, config))
        if args.command == "transcribe-file":
            asyncio.run(
                _transcribe_source(
                    WaveFileAudioSource(Path(args.path), config.audio.frame_duration_ms), config
                )
            )
        elif args.command == "replay":
            return asyncio.run(
                replay(
                    Path(args.audio),
                    config,
                    args.model,
                    Path(args.adsb_fixture) if args.adsb_fixture else None,
                )
            )
        elif args.command == "monitor-microphone":
            device: int | str | None = (
                int(args.device) if args.device and args.device.isdigit() else args.device
            )
            asyncio.run(
                _transcribe_source(
                    MicrophoneAudioSource(
                        device, config.audio.sample_rate, config.audio.frame_duration_ms
                    ),
                    config,
                )
            )
        return 0
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, ValueError, OSError) as exc:
        parser = build_parser()
        parser.error(str(exc))
        return 2
