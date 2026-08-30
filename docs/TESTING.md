# Testing

Install the development dependencies and run:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy src
```

Validate the extension package and notification callback behavior with:

```bash
node --check extension/background.js
node --check extension/popup.js
node --check extension/offscreen.js
node --check extension/audio-processor.js
node tests/extension_notification_test.mjs
```

The tests cover recognition-variant normalization, full and abbreviated callsigns, explainable controller/pilot/unknown role inference, controller parking prompts and pilot replies, controller destination instructions, ambiguous and interleaved turns, weak and strong destination evidence, deterministic matching, pending confirmation, corrections before and after publication, per-aircraft isolation, conservative handling of “no”, duplicate suppression, ADS-B failure continuity, REST authentication, pending/correction extension behavior, notification delivery, audio-WebSocket authentication, injected PCM processing, and the exact authorized LiveATC URL boundary. Confirmation tests use short configured delays; all nearby-aircraft and audio inputs are mocks, so unit tests never contact an external API, download a speech model, open audio hardware, or access LiveATC.

Model-evaluation tests additionally cover standard and Hugging Face identifiers, invalid identifier
rejection, CPU/CUDA diagnostics, explicit and disabled fallback, per-model overrides, local WER/CER,
aviation entity scoring, RTF categories, expected text/JSON loading, downstream detector reports,
default-disabled notification delivery, backlog warnings, and secret-free logs. Fake transcribers
and CTranslate2 modules ensure these tests require no GPU, model download, network, or audio device.

Manual quality evaluation uses lawfully obtained 16 kHz mono PCM WAVs that are intentionally not
committed. Run `compare-models` for one file and `evaluate-models` for a directory. Review entity
accuracy and alert precision/recall alongside WER and RTF; do not select a model solely from decoder
log probability.

Subprocess tests simulate success, native hangs/timeouts, exceptions, cleanup, continuation, and
Ctrl+C without loading a model or GPU library. ADS-B tracker tests use local observations for
haversine distance, toward/away trends, recent landing, taxi/ground phase, speech conflicts,
minimum margins, stale/airborne rejection, duplicate registrations, missing fields, and fixture
loading. No test sends notifications or performs a live ADS-B request.

Live-regression coverage also includes KMRY-scoped `say barking` recovery, the observed garbled runway/Foxtrot transcript, “Alpha Echo to the Jet Center,” generic business-mention rejection, repetitive decoder tails, spoken registrations without ADS-B, exact and ambiguous suffixes, damaged fuzzy callsigns, unsafe fragments, unique and competing ground candidates, transcript-quality logging, delivery counts, partial failures, no-client delivery, correction logging, token/PCM redaction, and duplicate delivery-log suppression.

Speech-pipeline tests mock model inference and cover the `small.en` default, configurable decoding arguments, static-plus-dynamic prompting, N-number spoken forms, candidate limits, ADS-B absence, preprocessing silence/low-level/clipped/disabled behavior, pre-roll, 350 ms silence termination, and the 8-second segment ceiling. Tests never download a model.

Acceptance fixture checks:

```bash
python -m mry_alert simulate --fixture tests/fixtures/direct_request.json
python -m mry_alert simulate --fixture tests/fixtures/split_exchange.json
python -m mry_alert simulate --fixture tests/fixtures/ambiguous.json
```

Expected results are a confirmed N123AB alert, a context-linked likely N123AB alert, and no emitted alert for two equally plausible suffix matches.

To perform a manual server/extension smoke test, start `python -m mry_alert serve --config config.yaml`, open `chrome://extensions`, press **Reload** on the unpacked extension after any source change, pair it with `data/pairing_token.txt`, and use **Test notification**. Confirm the Windows notification appears and the extension service-worker console has no errors. This validates the real REST-to-WebSocket-to-Chrome path without audio or ADS-B.

Optional hardware tests depend on the user's OS devices and are deliberately outside the automated suite. Test only audio that was obtained lawfully.

For an authorized LiveATC manual test, install `.[liveatc]`, start the backend, reload the extension, play the approved KMRY player, and press **Start KMRY audio** while that tab is active. Confirm the popup reaches `monitoring`, shows a new transcript after a transmission, remains audible, and returns to `stopped` when requested. Do not test another feed.

Desktop-control tests cover source and PyInstaller path resolution, command construction,
whitelisted Windows parsing, shell-operator rejection, process ownership, external backend
detection, health/test-alert behavior, status mapping, inactive audio, ADS-B degradation,
configuration validation, and token/log redaction. Core tests do not import PySide6 or require a
desktop session; GUI packaging is verified separately by the Windows build script.

Audio-classifier tests use tiny synthetic PCM and mock classifiers. They cover the protocol,
default-disabled compatibility, strong Monterey evidence, Del Monte suppression, correction
linkage, noise and weak-evidence rejection, required ADS-B resolution, Whisper disagreement,
inference timeout, model-load failure, token redaction, reviewed-label precedence, hard-negative
queues, class weights, metrics, confusion matrices, and the offscreen desktop review workflow.
No classifier test downloads a model, opens hardware, performs a cloud call, or delivers a real
notification.
