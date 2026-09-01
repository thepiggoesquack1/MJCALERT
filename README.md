# MRY Jet Center Alert

MRY Jet Center Alert is a local, advisory application that listens to legally obtained ATC audio, transcribes it on your computer, recognizes strong evidence that an aircraft is going to Monterey Jet Center at KMRY, correlates the spoken callsign with nearby ADS-B reports, and sends an event to a local Chrome extension. Written permission now allows this internal tool to use the specified KMRY LiveATC web player through an explicit, user-initiated browser capture.

It supports offline transcript simulations now, plus optional WAV and microphone/virtual-device input using local Whisper speech recognition. It uses no OpenAI API, cloud speech API, paid AI service, or token-based inference service.

> **Advisory only:** ATC transcripts can be wrong. ADS-B registrations can be missing or stale, and abbreviated callsigns can be ambiguous. Never treat the popup as a safety-critical source. Raw audio is not saved by default. Respect the terms of every external data provider.

## Authorized LiveATC scope

LiveATC granted written permission on July 18, 2026 for the internal, noncommercial use described in the request. The extension captures only `https://www.liveatc.net/hlisten.php?mount=kmry&icao=kmry` after the user presses **Start KMRY audio**. It does not scrape the page, discover or embed an underlying stream URL, use archives, redistribute/rebroadcast audio, or save LiveATC audio. See [the permission record and enforced boundary](docs/LIVEATC_INTEGRATION.md).

## Authorized ADSB.fi scope

The project owner confirmed written ADSB.fi permission on August 9, 2026 for this internal Monterey
Jet Center use case. The backend uses the documented ADSB.fi open-data v3 nearby-aircraft endpoint,
not the interactive Globe webpage, and polls every ten seconds. ADS-B data is provided courtesy of
[ADSB.fi](https://adsb.fi/). See [the authorization record and integration boundary](docs/ADSB_FI_INTEGRATION.md).

This first MVP does not provide an installer, historical database, model-management UI, production service manager, or guaranteed aircraft identification. ADS-B API availability and terms can change.

## Download and use on Windows

These steps install the complete application from source. You need Windows 11, Google Chrome,
Python 3.11 or newer, and an internet connection for the initial Python/Whisper installation and
live ADS-B data. An NVIDIA GPU is optional; local transcription can run on the CPU.

### 1. Download the project

1. Open the [MJCALERT GitHub repository](https://github.com/thepiggoesquack1/MJCALERT).
2. Select **Code**, then **Download ZIP**.
3. Extract the ZIP to a permanent folder such as `Documents\MJCALERT`. Do not run it from inside
   the ZIP file.
4. Open the extracted folder in File Explorer, click the address bar, type `powershell`, and press
   Enter.

Git users may instead run `git clone https://github.com/thepiggoesquack1/MJCALERT.git` and open
PowerShell in the cloned folder.

### 2. Install the application

Install 64-bit [Python 3.11 or newer](https://www.python.org/downloads/) and enable **Add Python to
PATH** in the installer. Then run these commands in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[liveatc,control]"
Copy-Item config.example.yaml config.yaml
```

The last command creates the private, working configuration. `config.example.yaml` is the public
template; `config.yaml` is intentionally excluded from Git so that local settings and credentials
are not published. If `config.yaml` already exists, keep it and do not overwrite it.

### 3. Start the desktop application

Run:

```powershell
.\.venv\Scripts\python.exe -m mry_alert.control
```

In **MRY Alert Control**:

1. Confirm that the selected configuration is `config.yaml`. Use **Select Config** if necessary.
2. Select **Start Server**.
3. Wait for the backend and configuration indicators to show healthy.
4. Select **Test Alert** after the Chrome extension is paired in the next step.

The first real audio session may download the configured local Whisper model and can take several
minutes to become ready. Later launches reuse the downloaded model.

### 4. Install and pair the Chrome extension

1. Open `chrome://extensions` in Chrome.
2. Enable **Developer mode**.
3. Select **Load unpacked**, then choose the project's `extension` folder.
4. Return to MRY Alert Control and select **Copy Pairing Token**.
5. Open the MRY Alert extension, leave the backend URL as `http://127.0.0.1:8765`, paste the token,
   and select **Save & connect**.
6. Confirm that the extension says **Connected**.
7. Select **Test notification** in the extension or **Test Alert** in the desktop application. A
   clearly labeled test notification should appear in Windows.

The backend and Chrome must run on the same computer. If Windows notifications do not appear,
allow notifications for Google Chrome in Windows Settings and make sure Do Not Disturb is off.

### 5. Start authorized monitoring

LiveATC permission described in this repository applies only to the documented internal Monterey
Jet Center use case; downloading the software does not grant another person permission to capture a
LiveATC feed. Other users must obtain any required permission and comply with the relevant provider
terms before enabling capture.

For the authorized KMRY workflow:

1. Keep MRY Alert Control and its backend running.
2. Open the approved [KMRY LiveATC player](https://www.liveatc.net/hlisten.php?mount=kmry&icao=kmry)
   in Chrome and start its audio playback.
3. While that player tab is active, open the extension and select **Start KMRY audio**.
4. Wait for the extension status to reach **Monitoring**. Keep the player tab open.
5. Use the desktop dashboard's **Recent Notifications** and **Event History** tabs to review
   results. The history is cleared whenever the backend is restarted.

To stop, select **Stop** in the extension and **Stop Server** in MRY Alert Control. For future use,
open PowerShell in the project folder and run only:

```powershell
.\.venv\Scripts\python.exe -m mry_alert.control
```

If setup fails, see [Troubleshooting](#troubleshooting) and the
[Windows control application guide](docs/WINDOWS_CONTROL_APP.md).

## How it works

Audio arrives as 16 kHz mono signed 16-bit PCM. WebRTC VAD groups speech into individual push-to-talk turns using a 250 ms pre-roll, 350 ms end-silence threshold, and 8-second ceiling. A conservative preprocessing pass removes DC offset and radio rumble, low-passes above 3.8 kHz, normalizes level, and protects against clipping before local `faster-whisper` decoding. Deterministic normalization, callsign parsing, and 120-second contact context then identify destination evidence. A Monterey Jet Center detection remains pending for 8 seconds by default, allowing a same-contact correction before a desktop alert appears.

The confirmation delay and 20-second post-alert correction window are configurable as `detection.destination_confirmation_delay_seconds` and `detection.destination_correction_window_seconds`. During the delay, the extension may show pending state in recent history but never raises a desktop notification. A correction after confirmation creates a high-priority correction notification, references and visibly updates the original event, and states that the aircraft is no longer expected at Monterey Jet Center. Corrections are associated conservatively with the same full/abbreviated callsign or unambiguous radio-contact context; an unrelated use of “no” cannot cancel an alert.

Every transmission also receives an explainable `speaker_role`, `speaker_role_confidence`, and `speaker_role_reasons`. The role is inferred only from transcript structure, aviation phraseology, callsign position, and turn-taking. It does not identify or compare the actual controller or pilot voice, and no voice biometrics are used. Ambiguous wording remains `unknown`. Role evidence can strengthen an already supported conversational link or suppress an obvious controller instruction, but it cannot by itself assign an aircraft or cancel an alert.

For the authorized KMRY ground source only, an ATC intent-normalization pass recovers a small set of predictable recognition errors such as `say barking`/`say marking` to `say parking`, `going on/for Monterey Jet` to `going to Monterey Jet`, Jet Center homophones, and strongly contextual `tech to`/`taxi two`. Aviation digit variants are normalized only inside callsign-like spans. Generic English is left unchanged. The destination detector separately classifies explicit taxi requests, parking statements, prompt responses, strong ground routes, weak mentions, and no intent. High-confidence routes such as “Alpha Echo to the Jet Center” no longer require the literal word `taxi`.

ADS-B remains secondary to speech intent. It first tries a full registration, ADS-B flight callsign, and unique suffix; then optionally tries high-threshold fuzzy spoken-form recovery against already filtered nearby aircraft. With no usable callsign, exactly one recent, low-speed aircraft on or immediately adjacent to KMRY may be labeled `likely` as `unique_ground_candidate`. Proximity alone is never `confirmed`; competing candidates remain ambiguous, and a full spoken N-number is retained if ADS-B is unavailable.

Live segmentation defaults are tuned for the observed KMRY radio turns: 300 ms pre-roll, 650 ms
end silence, and 15-second maximum segments. This keeps brief radio pauses and callsign readbacks in
the same segment while still separating distinct transmissions. Maximum duration creates an adjacent
segment rather than discarding the continuation, so conversational context survives. Repeated decoder
tails are trimmed from detector text only. Raw LiveATC PCM remains transient unless training-data
collection is explicitly enabled.

The active `config.yaml` uses the same `300/650/15` values. The previous eight-second maximum was
observed cutting longer taxi instructions before or immediately after their aircraft identity.

## Local audio-intent classifier

Full transcription remains useful for operators, callsign clues, and diagnostics, but short,
compressed radio turns can be transcribed incorrectly. The optional classifier answers only the
operational questions: destination, taxi/parking intent, correction, competing Del Monte
destination, and unusable audio. When enabled with `decision_fusion.classifier_primary: true`, a
high-confidence Monterey destination and relevant intent plus a resolved ADS-B aircraft are
required. Whisper disagreement cannot erase that evidence, and Whisper alone cannot alert unless
`allow_whisper_only_alerts` is explicitly enabled.

The classifier and audio collection are disabled by default. To collect authorized clips, manually
set `training_data.enabled: true` and `training_data.save_audio: true`, then start normal monitoring.
Individual segmented transmissions are queued automatically; there is no Record button. Collection
works before a classifier model exists and records classifier status as disabled, unavailable,
failed, or available. **Open Review Queue** validates the files, skips damaged/incomplete items with
an explanation, and lets the reviewer play, correct, reject, or mark a hard negative. Nothing is
uploaded. Disable collection again after obtaining the needed clips.

Train and evaluate the lightweight local baseline:

```powershell
python -m mry_alert train-audio-classifier --dataset data/training_clips/reviewed --output data/models/atc_intent_classifier --config config.yaml
python -m mry_alert evaluate-audio-classifier --dataset data/training_clips/reviewed --model data/models/atc_intent_classifier --config config.yaml
```

Only valid human-reviewed labels enter training. Approved hard negatives are forced onto the
`no_destination` / `no_relevant_intent` path and are included alongside reviewed positives.
Predictions remain reviewable and are never promoted to ground truth automatically. The portable
centroid baseline is intended to establish the local data,
review, fusion, and evaluation workflow; do not trust it operationally until held-out evaluation
shows very low Monterey false-positive rates across representative radio conditions.

`detection.alert_on_any_destination_mention` remains in the schema for configuration compatibility,
but a bare Monterey Jet Center mention never creates an arrival. Arrival evidence must describe a
pilot request, route, parking-prompt response, or equivalent supported turn. Current-location and
departure wording is evaluated separately against FBO geofences and movement history.

`immediate_notification_on_clear_ground_match` can confirm an explicit pilot-arrival event immediately
when ADS-B already supplies a clear registration and confirmed ground/recent-landing match.
Corrections remain accepted afterward; ambiguous or unresolved identities still wait or are
withheld.

## Understanding the PowerShell output

Operational blocks use the `mry_alert.operations` logger so they are visually distinct from Uvicorn and faster-whisper messages. With the default `logging.format: detailed`, every completed segment shows the original transcript, inferred role, parsed callsign, normalized intent and recovery reasons, route cues, destination candidate, ADS-B candidates, and one detector decision:

- `TRANSCRIBED`: transcription completed but no later decision has yet been recorded.
- `IGNORED`: no qualifying MJC intent, a weak business mention, or a strongly structured non-arrival transmission.
- `PENDING`: qualifying intent is waiting for the correction delay.
- `CONFIRMED`: the delay elapsed and the alert reached publication.
- `AMBIGUOUS`: contact association or aircraft candidates could not be separated safely.
- `UNRESOLVED`: intent may be understood but aircraft identity could not be resolved.
- `CANCELLED` / `CORRECTED`: a same-contact destination correction changed the pending or published event.

Delivery is logged only after the WebSocket broadcaster returns actual counts. `NOTIFICATION SENT` means at least one connected extension accepted the event; `NOTIFICATION NOT DELIVERED` means none were connected; `NOTIFICATION DELIVERY FAILED` means all attempted sends failed; and `NOTIFICATION DELIVERY PARTIAL` reports successful and failed counts. A correction published after confirmation uses `CORRECTION NOTIFICATION SENT`. Event IDs are de-duplicated in the delivery logger.

`logging.format: compact` produces one-line summaries. The user-facing `Decoder confidence` label is a heuristic over average log probability and no-speech probability. It reflects confidence in the token sequence and does not guarantee transcription accuracy. Operational logs never include pairing tokens, authenticated stream URLs, raw PCM, or saved audio. The Control app's **Copy Pairing Token** button copies the configured token directly to the clipboard without displaying or logging it.

## Install Python

Install Python 3.11 or newer from [python.org](https://www.python.org/downloads/). On Windows, enable **Add Python to PATH** during installation. The commands below use `python`; on some macOS/Linux systems use `python3` instead.

Create and activate a virtual environment:

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the core application and development tools:

```bash
python -m pip install -e ".[dev]"
```

Copy the sample configuration before changing settings:

```powershell
Copy-Item config.example.yaml config.yaml
```

On macOS/Linux, use `cp config.example.yaml config.yaml`.

## Run without audio hardware or a model

Simulation is fully offline and does not import `faster-whisper`, access audio hardware, or contact ADS-B:

```bash
python -m mry_alert simulate --fixture tests/fixtures/direct_request.json
python -m mry_alert simulate --fixture tests/fixtures/split_exchange.json
python -m mry_alert simulate --fixture tests/fixtures/ambiguous.json
```

The first two print N123AB alerts. The ambiguous fixture prints `No alert produced.` because normal unresolved/ambiguous notifications are disabled.

## Run the local backend and extension

Start the backend:

```bash
python -m mry_alert serve --config config.yaml
```

It binds to `127.0.0.1:8765` by default. On first run it creates a random token at `data/pairing_token.txt`; startup prints the path but never logs the token. Open that file locally to copy the token.

Load the extension:

1. Open `chrome://extensions` in Chrome.
2. Enable **Developer mode**.
3. Choose **Load unpacked** and select this repository's `extension` folder.
4. Open the extension popup.
5. Keep the backend URL as `http://127.0.0.1:8765`, paste the pairing token, and select **Save & connect**.
6. Confirm the popup says **Connected**.
7. Select **Test notification**. The backend creates a clearly labeled test event and sends it over the WebSocket.

The Chrome 116+ extension stores its URL, token, mute setting, minimum confidence, recently seen event IDs, and ten latest alerts in `chrome.storage.local`. Its service worker keeps the WebSocket alive with a 20-second local message, reconnects with capped exponential backoff, and escapes displayed content using DOM text nodes.

## Local speech and audio

Install local Whisper support and audio/VAD support separately because these packages are larger and platform-dependent:

```bash
python -m pip install -e ".[speech,audio]"
```

For the authorized LiveATC browser workflow, microphone support is unnecessary; install:

```bash
python -m pip install -e ".[liveatc]"
```

The first transcription run downloads the configured `small.en` model through `faster-whisper`; this can take several minutes and requires internet access once. Afterwards inference is local. `small.en` improves fast compressed radio recognition over the legacy `base.en` at a moderate CPU/memory cost. Set `speech.model` to `medium.en` or `distil-large-v3` for potentially better recognition with substantially higher memory, download size, and latency. CPU `int8` remains the default.

The focused KMRY prompt and decoding controls (`language`, `beam_size`, `temperature`, `condition_on_previous_text`, and `vad_filter`) are configurable. Previous-text conditioning and faster-whisper internal VAD are disabled by default because this application already segments radio turns. A background 7-second ADS-B cache may append up to five nearby N-numbers and ICAO spoken forms to the static prompt; prompted registrations are hints only and never establish identity.

Audio preprocessing is independently configurable under `audio.preprocessing`. Noise reduction remains disabled because the project has no lightweight suitable dependency; no new dependency was added. Debug WAV output, when explicitly enabled, saves the original segmented PCM before preprocessing.

No sample radio WAV files are included in the repository. To compare a lawfully obtained 16 kHz mono PCM WAV manually without inventing an accuracy score:

```powershell
python -m mry_alert replay --audio path\to\sample.wav --model base.en --config config.yaml
python -m mry_alert replay --audio path\to\sample.wav --model small.en --config config.yaml
```

Replay output includes model, audio duration, segment boundaries, processing time, real-time factor, transcript, decoder metrics, and prompted registrations (empty during offline replay).

### ATC-specialized model evaluation

`speech.model` accepts either a standard faster-whisper name or a conservative Hugging Face
`owner/repository` identifier. The ATC-tuned candidate is optional and is not assumed to be
better than the standard models. Check the configured local runtime first:

```powershell
python -m mry_alert check-speech-runtime --config config.yaml
```

This initializes the configured model and reports CTranslate2/CUDA capability, supported CUDA
compute types, initialization time, and observable GPU memory. It may download the model on its
first run. A CUDA failure usually means the NVIDIA driver or compatible CUDA/cuDNN runtime is
missing; test CPU `int8` if CUDA is not available.

Compare the models using exactly the same WAV, external VAD, preprocessing, prompt, and decoding
settings:

```powershell
python -m mry_alert compare-models --audio "test_audio\direct_request\audio.wav" --expected-transcript "test_audio\direct_request\expected.txt" --models base.en small.en jacktol/whisper-medium.en-fine-tuned-for-ATC-faster-whisper --config config.yaml
```

Without `--expected-transcript`, WER/CER and reference-based aviation entity scores are omitted.
The report still includes transcripts, decoder diagnostics, callsign/destination/intent decisions,
artifact filtering, role inference, notification decisions, per-segment timing, and a final table.
Decoder log probability alone does not determine the best model: correct callsign, destination,
route terms, corrections, alert behavior, and live throughput matter more to this application.

Real-time factor (RTF) is processing time divided by audio duration: `<=0.50` is excellent,
`>0.50–0.80` acceptable, `>0.80–1.00` risky, and `>1.00` unsuitable for live use. Thresholds
are configurable under `speech_performance`. Live monitoring reports backlog and a rolling RTF
warning; it queues transmissions and does not silently discard segmented audio. Slow models remain
available for offline replay.

A local dataset needs no committed audio and can use this structure:

```text
test_audio/
  direct_request/
    audio.wav
    expected.txt
    expected.json
```

`expected.json` may contain `should_notify`, `destination`, `registration`, `intent`, and
`correction`. Evaluate all case directories with:

```powershell
python -m mry_alert evaluate-models --dataset test_audio --models base.en small.en jacktol/whisper-medium.en-fine-tuned-for-ATC-faster-whisper --config config.yaml
```

To switch the live model, manually set `speech.model` in `config.yaml`, choose a compatible
`device` and `compute_type`, then restart the backend. The application never changes that setting
automatically. Optional `speech_model_overrides` apply validated decoding/device settings by model.
Fallback occurs only when both `speech.allow_model_fallback: true` and a valid
`speech.fallback_model` are configured; requested model, fallback, and the actual failure are
prominently logged.

The first use of a model can download a large snapshot. It remains in the local Hugging Face cache
afterward. An HF token is not required, although anonymous downloads can be rate-limited. Windows
symlink/cache warnings are normally nonfatal and may use more disk space. Never put an HF token in
the YAML. Medium-sized models require substantially more RAM/VRAM, and CUDA `float16` must be
supported by both the GPU and installed runtime.

Multi-model comparison runs each model in a separate Python subprocess. This isolates Windows
CUDA/CTranslate2 state and guarantees that a stuck native decoder cannot hang the whole comparison.
`benchmark.model_timeout_seconds` defaults to 180 seconds; timed-out workers are terminated and the
next model continues. Timing logs distinguish audio loading, model construction, preprocessing,
transcribe entry, the single segment-generator consumption, detector work, and cleanup. Ctrl+C
terminates the active worker and returns exit code 130 without nested tracebacks.

## ADS-B-first aircraft correlation

Speech remains the sole trigger for Monterey Jet Center destination/taxi intent. Once speech
qualifies, ADS-B is the primary aircraft-identity source. A later callsign/readback can safely enrich
one recent anonymous arrival candidate when there is no competing pending candidate, and the longer
ADS-B window allows delayed observations to resolve it. The in-memory tracker retains recent
positions, speeds, altitudes, ground state, heading, vertical rate, squawk, first/last seen times,
distance trend, and recent-landing evidence. Nothing is uploaded or persisted unless the optional
local debug snapshot is explicitly enabled.

Candidate scoring is fully configurable under `adsb_scoring`. Positive evidence includes being
inside the airport, on ground, recently landed, taxiing, near or moving toward the Jet Center,
recent activity, and speech agreement. Staleness, airborne climbing, moving away, incomplete data,
and duplicate registrations are negative. `adsb_decision` requires a minimum score, minimum margin,
maximum age, and an on-ground/recently-landed phase. Speech/ADS-B conflicts, stale winners, airborne
departures, and close candidates are never assigned a guessed tail. If the same full spoken
registration is linked in at least two transmissions and nearby ADS-B confirms KMRY ground traffic,
the alert may use that repeated radio identity without substituting another aircraft. If a strong
pilot arrival still cannot be identified by the end of the 45-second correlation window, an
unidentified-arrival notification is sent instead of silently losing the arrival. Controller-only
route candidates remain pending and expire without notification unless linked pilot evidence arrives.

The existing airport coordinates seed only the broad KMRY circle. Jet Center, runway, movement,
and approach geofences remain `null` in `config.example.yaml`; enter coordinates verified by the
operator. The software does not guess them. Directional Jet Center evidence is unavailable until
that geofence is configured, and ADS-B coverage/fields can be incomplete or delayed.

Replay with local timestamped ADS-B observations is offline and never sends notifications:

```powershell
python -m mry_alert replay --audio test.wav --adsb-fixture adsb_history.json --config config.yaml
```

The fixture root contains `observations`; each item accepts `timestamp_offset_seconds`, `hex`,
`registration`, `flight`, `latitude`, `longitude`, `altitude_ft`, `ground_speed_knots`, `track`,
`vertical_rate`, `squawk`, and `on_ground`. Replay reports the detector decision, correlated
registration, and whether a notification would have been sent.

List devices:

```bash
python -m mry_alert list-audio-devices
```

Monitor a microphone or virtual audio device:

```bash
python -m mry_alert monitor-microphone --device 3 --config config.yaml
```

Transcribe a development WAV file:

```bash
python -m mry_alert transcribe-file path/to/audio.wav --config config.yaml
```

WAV input must already be 16 kHz, mono, signed 16-bit PCM. The file and microphone commands run local transcription and optional live ADS-B lookup. Raw audio is discarded after processing unless `audio.save_debug_audio` is explicitly enabled. When enabled, each segmented transmission is written under the configured `audio.debug_audio_directory`; these recordings are ignored by Git.

## Monitor the authorized KMRY LiveATC player

1. Install `.[liveatc]`, keep `liveatc.enabled: true`, and start the backend with `python -m mry_alert serve --config config.yaml`.
2. Reload the unpacked extension after updating it. Chrome will show new `activeTab`, `tabCapture`, and `offscreen` permissions.
3. Open the approved [KMRY LiveATC player](https://www.liveatc.net/hlisten.php?mount=kmry&icao=kmry) and start playback.
4. While that player tab is active, open the extension and press **Start KMRY audio**.
5. Watch the popup progress through `connected`, `loading_model`, and `monitoring`. The first run may download the configured Whisper model; audio arriving before the model is ready is discarded.
6. The popup shows the last locally produced transcript. Press **Stop** when monitoring is no longer needed.

Capture is rejected for HTTP, another host/path, or any mount/ICAO other than KMRY. Player audio continues through the speakers during monitoring. LiveATC PCM is never written to disk, including when general microphone/file debug recording is enabled.

## Tests and checks

```bash
python -m pytest
python -m ruff check .
python -m mypy src
```

Unit tests mock external data and never make live network requests. More detail is in [docs/TESTING.md](docs/TESTING.md).

## Merging new configuration sections

`config.example.yaml` is the schema reference; do not replace a working `config.yaml` with it.
To adopt a new section, stop the backend, copy only that top-level block (for example
`traffic_filter:`, `intent_detection:`, or `ntfy:`) into `config.yaml`, preserve all existing paths,
tokens, URLs, LiveATC settings, and local model choices, then restart the backend. YAML
top-level keys must have no indentation and each nested setting must remain indented by two
spaces. The application supplies safe defaults when either new block is absent.

### Optional ntfy phone push

Install the ntfy app on the phone, subscribe to a private topic name, and put that same
topic under `ntfy.topic`. Set `enabled: true` and restart the backend. The desktop
Notifications tab shows the configured server, subscribed topic, subscription URL,
whether an Authorization header is configured (the secret itself stays hidden), and
the most recent delivery result. Chrome notification delivery is independent and
continues normally if ntfy is disabled or temporarily unavailable.

For an access-controlled ntfy server, set `authorization` to the complete header value
required by that server, such as `Bearer tk_example`. Use a hard-to-guess topic when
publishing through the public `https://ntfy.sh` service.

Outbound parking reports are checked against the Monterey Jet Center polygon and any
optional named entries under `adsb_geofences.fbo_geofences`. A clear request from an
aircraft that was already parked inside an FBO and is taxiing for departure is filtered.
Recently landed aircraft, aircraft moving toward Monterey Jet Center, and explicit
"taxi to Jet Center" wording remain arrival-eligible. Ambiguous movement is recorded as
unresolved instead of being silently suppressed.

## Troubleshooting

- **`python` is not recognized:** reinstall Python with PATH enabled, restart the terminal, or use `py -3.11` on Windows.
- **Optional package error:** install the relevant `speech` or `audio` extra shown above.
- **No audio devices / PortAudio error:** verify OS microphone permission, reconnect the device, and rerun `list-audio-devices`. Linux may require the distribution's PortAudio package.
- **Wrong device:** use the numeric input-device index, not an output-device index.
- **Model download or memory problem:** check disk/network access for the first download, keep the CPU `int8` default, or choose a smaller configured model.
- **WAV rejected:** convert it to exactly 16 kHz, mono, signed 16-bit PCM.
- **Extension disconnected:** keep the backend running, verify `127.0.0.1:8765`, paste the current token exactly, then save again.
- **KMRY capture will not start:** make the exact authorized LiveATC player the active tab, start its playback, then open the extension and press Start. Reload the extension once after installing this update.
- **Audio status says error:** install `.[liveatc]`; inspect the popup detail and backend log for model/VAD initialization errors.
- **No aircraft resolution:** ADS-B may be unavailable, stale, absent, or ambiguous. The pipeline continues safely without guessing.
- **No alert from a destination mention:** this is intentional; the detector requires independent taxi/parking or conversational evidence, then waits for the configured confirmation delay.

## Windows control application

The optional PySide6 desktop application starts and monitors the same backend without requiring an
open PowerShell window:

```powershell
python -m pip install -e ".[liveatc,control]"
python -m mry_alert.control
```

Use **Start Server** to launch the existing configured backend, **Stop Server** to stop only the
GUI-owned process, and **Test Alert** to call the authenticated endpoint. A healthy server already
using the configured port is shown as **External backend detected** and is never stopped by the GUI.

The command console accepts only supported `mry_alert` arguments. It never launches a shell and
rejects shell operators. Green is healthy, yellow is waiting/degraded, red is failed, and gray means
no observation yet. Extension connectivity, audio connection, and recent audio activity are
separate indicators.

The dashboard **Alert sensitivity** selector writes only
`detection.alert_sensitivity` and restarts a GUI-owned backend when **Apply & Restart** is pressed:

- `conservative` requires linked pilot-arrival phraseology and strong ADS-B evidence.
- `balanced` also sends a possible-arrival alert for a controller routing instruction when one
  ADS-B aircraft is confirmed and moving toward Monterey Jet Center. This is recommended for feeds
  that sometimes omit the pilot side of an exchange.
- `never_miss` accepts an exact Jet Center mention when plausible, non-outbound ground ADS-B traffic
  exists. It provides the highest coverage but can generate more possible-arrival alerts.

All modes retain outbound-from-MJC filtering, airline filtering, correction handling, and duplicate
suppression. Sensitivity-supported controller or mention-only alerts are labeled `possible`; they do
not claim that a pilot transmission was heard.

The **Event History** tab is a complete audit view for the current backend session. It includes
confirmed and possible alerts, pending/expired/corrected/cancelled events, denied or filtered outcomes,
duplicate suppression, and Chrome/ntfy delivery results. Search matches aircraft identity, type,
operator, destination, transcript, decision reasons, and event ID; filters and sorting operate on
the displayed rows. Selecting a row opens the full evidence breakdown. The buffer defaults to
1,000 records under `event_history.maximum_events`. It exists only in backend memory, survives a
GUI reconnect to the same running backend, and is empty after a backend restart. **Export** writes
only the currently displayed rows and only when pressed.

Pending rows are provisional: when the correlation window finishes, the same row is replaced by
its confirmed, expired, cancelled, unresolved, or duplicate-suppressed outcome. **Seen**,
**Aircraft Arrived**, and **False Detection** buttons are available below Event History and Recent
Notifications. These acknowledgements are operational feedback held only in the current backend
session; they are not written to the training dataset and do not retrain or modify the classifier.

The Training tab has a **Start Recording Clips** / **Stop Recording Clips** button. It changes only
`training_data.enabled` in the active `config.yaml`, validates the result, stops the Control-owned
backend, and relaunches both the Control app and backend. Clip collection is therefore applied as a
clean application-wide setting rather than changing midway through an audio session.

Aircraft type is taken from direct ADS-B/provider metadata when available. A type is never guessed
from speed, category, operator, appearance, or callsign, and an unknown type never blocks an alert.
Chrome and ntfy notifications explicitly say `Aircraft type unknown` when no reliable metadata is
available. The ntfy HTTP title uses an ASCII hyphen because Python's standard HTTP transport does
not permit an em dash in a header value; Chrome continues to use the typographic em dash.

Build the Windows 11 one-directory package with `.\build_windows.ps1`. The executable is created at
`dist\MRY Alert Control\MRY Alert Control.exe`. Keep `config.yaml`, `data\`, and `logs\` beside it.
Whisper weights remain external. See
[the Windows control application guide](docs/WINDOWS_CONTROL_APP.md).

## MVP limitations

The authorized KMRY player may contain whatever frequencies LiveATC supplies in that feed; this application does not manipulate or isolate an underlying stream. Model download progress is provided by the underlying package, ADS-B distance is delegated to the nearby endpoint, and configuration changes require restart. Notification history remains in its existing JSONL store; the comprehensive Event History is strictly session-memory-only. The Chrome extension must be loaded unpacked. Identification remains probabilistic and advisory.
