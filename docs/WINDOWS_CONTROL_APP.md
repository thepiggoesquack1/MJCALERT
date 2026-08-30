# MRY Alert Control

MRY Alert Control is a PySide6 Windows 11 launcher and dashboard for the existing local backend. It
does not replace transcription, ADS-B correlation, detection, Chrome notification, or authorized
LiveATC capture code.

## Run from source

```powershell
cd "C:\Users\liamr\OneDrive\Documents\ATC recognition software"
python -m pip install -e ".[liveatc,control]"
python -m mry_alert.control
```

The app resolves `config.yaml`, `data\`, and `logs\` relative to the project or packaged application
directory. **Select Config** chooses another YAML file. Invalid configuration blocks backend startup
and marks Config red.

## Indicators and ownership

- Green check: healthy or connected.
- Yellow warning: starting, waiting, inactive, or degraded.
- Red X: failed, unavailable, or stopped.
- Gray circle: no observation yet.

Audio connection and recent PCM activity are separate. ADS-B provider status, tracked-aircraft
count, errors, and recent correlation are separate observations.

The dashboard also shows whether the audio classifier is enabled and loaded, its model version,
whether dataset collection is active, and the pending review count. **Open Review Queue** plays a
saved WAV through the local Windows audio handler and lets the operator correct destination,
intent, correction, callsign, unintelligible, and hard-negative labels. **Train** and **Evaluate**
run the corresponding CLI commands in the existing separate command process, so the GUI remains
responsive and the command can be cancelled.

The Training tab's **Start Recording Clips** / **Stop Recording Clips** button toggles only
`training_data.enabled` in the active configuration. After validating the edited YAML, it stops a
Control-owned backend and relaunches the complete Control app and backend. An externally launched
backend must be stopped first because the Control app never terminates processes it does not own.

Start launches the existing backend as a child and captures stdout/stderr. Stop first requests
termination and then ends only that owned process tree if necessary. A healthy server already on
the configured port is labeled **External backend detected**; the app does not stop it. On exit,
the user chooses whether to stop a GUI-owned backend.

## Safe command console

Enter only arguments that normally follow `python -m mry_alert`. Diagnostics, comparison, replay,
evaluation, device listing, microphone monitoring, transcription, and simulation are supported.
Local classifier training and evaluation are also whitelisted.
Commands run separately, one at a time, and can be cancelled. The parser rejects `&`, pipes,
redirects, semicolons, backticks, and `$()` and never invokes PowerShell, cmd.exe, or `shell=True`.

## Logs and token protection

Live stdout/stderr supports filtering, searching, pausing, auto-scroll, copying, and saving. GUI
sessions are stored under `logs\gui\`. `token=...` and pairing-token headers are redacted before
display or storage. The token card shows only Found or Missing. **Copy Pairing Token** reads the
configured token into the Windows clipboard without displaying or logging it. Test Alert reads it only for
the authenticated request.

## Build and packaging

```powershell
.\build_windows.ps1
```

Output:

```text
dist\MRY Alert Control\
  MRY Alert Control.exe
  MRY Alert Backend.exe
  config.yaml
  config.example.yaml
  data\
  logs\gui\
```

One-directory mode avoids one-file extraction delay, exposes DLL issues, and keeps operational files
editable. Whisper weights and Hugging Face/CTranslate2 caches stay external. CUDA still needs
compatible NVIDIA, CUDA, and cuDNN DLLs.

Optional shortcuts:

```powershell
.\create_windows_shortcuts.ps1 -Desktop -StartMenu
```

Windows startup is never modified automatically. Its GUI option requires confirmation.

## Troubleshooting

- Backend red: inspect logs for port, Python, or configuration errors.
- Extension yellow: reload and pair the extension; verify WebSocket support.
- Audio inactive: start the authorized KMRY player and capture.
- Speech yellow: the model is lazy-loaded with audio.
- ADS-B yellow: no nearby tracked aircraft may be available.
- ADS-B red: inspect the provider error and network.
- Test alert failure: verify health, token, and extension connectivity.

The dashboard remains advisory and cannot guarantee ADS-B, transcription, GPU, or Chrome behavior.
The bundled application does not include a trained classifier or Whisper weights. Keep
`data\models\atc_intent_classifier\model.json` external and review evaluation output before
enabling the model.
