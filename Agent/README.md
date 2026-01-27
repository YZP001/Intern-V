# XLeRobot Voice Agent (RoboCrew)

This folder contains a small voice-controlled agent for XLeRobot based on the RoboCrew library.

The robot only reacts to speech that contains the activation keyword (wakeword). Default wakeword: `robot`.

## Install

Create/activate a Python environment on the machine that is connected to the robot (Raspberry Pi / laptop), then:

```bash
pip install robocrew
```

If you use Gemini models, also create a `.env` file (or export env vars) with your API key, e.g.:

```bash
GOOGLE_API_KEY=...
```

Important: keep your API key secret. Do not hardcode it into scripts or commit it to git.
This agent will also auto-load a local `.env` (current directory or `Agent/.env`) if present.

## Configure

You need:

- Serial port of the arm connected to the wheels (examples: `/dev/ttyUSB0`, `/dev/right_arm`, `COM3`)
- Main camera (Linux example: `/dev/video0`; Windows example: OpenCV index `0`)
- Microphone device index (see below)

Optional environment variables:

- `XLE_ROBOT_SERIAL_PORT`
- `XLE_ROBOT_LEFT_ARM_PORT` (optional; not used by this wheels-only agent)
- `XLE_ROBOT_MAIN_CAMERA_PORT`
- `XLE_ROBOT_SOUNDDEVICE_INDEX`
- `XLE_ROBOT_WAKEWORD` (default: `robot`)
- `XLE_ROBOT_WAKEWORD_ALIASES` (comma-separated extra wakewords that also activate the robot)
- `XLE_ROBOT_LLM_MODEL` (default: `google_genai:gemini-3-flash-preview`)
- `XLE_ROBOT_WHEEL_STEP_SECONDS` (default: `0.1`, smaller = faster interrupt)
- `XLE_ROBOT_STT_BACKEND` (default: `vosk`, options: `vosk`, `gemini`, `robocrew_openai`)
- `XLE_ROBOT_STT_MODEL` (default: `gemini-3-flash-preview`)
- `XLE_ROBOT_STT_MIN_REQUEST_INTERVAL` (default: `13.0`, helps avoid Gemini free-tier rate limits)
- `XLE_ROBOT_STT_DROP_OLD_SEGMENTS` (default: `true`)
- `XLE_ROBOT_STT_VOICE_FRAMES_NEEDED` (default: `2`)
- `XLE_ROBOT_STT_COOLDOWN_SECONDS` (default: `0.5`)
- `XLE_ROBOT_STT_MODEL_PATH` (Vosk model directory when using offline STT)
- `XLE_ROBOT_STT_SAMPLE_RATE` (default: `16000`)
- `XLE_ROBOT_STT_PRINT_PARTIALS` (default: `false`)

## Call Gemini API (gemini-3-flash-preview)

This repo also includes a minimal standalone demo that calls Gemini directly:

```bash
# PowerShell (prompt usually starts with `PS`, e.g. `PS E:\VLA>`)
$env:GOOGLE_API_KEY="YOUR_KEY"
python Agent/gemini_3_flash_preview_demo.py "Explain what VLA is in 3 sentences."
```

If your terminal prompt looks like `E:\VLA>` (CMD), use:

```bat
set "GOOGLE_API_KEY=YOUR_KEY"
python Agent\gemini_3_flash_preview_demo.py "Explain what VLA is in 3 sentences."
```

You can use the same key for the voice agent. Example (LangChain model string):

```bash
python Agent/xlerobot_voice_agent.py --model google_genai:gemini-3-flash-preview ...
```

## List audio devices

```bash
python Agent/xlerobot_voice_agent.py --list-audio-devices
```

Pick an index with a non-zero `in=` value and pass it via `--sounddevice-index`.

## Run

Speech-to-text:

- Default: Offline Vosk STT (recommended; no API quota).
- Optional: RoboCrew OpenAI STT: `--stt-backend robocrew_openai` (requires `OPENAI_API_KEY`).
- Optional: Gemini STT: `--stt-backend gemini` (uses `GOOGLE_API_KEY`, may hit 429 free-tier limits).

Note on Gemini rate limits (HTTP 429):

- Gemini free tier has a low requests-per-minute quota per model. STT and the LLM both consume it.
- If you hit 429, either wait, enable billing / increase quota, or switch STT to `robocrew_openai`.
- You can also reduce accidental STT calls by increasing `--stt-rms-threshold` and `--stt-min-record-seconds`.

Windows example (right arm/wheels = COM3, main camera = index 0):

```bash
python Agent/xlerobot_voice_agent.py --right-arm-port COM3 --main-camera 0 --sounddevice-index 0 --wakeword robot
```

Offline STT setup (Vosk):

1) Install Vosk:

```bash
pip install vosk
```

2) Download and unzip a Vosk model (example: multilingual small model) into:

`Agent/models/vosk-model-small-cn-0.22`

3) Run with:

```bash
python Agent/xlerobot_voice_agent.py --stt-backend vosk --stt-model-path Agent/models/vosk-model-small-cn-0.22 ...
```

Wakeword tip for offline STT (Vosk):

- Vosk models are language-specific. If you use a Chinese model, saying `robot` may be transcribed as Chinese characters.
- If that happens, add aliases so the robot still activates, e.g.:

```bash
python Agent/xlerobot_voice_agent.py --wakeword robot --wakeword-aliases "萝卜,罗伯特" --stt-backend vosk ...
```

Preemption (interrupt) behavior:

- While executing a command, the agent keeps listening.
- When it hears a new sentence that contains `robot`, it sends an emergency stop to the wheels and switches to the new command.
- `--wheel-step-seconds` controls how quickly a running movement can be interrupted (smaller = faster stop).

Linux example:

```bash
python Agent\xlerobot_voice_agent.py --right-arm-port COM3 --main-camera 0 --sounddevice-index 2 --wakeword robot --wakeword-aliases "萝卜,罗伯特" --stt-backend vosk --wheel-step-seconds 0.05
```

Example phrases:

- `robot move forward`
- `robot turn left`
- `robot turn right`
