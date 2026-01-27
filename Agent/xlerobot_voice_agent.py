import argparse
import base64
import os
import queue
import sys
import threading
import time
from typing import Any

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import pyaudio  # type: ignore
except Exception:  # pragma: no cover
    pyaudio = None  # type: ignore[assignment]

try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None  # type: ignore[assignment]

try:
    from langchain.chat_models import init_chat_model  # type: ignore
except Exception:  # pragma: no cover
    init_chat_model = None  # type: ignore[assignment]

try:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage  # type: ignore
except Exception:  # pragma: no cover
    HumanMessage = None  # type: ignore[assignment]
    SystemMessage = None  # type: ignore[assignment]
    ToolMessage = None  # type: ignore[assignment]

try:
    from robocrew.core.sound_receiver import SoundReceiver  # type: ignore
except Exception:  # pragma: no cover
    SoundReceiver = None  # type: ignore[assignment]

try:
    from robocrew.core.tools import finish_task  # type: ignore
except Exception:  # pragma: no cover
    finish_task = None  # type: ignore[assignment]

try:
    from robocrew.core.utils import horizontal_angle_grid  # type: ignore
except Exception:  # pragma: no cover
    horizontal_angle_grid = None  # type: ignore[assignment]

try:
    from robocrew.robots.XLeRobot.tools import (  # type: ignore
        create_move_forward,
        create_turn_left,
        create_turn_right,
    )
except Exception:  # pragma: no cover
    create_move_forward = None  # type: ignore[assignment]
    create_turn_left = None  # type: ignore[assignment]
    create_turn_right = None  # type: ignore[assignment]

try:
    from robocrew.robots.XLeRobot.wheel_controls import XLeRobotWheels  # type: ignore
except Exception:  # pragma: no cover
    XLeRobotWheels = None  # type: ignore[assignment]

try:
    from gemini_wakeword_listener import GeminiWakewordListener
except Exception:  # pragma: no cover
    GeminiWakewordListener = None  # type: ignore[assignment]

try:
    from vosk_wakeword_listener import VoskWakewordListener
except Exception:  # pragma: no cover
    VoskWakewordListener = None  # type: ignore[assignment]


def _load_dotenv_if_present() -> None:
    """Best-effort .env loader without external deps.

    - Only sets variables that are not already set in the environment.
    - Keeps parsing minimal: KEY=VALUE, ignores blank lines and # comments.
    """

    def load_file(path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'").strip('"')
                    if key and key not in os.environ:
                        os.environ[key] = value
        except FileNotFoundError:
            return
        except OSError:
            return

    # Prefer current working directory, then the script directory.
    load_file(os.path.join(os.getcwd(), ".env"))
    load_file(os.path.join(os.path.dirname(__file__), ".env"))


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError as exc:
        raise SystemExit(f"Invalid int in ${name}: {val!r}") from exc


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return float(default)
    try:
        return float(val)
    except ValueError as exc:
        raise SystemExit(f"Invalid float in ${name}: {val!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None or val == "":
        return bool(default)
    v = val.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise SystemExit(f"Invalid bool in ${name}: {val!r} (use true/false or 1/0)")


def _optional_list_audio_devices() -> None:
    """List audio devices.

    RoboCrew's built-in SoundReceiver uses PyAudio. Prefer listing PyAudio devices when available,
    otherwise fall back to sounddevice.
    """

    if pyaudio is not None:
        try:
            pa = pyaudio.PyAudio()
            try:
                for idx in range(pa.get_device_count()):
                    info = pa.get_device_info_by_index(idx)
                    name = info.get("name", "")
                    max_in = int(info.get("maxInputChannels", 0))
                    max_out = int(info.get("maxOutputChannels", 0))
                    rate = int(info.get("defaultSampleRate", 0))
                    print(f"[{idx:>2}] in={max_in:<2} out={max_out:<2} rate={rate:<6}  {name}")
            finally:
                pa.terminate()
            return
        except Exception:
            # If PyAudio fails, try sounddevice next.
            pass

    if sd is None:  # pragma: no cover
        raise SystemExit(
            "Could not import pyaudio or sounddevice. Install microphone dependencies to list audio devices."
        )

    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        in_ch = dev.get("max_input_channels", 0)
        out_ch = dev.get("max_output_channels", 0)
        name = dev.get("name", "")
        hostapi = dev.get("hostapi", "")
        print(f"[{idx:>2}] in={in_ch:<2} out={out_ch:<2} hostapi={hostapi}  {name}")


def _default_serial_port() -> str:
    # User's setup is on Windows (COM ports). Keep Linux default for portability.
    return "COM3" if os.name == "nt" else "/dev/ttyUSB0"


def _default_main_camera() -> str:
    # On Windows, OpenCV camera indexes are typically 0,1,2... (as strings here, parsed later).
    return "0" if os.name == "nt" else "/dev/video0"


def _parse_camera_device(value: str) -> object:
    # RoboCrew examples use "/dev/video0" (string). On Windows, OpenCV expects int index.
    v = value.strip()
    if v.isdigit():
        return int(v)
    return v


def _sanitize_task(text: str, wakeword: str) -> str:
    """Remove wakeword and trivial punctuation; return empty string if nothing left."""
    t = (text or "").strip()
    if not t:
        return ""
    lower = t.lower()
    w = (wakeword or "").strip().lower()
    if w:
        idx = lower.find(w)
        if idx != -1:
            t = t[idx + len(w) :]
    # Keep the file ASCII: common CJK punctuation is expressed via unicode escapes.
    strip_chars = " ,.!?\":;'\uFF0C\u3002\uFF01\uFF1F\uFF1A\uFF1B"
    return t.strip().strip(strip_chars).strip()


class _InterruptibleWheels:
    """Wrap XLeRobotWheels to support cooperative cancellation (preemption)."""

    def __init__(
        self,
        wheels: Any,
        *,
        cancel_event: threading.Event,
        io_lock: threading.Lock,
        step_s: float,
    ) -> None:
        self._wheels = wheels
        self._cancel = cancel_event
        self._io_lock = io_lock
        self._step_s = max(0.01, float(step_s))

    # --- Public helpers -------------------------------------------------

    def stop(self) -> None:
        with self._io_lock:
            self._stop_all_locked()

    # --- API expected by RoboCrew's XLeRobot tools ----------------------

    def go_forward(self, meters: float) -> Any:
        distance = float(meters)
        if distance < 0:
            return self.go_backward(-distance)
        if distance == 0:
            return {}
        return self._run_for("Up", self._distance_to_duration(distance))

    def go_backward(self, meters: float) -> Any:
        distance = float(meters)
        if distance < 0:
            return self.go_forward(-distance)
        if distance == 0:
            return {}
        return self._run_for("Down", self._distance_to_duration(distance))

    def turn_left(self, degrees: float) -> Any:
        angle = float(degrees)
        if angle < 0:
            return self.turn_right(-angle)
        if angle == 0:
            return {}
        return self._run_for("Left", self._angle_to_duration(angle))

    def turn_right(self, degrees: float) -> Any:
        angle = float(degrees)
        if angle < 0:
            return self.turn_left(-angle)
        if angle == 0:
            return {}
        return self._run_for("Right", self._angle_to_duration(angle))

    # --- Internals ------------------------------------------------------

    def _distance_to_duration(self, distance_m: float) -> float:
        # Prefer the wheel controller's configured speed if available.
        linear_mps = float(getattr(self._wheels, "linear_mps", 0.25))
        linear_mps = max(1e-6, linear_mps)
        return abs(float(distance_m)) / linear_mps

    def _angle_to_duration(self, degrees: float) -> float:
        angular_dps = float(getattr(self._wheels, "angular_dps", 100.0))
        angular_dps = max(1e-6, angular_dps)
        return abs(float(degrees)) / angular_dps

    def _apply_action_locked(self, action: str) -> Any:
        # Use RoboCrew implementation if present.
        fn = getattr(self._wheels, "_apply_action", None)
        if callable(fn):
            return fn(action)

        # Fallback: compute from config if available.
        cfg = getattr(self._wheels, "config", None)
        sdk = getattr(self._wheels, "sdk", None)
        if cfg is None or sdk is None:
            raise RuntimeError("Wheel controller missing config/sdk; cannot apply action.")
        payload = {wheel.id: wheel.speed_for(action, cfg.speed) for wheel in cfg.wheels}
        sdk.sync_write_wheel_speeds(payload)
        return payload

    def _stop_all_locked(self) -> Any:
        fn = getattr(self._wheels, "_stop_all", None)
        if callable(fn):
            return fn()

        cfg = getattr(self._wheels, "config", None)
        sdk = getattr(self._wheels, "sdk", None)
        if cfg is None or sdk is None:
            return {}
        payload = {wheel.id: 0 for wheel in cfg.wheels}
        sdk.sync_write_wheel_speeds(payload)
        return payload

    def _run_for(self, action: str, duration_s: float) -> Any:
        # Start motion.
        with self._io_lock:
            payload = self._apply_action_locked(action)

        # Keep "cooperatively" checking cancellation.
        remaining = max(0.0, float(duration_s))
        while remaining > 0.0 and not self._cancel.is_set():
            dt = self._step_s if remaining > self._step_s else remaining
            time.sleep(dt)
            remaining -= dt

        # Always stop.
        with self._io_lock:
            self._stop_all_locked()

        return payload


def main(argv: list[str]) -> int:
    _load_dotenv_if_present()

    parser = argparse.ArgumentParser(
        description=(
            "Voice-controlled RoboCrew agent for XLeRobot. "
            "The robot only reacts to sentences containing the wakeword (default: 'robot')."
        )
    )

    parser.add_argument(
        "--serial-port",
        "--right-arm-port",
        dest="serial_port",
        default=os.getenv("XLE_ROBOT_SERIAL_PORT", _default_serial_port()),
        help="Right arm serial port (arm connected to wheels/base), e.g. COM3 or /dev/ttyUSB0.",
    )
    parser.add_argument(
        "--main-camera-usb-port",
        "--main-camera",
        dest="main_camera_usb_port",
        default=os.getenv("XLE_ROBOT_MAIN_CAMERA_PORT", _default_main_camera()),
        help="Main camera (Windows: OpenCV index like 0; Linux: /dev/video0).",
    )
    parser.add_argument(
        "--camera-fov",
        type=float,
        default=float(os.getenv("XLE_ROBOT_CAMERA_FOV", "110")),
        help="Horizontal FOV of main camera in degrees (used for image overlay).",
    )
    parser.add_argument(
        "--left-arm-port",
        default=os.getenv("XLE_ROBOT_LEFT_ARM_PORT", "COM4" if os.name == "nt" else ""),
        help="Left arm serial port (optional; not used by this wheels-only voice agent).",
    )
    parser.add_argument(
        "--sounddevice-index",
        type=int,
        default=_env_int("XLE_ROBOT_SOUNDDEVICE_INDEX", 0),
        help="Microphone device index (see --list-audio-devices).",
    )
    parser.add_argument(
        "--wakeword",
        default=os.getenv("XLE_ROBOT_WAKEWORD", "robot"),
        help="Activation keyword required in your speech for the robot to act.",
    )
    parser.add_argument(
        "--wakeword-aliases",
        default=os.getenv("XLE_ROBOT_WAKEWORD_ALIASES", ""),
        help=(
            "Comma-separated additional wakewords that also activate the robot. "
            "Useful when offline STT transliterates the wakeword (example: robot,萝卜)."
        ),
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=_env_int("XLE_ROBOT_HISTORY_LEN", 4),
        help="How many recent movements the agent keeps in memory.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("XLE_ROBOT_LLM_MODEL", "google_genai:gemini-3-flash-preview"),
        help="LangChain model name, e.g. google_genai:gemini-3-flash-preview",
    )
    parser.add_argument(
        "--stt-backend",
        choices=["vosk", "gemini", "robocrew_openai"],
        default=os.getenv("XLE_ROBOT_STT_BACKEND", "vosk"),
        help="Speech-to-text backend used for microphone listening (default: vosk).",
    )
    parser.add_argument(
        "--stt-model",
        default=os.getenv("XLE_ROBOT_STT_MODEL", "gemini-3-flash-preview"),
        help="Gemini model id for STT when --stt-backend=gemini (example: gemini-3-flash-preview).",
    )
    parser.add_argument(
        "--stt-model-path",
        default=os.getenv(
            "XLE_ROBOT_STT_MODEL_PATH",
            os.path.join(os.path.dirname(__file__), "models", "vosk-model-small-cn-0.22"),
        ),
        help="Vosk model directory when --stt-backend=vosk.",
    )
    parser.add_argument(
        "--stt-sample-rate",
        type=int,
        default=_env_int("XLE_ROBOT_STT_SAMPLE_RATE", 16000),
        help="Microphone sample rate for Vosk (try 16000).",
    )
    parser.add_argument(
        "--stt-print-partials",
        action="store_true",
        default=_env_bool("XLE_ROBOT_STT_PRINT_PARTIALS", False),
        help="Print partial STT results to terminal (debug).",
    )
    parser.add_argument(
        "--stt-timeout",
        type=int,
        default=_env_int("XLE_ROBOT_STT_TIMEOUT", 60),
        help="STT request timeout in seconds.",
    )
    parser.add_argument(
        "--stt-rms-threshold",
        type=float,
        default=_env_float("XLE_ROBOT_STT_RMS_THRESHOLD", 400.0),
        help="Mic VAD threshold (bigger = less sensitive).",
    )
    parser.add_argument(
        "--stt-silence-seconds",
        type=float,
        default=_env_float("XLE_ROBOT_STT_SILENCE_SECONDS", 0.7),
        help="Silence duration that ends a speech segment (smaller = faster).",
    )
    parser.add_argument(
        "--stt-min-record-seconds",
        type=float,
        default=_env_float("XLE_ROBOT_STT_MIN_RECORD_SECONDS", 0.6),
        help="Minimum segment duration to send to STT (avoid wheel noise).",
    )
    parser.add_argument(
        "--stt-pre-roll-seconds",
        type=float,
        default=_env_float("XLE_ROBOT_STT_PRE_ROLL_SECONDS", 0.5),
        help="Audio pre-roll to include before detected speech.",
    )
    parser.add_argument(
        "--stt-max-record-seconds",
        type=float,
        default=_env_float("XLE_ROBOT_STT_MAX_RECORD_SECONDS", 8.0),
        help="Maximum segment duration to upload to STT.",
    )
    parser.add_argument(
        "--stt-frames-per-buffer",
        type=int,
        default=_env_int("XLE_ROBOT_STT_FRAMES_PER_BUFFER", 2048),
        help="PyAudio frames_per_buffer (affects latency and CPU usage).",
    )
    parser.add_argument(
        "--stt-min-request-interval",
        type=float,
        default=_env_float("XLE_ROBOT_STT_MIN_REQUEST_INTERVAL", 13.0),
        help="Minimum seconds between STT requests (helps avoid rate limits).",
    )
    stt_seg_group = parser.add_mutually_exclusive_group()
    stt_seg_group.add_argument(
        "--stt-drop-old-segments",
        dest="stt_drop_old_segments",
        action="store_true",
        help="Drop older pending audio segments and keep only the newest (reduces backlog).",
    )
    stt_seg_group.add_argument(
        "--stt-keep-old-segments",
        dest="stt_drop_old_segments",
        action="store_false",
        help="Keep all pending audio segments (may increase STT calls).",
    )
    parser.set_defaults(stt_drop_old_segments=_env_bool("XLE_ROBOT_STT_DROP_OLD_SEGMENTS", True))
    parser.add_argument(
        "--stt-voice-frames-needed",
        type=int,
        default=_env_int("XLE_ROBOT_STT_VOICE_FRAMES_NEEDED", 2),
        help="How many consecutive loud frames are required to start recording a segment.",
    )
    parser.add_argument(
        "--stt-cooldown-seconds",
        type=float,
        default=_env_float("XLE_ROBOT_STT_COOLDOWN_SECONDS", 0.5),
        help="Cooldown after finishing a segment before starting a new one.",
    )
    parser.add_argument(
        "--wheel-step-seconds",
        type=float,
        default=float(os.getenv("XLE_ROBOT_WHEEL_STEP_SECONDS", "0.1")),
        help="Motion interrupt check period in seconds (smaller = faster stop on new command).",
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List audio devices and exit.",
    )

    args = parser.parse_args(argv)

    if args.list_audio_devices:
        _optional_list_audio_devices()
        return 0

    if args.model.startswith("google_genai:") and not (
        os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    ):
        raise SystemExit(
            "Missing GOOGLE_API_KEY. Set it as an environment variable (recommended) "
            "or put it into a local .env file."
        )

    if args.stt_backend == "gemini" and not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise SystemExit("Missing GOOGLE_API_KEY for Gemini speech-to-text.")
    if args.stt_backend == "vosk":
        if VoskWakewordListener is None:  # pragma: no cover
            raise SystemExit("Missing Vosk STT. Install with: pip install vosk")
        if pyaudio is None:  # pragma: no cover
            raise SystemExit("Missing PyAudio. Install it to use microphone input.")
        if not args.stt_model_path or not os.path.isdir(args.stt_model_path):
            raise SystemExit(
                "Missing Vosk model directory. Download a Vosk model and set XLE_ROBOT_STT_MODEL_PATH "
                "or pass --stt-model-path."
            )
    if args.stt_backend == "robocrew_openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY for RoboCrew OpenAI speech-to-text.")

    if cv2 is None:  # pragma: no cover
        raise SystemExit("Missing OpenCV (cv2). Install robocrew/opencv-python first.")
    if init_chat_model is None or HumanMessage is None or SystemMessage is None or ToolMessage is None:  # pragma: no cover
        raise SystemExit("Missing LangChain dependencies. Install robocrew first.")
    if finish_task is None or XLeRobotWheels is None:  # pragma: no cover
        raise SystemExit("Missing RoboCrew XLeRobot dependencies. Install robocrew first.")
    if create_move_forward is None or create_turn_left is None or create_turn_right is None:  # pragma: no cover
        raise SystemExit("Missing RoboCrew XLeRobot tools. Install robocrew first.")

    camera_device = _parse_camera_device(args.main_camera_usb_port)
    if isinstance(camera_device, int) and camera_device < 0:
        raise SystemExit("--main-camera must be >= 0 on Windows.")

    print(
        "Config: "
        f"model={args.model!r}, "
        f"wakeword={args.wakeword!r}, "
        f"right_arm_port={args.serial_port!r}, "
        f"main_camera={camera_device!r}, "
        f"mic_index={args.sounddevice_index}, "
        f"stt_backend={args.stt_backend!r}"
    )
    wakeword_aliases = [w.strip() for w in str(getattr(args, "wakeword_aliases", "")).split(",") if w.strip()]
    if wakeword_aliases:
        print(f"Wakeword aliases: {wakeword_aliases}", flush=True)

    # Wheels
    sdk = XLeRobotWheels.connect_serial(args.serial_port)
    wheel_controller = XLeRobotWheels(sdk)
    wheel_io_lock = threading.Lock()
    cancel_event = threading.Event()
    interruptible_wheels = _InterruptibleWheels(
        wheel_controller, cancel_event=cancel_event, io_lock=wheel_io_lock, step_s=args.wheel_step_seconds
    )

    # Movement tools
    move_forward = create_move_forward(interruptible_wheels)
    turn_left = create_turn_left(interruptible_wheels)
    turn_right = create_turn_right(interruptible_wheels)
    tools = [move_forward, turn_left, turn_right, finish_task]
    tool_name_to_tool = {tool.name: tool for tool in tools}

    # Main camera (central camera only).
    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera {camera_device!r}.")
    stt_listener: Any | None = None
    try:
        # Reduce latency if backend supports it.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        def capture_image_bytes() -> bytes:
            cap.grab()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read from camera.")
            if horizontal_angle_grid is not None:
                frame = horizontal_angle_grid(frame, h_fov=args.camera_fov)
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                raise RuntimeError("Failed to encode camera frame.")
            return buffer.tobytes()

        # Voice receiver -> raw queue (contains wakeword already in RoboCrew implementation).
        raw_task_queue: "queue.Queue[str]" = queue.Queue()
        command_queue: "queue.Queue[str]" = queue.Queue()

        def stop_now() -> None:
            cancel_event.set()
            try:
                interruptible_wheels.stop()
            except Exception:
                # Best-effort emergency stop.
                pass

        def task_router_loop() -> None:
            # Whenever a new wakeword+command arrives, stop current motion and keep only the latest command.
            while True:
                raw = raw_task_queue.get()
                cmd = _sanitize_task(raw, args.wakeword)
                if not cmd:
                    # Wakeword without a command -> treat as "stop".
                    print(f"[VOICE] {raw}", flush=True)
                    print("[CMD] (wakeword only) -> STOP", flush=True)
                    stop_now()
                    continue
                print(f"[VOICE] {raw}", flush=True)
                print(f"[CMD] {cmd}", flush=True)
                stop_now()
                # Keep latest only.
                try:
                    while True:
                        command_queue.get_nowait()
                except queue.Empty:
                    pass
                command_queue.put(cmd)

        # Start listening (keeps running while the robot moves).
        try:
            if args.stt_backend == "vosk":
                if VoskWakewordListener is None:  # pragma: no cover
                    raise RuntimeError("VoskWakewordListener import failed.")
                stt_listener = VoskWakewordListener(
                    device_index=args.sounddevice_index,
                    task_queue=raw_task_queue,
                    wakeword=args.wakeword,
                    wakeword_aliases=wakeword_aliases,
                    model_path=args.stt_model_path,
                    sample_rate=args.stt_sample_rate,
                    frames_per_buffer=args.stt_frames_per_buffer,
                    print_partials=args.stt_print_partials,
                )
            elif args.stt_backend == "gemini":
                if GeminiWakewordListener is None:  # pragma: no cover
                    raise RuntimeError("GeminiWakewordListener import failed.")
                if pyaudio is None:  # pragma: no cover
                    raise RuntimeError("PyAudio is required for STT.")
                stt_listener = GeminiWakewordListener(
                    device_index=args.sounddevice_index,
                    task_queue=raw_task_queue,
                    wakeword=args.wakeword,
                    wakeword_aliases=wakeword_aliases,
                    model=args.stt_model,
                    timeout_s=args.stt_timeout,
                    min_request_interval_s=args.stt_min_request_interval,
                    drop_old_segments=args.stt_drop_old_segments,
                    rms_threshold=args.stt_rms_threshold,
                    silence_seconds=args.stt_silence_seconds,
                    min_record_seconds=args.stt_min_record_seconds,
                    pre_roll_seconds=args.stt_pre_roll_seconds,
                    max_record_seconds=args.stt_max_record_seconds,
                    frames_per_buffer=args.stt_frames_per_buffer,
                    voice_frames_needed=args.stt_voice_frames_needed,
                    cooldown_seconds=args.stt_cooldown_seconds,
                )
            else:
                # RoboCrew SoundReceiver transcribes via OpenAI (gpt-4o-transcribe).
                if SoundReceiver is None:  # pragma: no cover
                    raise RuntimeError("RoboCrew SoundReceiver import failed.")
                stt_listener = SoundReceiver(args.sounddevice_index, raw_task_queue, args.wakeword)
        except Exception as exc:  # pragma: no cover
            if args.stt_backend == "vosk":
                raise SystemExit("Failed to start Vosk STT listener. Check model path and mic settings.") from exc
            if args.stt_backend == "gemini":
                raise SystemExit("Failed to start Gemini STT listener. Check GOOGLE_API_KEY and mic index.") from exc
            raise SystemExit("Failed to start RoboCrew OpenAI STT listener. Check OPENAI_API_KEY and mic index.") from exc

        threading.Thread(target=task_router_loop, daemon=True).start()

        # LLM init
        base_system_prompt = (
            "You are a voice-controlled mobile robot.\n"
            "You control the base using tools:\n"
            "- move_forward(distance_meters)\n"
            "- turn_left(angle_degrees)\n"
            "- turn_right(angle_degrees)\n"
            "- finish_task()\n\n"
            "Rules:\n"
            "- Execute the user's latest command.\n"
            "- For simple commands, call exactly one movement tool (if needed) then finish_task.\n"
            "- If the user says to stop, do not move; just finish_task.\n"
        )
        system_message = SystemMessage(content=base_system_prompt)
        message_history: list[Any] = [system_message]

        llm = init_chat_model(args.model)
        try:
            llm = llm.bind_tools(tools, parallel_tool_calls=False)
        except TypeError:
            llm = llm.bind_tools(tools)

        def cut_off_context(nr_of_loops: int) -> None:
            human_indices = [i for i, msg in enumerate(message_history) if getattr(msg, "type", "") == "human"]
            if len(human_indices) >= nr_of_loops:
                start_index = human_indices[-nr_of_loops]
                del message_history[1:start_index]

        def invoke_tool(tool_call: dict[str, Any]) -> Any:
            requested_tool = tool_name_to_tool[tool_call["name"]]
            tool_output = requested_tool.invoke(tool_call.get("args", {}))
            return ToolMessage(str(tool_output), tool_call_id=tool_call["id"])

        print(
            "Starting preemptible voice agent. "
            f"Say '{args.wakeword} ...' to start, and say '{args.wakeword} ...' again to interrupt and override."
        )

        current_task = None
        while True:
            # Block until we have a task.
            if current_task is None:
                current_task = command_queue.get()
                cancel_event.clear()
                # New command -> reset context so we don't mix tasks.
                message_history = [system_message]

            # If an override arrived, switch immediately.
            if not command_queue.empty():
                current_task = command_queue.get()
                cancel_event.clear()
                message_history = [system_message]

            # Fast path: stop commands without bothering the LLM.
            if current_task.strip().lower() in {
                "stop",
                "pause",
                "halt",
                "\u505c\u6b62",  # "stop" (zh)
                "\u505c\u4e0b",  # "stop" (zh)
                "\u5239\u8f66",  # "brake" (zh)
            }:
                interruptible_wheels.stop()
                current_task = None
                continue

            # Vision + task prompt.
            try:
                image_bytes = capture_image_bytes()
            except Exception:
                # If camera fails temporarily, still allow non-vision commands to execute.
                image_bytes = b""

            if cancel_event.is_set() or not command_queue.empty():
                # A new command arrived while capturing; restart loop.
                continue

            if image_bytes:
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": "Main camera view:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": f"Your task is: '{current_task}'"},
                    ]
                )
            else:
                message = HumanMessage(content=f"Your task is: '{current_task}'")

            message_history.append(message)
            response = llm.invoke(message_history)
            message_history.append(response)
            if args.history_len:
                cut_off_context(args.history_len)

            # If interrupted while the model was thinking, ignore this response.
            if cancel_event.is_set() or not command_queue.empty():
                continue

            # Execute tool calls; allow preemption between calls.
            finished = False
            for tool_call in getattr(response, "tool_calls", []):
                if cancel_event.is_set() or not command_queue.empty():
                    break
                tool_response = invoke_tool(tool_call)
                message_history.append(tool_response)
                if tool_call.get("name") == "finish_task":
                    finished = True
                    break

            if finished:
                current_task = None

    finally:
        try:
            if stt_listener is not None and hasattr(stt_listener, "stop"):
                stt_listener.stop()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        try:
            interruptible_wheels.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
