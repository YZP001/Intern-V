from __future__ import annotations

import audioop
import json
import os
import queue
import threading
import time
from typing import Optional

try:
    import pyaudio  # type: ignore
except Exception:  # pragma: no cover
    pyaudio = None  # type: ignore[assignment]

try:
    from vosk import KaldiRecognizer, Model, SetLogLevel  # type: ignore
except Exception:  # pragma: no cover
    KaldiRecognizer = None  # type: ignore[assignment]
    Model = None  # type: ignore[assignment]
    SetLogLevel = None  # type: ignore[assignment]


class VoskWakewordListener:
    """Offline speech-to-text using Vosk.

    Streams microphone audio into Vosk and pushes recognized phrases that contain `wakeword`
    into `task_queue`.
    """

    def __init__(
        self,
        *,
        device_index: int,
        task_queue: "queue.Queue[str]",
        wakeword: str = "robot",
        wakeword_aliases: Optional[list[str]] = None,
        model_path: str,
        sample_rate: int = 16000,
        frames_per_buffer: int = 4000,
        print_partials: bool = False,
        print_finals: bool = True,
        partial_throttle_s: float = 0.25,
    ) -> None:
        if pyaudio is None:  # pragma: no cover
            raise RuntimeError("PyAudio is required for offline STT.")
        if Model is None or KaldiRecognizer is None:  # pragma: no cover
            raise RuntimeError("vosk is required for offline STT (pip install vosk).")
        if not model_path or not os.path.isdir(model_path):
            raise RuntimeError(
                "Missing Vosk model. Set --stt-model-path (directory) or XLE_ROBOT_STT_MODEL_PATH."
            )

        # Reduce Vosk internal logging.
        try:
            if SetLogLevel is not None:
                SetLogLevel(-1)
        except Exception:
            pass

        self._task_queue = task_queue
        self._wakeword = (wakeword or "").strip().lower()
        self._wakeword_aliases = [w.strip().lower() for w in (wakeword_aliases or []) if w and w.strip()]
        # Match against wakeword + any aliases (useful when STT model transliterates).
        self._wakewords = [w for w in [self._wakeword, *self._wakeword_aliases] if w]
        self._print_partials = bool(print_partials)
        self._print_finals = bool(print_finals)
        self._partial_throttle_s = max(0.05, float(partial_throttle_s))

        self._pa = pyaudio.PyAudio()
        self._format = pyaudio.paInt16
        self._channels = 1
        self._device_index = int(device_index)
        # Vosk models are typically trained at 16kHz. Keep model/sample rate separate from device capture rate.
        self._model_rate = int(sample_rate)
        self._capture_rate = int(sample_rate)
        self._frames_per_buffer = int(frames_per_buffer)
        self._sample_width = self._pa.get_sample_size(self._format)
        self._resample_state = None

        # Open stream first; if the requested capture rate is unsupported, fall back to device default.
        self._stream = self._open_stream()

        self._model = Model(model_path)
        self._rec = KaldiRecognizer(self._model, float(self._model_rate))
        try:
            self._rec.SetWords(True)
        except Exception:
            pass

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

        self._stream.start_stream()
        self._thread.start()

    def _open_stream(self):
        def try_open(rate: int):
            return self._pa.open(
                format=self._format,
                channels=self._channels,
                rate=rate,
                input=True,
                input_device_index=self._device_index,
                frames_per_buffer=self._frames_per_buffer,
            )

        try:
            self._capture_rate = int(self._model_rate)
            return try_open(self._capture_rate)
        except Exception:
            try:
                info = self._pa.get_device_info_by_index(self._device_index)
                fallback_rate = int(float(info.get("defaultSampleRate", self._model_rate)))
            except Exception:
                fallback_rate = int(self._model_rate)

            if fallback_rate != self._model_rate:
                try:
                    stream = try_open(fallback_rate)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to open microphone device index {self._device_index} at {self._model_rate} Hz "
                        f"(and fallback {fallback_rate} Hz): {exc}"
                    ) from exc
                self._capture_rate = int(fallback_rate)
                return stream

            raise RuntimeError(
                f"Failed to open microphone device index {self._device_index} at {self._model_rate} Hz."
            )

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                except Exception:
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
        finally:
            try:
                self._pa.terminate()
            except Exception:
                pass

    def _loop(self) -> None:
        last_partial = ""
        next_partial_print = 0.0

        if self._print_finals or self._print_partials:
            if self._capture_rate != self._model_rate:
                print(
                    f"[STT] Vosk listening (device_rate={self._capture_rate}Hz -> model_rate={self._model_rate}Hz)",
                    flush=True,
                )
            else:
                print(f"[STT] Vosk listening (rate={self._model_rate}Hz)", flush=True)

        while not self._stop_event.is_set():
            try:
                data = self._stream.read(self._frames_per_buffer, exception_on_overflow=False)
            except Exception:
                time.sleep(0.05)
                continue

            if not data:
                continue

            if self._capture_rate != self._model_rate:
                try:
                    data, self._resample_state = audioop.ratecv(
                        data,
                        self._sample_width,
                        self._channels,
                        self._capture_rate,
                        self._model_rate,
                        self._resample_state,
                    )
                except Exception:
                    # If resampling fails, skip this chunk but keep the listener alive.
                    continue

            if self._rec.AcceptWaveform(data):
                try:
                    result = json.loads(self._rec.Result())
                except Exception:
                    result = {}
                text = str(result.get("text", "")).strip()
                if not text:
                    continue

                if self._print_finals:
                    print(f"[STT] {text}", flush=True)

                text_lower = text.lower()
                matched = None
                for w in self._wakewords:
                    if w and w in text_lower:
                        matched = w
                        break
                if matched is not None:
                    # Normalize any alias back to the canonical wakeword so downstream logic can strip it.
                    if matched != self._wakeword and self._wakeword:
                        idx = text_lower.find(matched)
                        if idx != -1:
                            text = text[:idx] + self._wakeword + text[idx + len(matched) :]
                    self._task_queue.put(text)
            elif self._print_partials:
                now = time.monotonic()
                if now < next_partial_print:
                    continue
                next_partial_print = now + self._partial_throttle_s
                try:
                    partial = json.loads(self._rec.PartialResult()).get("partial", "")
                except Exception:
                    partial = ""
                partial = str(partial).strip()
                if partial and partial != last_partial:
                    last_partial = partial
                    print(f"[STT] {partial}", flush=True)
