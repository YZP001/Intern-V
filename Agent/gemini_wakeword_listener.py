from __future__ import annotations

import audioop
import base64
import io
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
import wave
from collections import deque
from typing import Optional

try:
    import pyaudio  # type: ignore
except Exception:  # pragma: no cover
    pyaudio = None  # type: ignore[assignment]


class GeminiSttHttpError(RuntimeError):
    def __init__(self, code: int, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.code = int(code)
        self.retry_after_s = retry_after_s


def _parse_retry_delay_seconds(error_json: dict) -> float | None:
    # Expected structure from Google APIs: error.details includes google.rpc.RetryInfo
    try:
        details = error_json.get("error", {}).get("details", [])
        for item in details:
            if not isinstance(item, dict):
                continue
            if item.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                raw = str(item.get("retryDelay", "")).strip()
                # Common format: "7s"
                m = re.match(r"^([0-9]+(?:\\.[0-9]+)?)s$", raw)
                if m:
                    return float(m.group(1))
    except Exception:
        return None
    return None


def _get_google_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY).")
    return key


def _gemini_transcribe_wav_bytes(*, wav_bytes: bytes, model: str, timeout_s: int) -> str:
    # Gemini REST API (Generative Language) - avoid printing URL because it contains the API key.
    api_key = _get_google_api_key()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Transcribe the user's speech from the provided audio.\n"
                            "Return ONLY the transcription text (no markdown, no extra words)."
                        )
                    },
                    {"inlineData": {"mimeType": "audio/wav", "data": wav_b64}},
                ],
            }
        ]
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        try:
            err_json = json.loads(details)
        except Exception:
            err_json = {}
        retry_after_s = _parse_retry_delay_seconds(err_json) if isinstance(err_json, dict) else None
        raise GeminiSttHttpError(int(getattr(e, "code", 0) or 0), details, retry_after_s) from e
    except urllib.error.URLError as e:
        raise GeminiSttHttpError(0, f"Gemini STT request failed: {getattr(e, 'reason', e)!s}") from e

    resp_json = json.loads(payload.decode("utf-8", errors="replace"))
    try:
        text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""
    return str(text).strip()


class GeminiWakewordListener:
    """Continuously listens to the microphone and pushes transcripts containing a wakeword to a queue.

    This is a lightweight replacement for RoboCrew's SoundReceiver when you want to use Google/Gemini
    for speech-to-text instead of OpenAI.
    """

    def __init__(
        self,
        *,
        device_index: int,
        task_queue: "queue.Queue[str]",
        wakeword: str = "robot",
        wakeword_aliases: Optional[list[str]] = None,
        model: str = "gemini-3-flash-preview",
        timeout_s: int = 60,
        min_request_interval_s: float = 13.0,
        drop_old_segments: bool = True,
        rms_threshold: float = 400.0,
        silence_seconds: float = 0.7,
        min_record_seconds: float = 0.6,
        pre_roll_seconds: float = 0.5,
        max_record_seconds: float = 8.0,
        frames_per_buffer: int = 2048,
        voice_frames_needed: int = 2,
        cooldown_seconds: float = 0.5,
    ) -> None:
        self._task_queue = task_queue
        self._wakeword = (wakeword or "").strip().lower()
        self._wakeword_aliases = [w.strip().lower() for w in (wakeword_aliases or []) if w and w.strip()]
        self._wakewords = [w for w in [self._wakeword, *self._wakeword_aliases] if w]
        self._model = model
        self._timeout_s = int(timeout_s)
        self._min_request_interval_s = max(0.0, float(min_request_interval_s))
        self._drop_old_segments = bool(drop_old_segments)

        self._rms_threshold = float(rms_threshold)
        self._silence_seconds = max(0.1, float(silence_seconds))
        self._min_record_seconds = max(0.1, float(min_record_seconds))
        self._pre_roll_seconds = max(0.0, float(pre_roll_seconds))
        self._max_record_seconds = max(self._min_record_seconds, float(max_record_seconds))
        self._frames_per_buffer = int(frames_per_buffer)
        self._voice_frames_needed = max(1, int(voice_frames_needed))
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._cooldown_until = 0.0
        self._next_request_time = 0.0
        self._last_rate_limit_log = 0.0

        # PyAudio init
        if pyaudio is None:  # pragma: no cover
            raise RuntimeError("PyAudio is required for microphone input.")

        self._pyaudio = pyaudio
        self._pa = pyaudio.PyAudio()
        self._format = pyaudio.paInt16
        self._channels = 1
        self._device_index = int(device_index)
        info = self._pa.get_device_info_by_index(self._device_index)
        default_rate = int(info.get("defaultSampleRate", 48000))
        self._rate = default_rate if default_rate > 0 else 48000
        self._sample_width = self._pa.get_sample_size(self._format)
        self._bytes_per_second = int(self._rate * self._channels * self._sample_width)

        self._stop_event = threading.Event()
        self._segment_queue: "queue.Queue[bytes]" = queue.Queue()

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._worker_thread = threading.Thread(target=self._transcribe_loop, daemon=True)

        self._stream = None
        self._open_stream()
        self._reader_thread.start()
        self._worker_thread.start()

    def _open_stream(self) -> None:
        try:
            self._stream = self._pa.open(
                format=self._format,
                channels=self._channels,
                rate=self._rate,
                input=True,
                input_device_index=self._device_index,
                frames_per_buffer=self._frames_per_buffer,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to open microphone device index {self._device_index}: {exc}") from exc

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
        finally:
            try:
                self._pa.terminate()
            except Exception:
                pass

    def _reader_loop(self) -> None:
        # Basic VAD based on RMS threshold.
        pre_roll_maxlen = max(1, int(self._pre_roll_seconds * self._rate / self._frames_per_buffer))
        pre_roll: deque[bytes] = deque(maxlen=pre_roll_maxlen)

        recording = False
        recorded: list[bytes] = []
        record_start = 0.0
        silence_start: Optional[float] = None
        voice_frames = 0

        assert self._stream is not None
        while not self._stop_event.is_set():
            try:
                data = self._stream.read(self._frames_per_buffer, exception_on_overflow=False)
            except Exception:
                time.sleep(0.05)
                continue

            pre_roll.append(data)
            rms = float(audioop.rms(data, self._sample_width)) if data else 0.0
            now = time.time()

            if not recording:
                if now < self._cooldown_until:
                    continue
                if rms >= self._rms_threshold:
                    voice_frames += 1
                    if voice_frames >= self._voice_frames_needed:
                        recording = True
                        record_start = now
                        silence_start = None
                        recorded = list(pre_roll)
                        voice_frames = 0
                else:
                    voice_frames = 0
            else:
                recorded.append(data)

                # Max duration cap (avoid huge uploads if mic is noisy).
                if now - record_start >= self._max_record_seconds:
                    recording = False
                    silence_start = None
                    self._enqueue_segment(recorded)
                    self._cooldown_until = time.time() + self._cooldown_seconds
                    recorded = []
                    continue

                if rms < self._rms_threshold:
                    if silence_start is None:
                        silence_start = now
                    elif now - silence_start >= self._silence_seconds:
                        recording = False
                        silence_start = None
                        self._enqueue_segment(recorded)
                        self._cooldown_until = time.time() + self._cooldown_seconds
                        recorded = []
                else:
                    silence_start = None

    def _enqueue_segment(self, frames: list[bytes]) -> None:
        audio_data = b"".join(frames).strip(b"\x00")
        if not audio_data:
            return
        duration_s = len(audio_data) / float(self._bytes_per_second)
        if duration_s < self._min_record_seconds:
            return

        # PCM -> WAV in memory
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(self._sample_width)
            wf.setframerate(self._rate)
            wf.writeframes(audio_data)
        self._segment_queue.put(buf.getvalue())

    def _transcribe_loop(self) -> None:
        pending_wav: Optional[bytes] = None
        while not self._stop_event.is_set():
            if pending_wav is None:
                try:
                    pending_wav = self._segment_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

            # Optionally keep only the newest segment (avoids backlog and reduces pointless calls).
            if self._drop_old_segments:
                try:
                    while True:
                        pending_wav = self._segment_queue.get_nowait()
                except queue.Empty:
                    pass

            now_mono = time.monotonic()
            if now_mono < self._next_request_time:
                time.sleep(min(0.2, self._next_request_time - now_mono))
                continue

            try:
                text = _gemini_transcribe_wav_bytes(
                    wav_bytes=pending_wav, model=self._model, timeout_s=self._timeout_s
                )
            except GeminiSttHttpError as exc:
                # Rate limit: wait and retry later (keep the latest pending segment).
                if exc.code == 429:
                    retry_after_s = float(exc.retry_after_s or 0.0)
                    if retry_after_s <= 0.0:
                        retry_after_s = max(1.0, self._min_request_interval_s)
                    now = time.monotonic()
                    if now - self._last_rate_limit_log > 5.0:
                        print(f"[STT] rate limited (429), waiting {retry_after_s:.1f}s")
                        self._last_rate_limit_log = now
                    self._next_request_time = time.monotonic() + retry_after_s
                    continue
                # Other errors: drop this segment but keep listener alive.
                print(f"[STT] transcription failed: {exc}")
                pending_wav = None
                continue
            except Exception as exc:
                print(f"[STT] transcription failed: {exc}")
                pending_wav = None
                continue

            if not text:
                pending_wav = None
                continue

            text_lower = text.lower()
            matched = None
            for w in self._wakewords:
                if w and w in text_lower:
                    matched = w
                    break
            if matched is not None:
                # Normalize alias back to canonical wakeword so downstream logic can strip it.
                if matched != self._wakeword and self._wakeword:
                    idx = text_lower.find(matched)
                    if idx != -1:
                        text = text[:idx] + self._wakeword + text[idx + len(matched) :]
                self._task_queue.put(text)
            self._next_request_time = time.monotonic() + self._min_request_interval_s
            pending_wav = None
