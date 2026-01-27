"""
Minimal demo: call Gemini API model `gemini-3-flash-preview`.

Security:
- Do NOT hardcode API keys in code.
- Set env var `GOOGLE_API_KEY` (recommended) or `GEMINI_API_KEY` instead.

Examples:
  # PowerShell
  $env:GOOGLE_API_KEY="YOUR_KEY"
  python Agent/gemini_3_flash_preview_demo.py "Say hello in Chinese."

  # With system instruction
  python Agent/gemini_3_flash_preview_demo.py "Write a 1-sentence summary." --system "Be concise."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

try:
    from google import genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore[assignment]

try:
    from google.genai import types as genai_types  # type: ignore
except Exception:  # pragma: no cover
    genai_types = None  # type: ignore[assignment]


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

    load_file(os.path.join(os.getcwd(), ".env"))
    load_file(os.path.join(os.path.dirname(__file__), ".env"))


def _get_api_key() -> str:
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        val = os.getenv(name)
        if val:
            return val
    raise SystemExit("Missing API key. Set env var GOOGLE_API_KEY (or GEMINI_API_KEY).")


def _call_with_google_genai_sdk(*, model: str, prompt: str, system: str | None) -> str | None:
    """Preferred path if `google-genai` is installed. Returns None if not available."""
    if genai is None:
        return None

    client = genai.Client(api_key=_get_api_key())

    # The SDK supports passing config; keep it optional to minimize assumptions about versions.
    if system:
        try:
            if genai_types is None:
                raise RuntimeError("google.genai.types is not available")
            cfg = genai_types.GenerateContentConfig(system_instruction=system)
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        except Exception:
            # Fallback if types/config path changed in the installed SDK version.
            resp = client.models.generate_content(model=model, contents=prompt)
    else:
        resp = client.models.generate_content(model=model, contents=prompt)

    text = getattr(resp, "text", None)
    return text if isinstance(text, str) and text else str(resp)


def _call_with_rest(*, model: str, prompt: str, system: str | None, timeout_s: int) -> str:
    api_key = _get_api_key()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body: dict[str, object] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }
    if system:
        # REST API uses proto-JSON (camelCase).
        body["systemInstruction"] = {"parts": [{"text": system}]}

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as e:
        # Do not print the URL (it contains the API key).
        details = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise SystemExit(f"Gemini API HTTP error: {e.code}\n{details}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Gemini API request failed: {getattr(e, 'reason', e)!s}") from e
    except Exception as e:  # pragma: no cover
        # Avoid leaking the URL (it contains the API key). Keep the message generic.
        raise SystemExit(f"Gemini API request failed: {e.__class__.__name__}") from e

    try:
        resp_json = json.loads(payload.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return payload.decode("utf-8", errors="replace")

    # Extract the primary text if present; otherwise return the full JSON.
    try:
        return resp_json["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return json.dumps(resp_json, ensure_ascii=False, indent=2)


def main(argv: list[str]) -> int:
    _load_dotenv_if_present()

    parser = argparse.ArgumentParser(description="Call Gemini model gemini-3-flash-preview.")
    parser.add_argument("prompt", help="User prompt text.")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model id.")
    parser.add_argument("--system", default=None, help="Optional system instruction.")
    parser.add_argument(
        "--force-rest",
        action="store_true",
        help="Force using REST (urllib) even if the google-genai SDK is installed.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="REST timeout (seconds).")
    args = parser.parse_args(argv)

    if not args.force_rest:
        text = _call_with_google_genai_sdk(model=args.model, prompt=args.prompt, system=args.system)
        if text is not None:
            print(text)
            return 0

    print(_call_with_rest(model=args.model, prompt=args.prompt, system=args.system, timeout_s=args.timeout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
