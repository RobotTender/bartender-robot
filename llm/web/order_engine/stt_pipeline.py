import json
import mimetypes
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

STT_MODEL = "gemini-flash-latest"
MAX_RETRIES = 3

PROMPT = """You are an audio order analyzer for a Korean bartender app.
Analyze the provided audio and return strict JSON only with these keys:
{
  "transcript": "recognized Korean text",
  "emotion": "happy|sad|angry|tired|neutral",
  "selected_menu": "soju|beer|somaek",
  "reason": "short reason in Korean"
}
Rules:
- Output must be valid JSON only. No markdown.
- If speech is unclear, set transcript to an empty string and emotion to neutral.
- Keep reason concise (max 1 sentence).
"""


@dataclass
class STTResult:
    text: str
    emotion: str = ""
    selected_menu: str = ""
    reason: str = ""


def _guess_mime_type(filename: str | None) -> str:
    guessed, _ = mimetypes.guess_type(filename or "")
    return guessed or "audio/webm"


def _extract_json(text: str) -> dict:
    content = (text or "").strip()
    if content.startswith("```json"):
        content = content.replace("```json", "", 1).replace("```", "", 1).strip()
    elif content.startswith("```"):
        content = content.replace("```", "", 2).strip()
    return json.loads(content)


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in (429, 500, 503):
        return True
    message = str(exc).upper()
    return "RESOURCE_EXHAUSTED" in message or "INTERNAL" in message or "UNAVAILABLE" in message


def _generate_content_with_retry(client: genai.Client, audio_part: types.Part):
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=STT_MODEL,
                contents=[PROMPT, audio_part],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_error(exc) or attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** (attempt - 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unknown STT failure")


def transcribe_audio_bytes(
    audio_bytes: bytes,
    *,
    filename: str = "recording.webm",
) -> STTResult:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) not found in .env file.")

    client = genai.Client(api_key=api_key)
    mime_type = _guess_mime_type(filename)
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

    response = _generate_content_with_retry(client, audio_part)

    try:
        payload = _extract_json(response.text)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse model response as JSON: {response.text}") from exc

    transcript = str(payload.get("transcript", "")).strip()

    emotion = str(payload.get("emotion", "")).strip()
    selected_menu = str(payload.get("selected_menu", "")).strip()
    reason = str(payload.get("reason", "")).strip()

    return STTResult(
        text=transcript,
        emotion=emotion,
        selected_menu=selected_menu,
        reason=reason,
    )
