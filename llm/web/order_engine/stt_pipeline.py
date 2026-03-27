import json
import mimetypes
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

STT_MODEL = os.getenv("STT_MODEL", "gemini-flash-latest")
MAX_RETRIES = int(os.getenv("STT_MAX_RETRIES", "2"))

SOJU_HINTS = (
    "소주", "소쥬", "쏘주", "소듀", "수주", "소주로", "소주를",
    
)
JUICE_HINTS = (
    "주스", "쥬스", "쥬쓰", "쥬수", "주수", "쥬", "주스를",
)

PROMPT = """You are an audio order analyzer for a Korean bartender app.
Analyze the provided audio and return strict JSON only with these keys:
{
  "transcript": "recognized Korean text",
  "emotion": "happy|sad|angry|tired|neutral",
  "recommend_menu": "soju|beer|somaek|juice|koktail|",
  "reason": "short reason in Korean"
}
Rules:
- Output must be valid JSON only. No markdown.
- If speech is unclear, set transcript to an empty string and emotion to neutral.
- Keep reason concise (max 1 sentence).
- Pay special attention to Korean menu words that sound similar.
- "소주" and "주스" must be clearly distinguished.
- If one of those two menus is intended, make both transcript and recommend_menu consistent with that same menu.
"""


@dataclass
class STTResult:
    text: str
    emotion: str = ""
    recommend_menu: str = ""
    reason: str = ""


_AUDIO_MIME_OVERRIDES = {
    ".webm": "audio/webm",
    ".mp4": "audio/mp4",
}


def _guess_mime_type(filename: str | None) -> str:
    if filename:
        ext = os.path.splitext(filename)[-1].lower()
        if ext in _AUDIO_MIME_OVERRIDES:
            return _AUDIO_MIME_OVERRIDES[ext]
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


def _align_menu_transcript(transcript: str, recommend_menu: str) -> str:
    text = (transcript or "").strip()
    menu = (recommend_menu or "").strip()
    if not text or menu not in {"soju", "juice"}:
        return text

    if menu == "soju" and "주스" in text and "소주" not in text:
        return text.replace("주스", "소주")
    if menu == "juice" and "소주" in text and "주스" not in text:
        return text.replace("소주", "주스")
    return text


def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)


def _resolve_soju_juice_menu(transcript: str, recommend_menu: str) -> str:
    text = (transcript or "").strip()
    menu = (recommend_menu or "").strip()
    if not text:
        return menu

    soju_score = 0
    juice_score = 0

    if _contains_any(text, SOJU_HINTS):
        soju_score += 2
    if _contains_any(text, JUICE_HINTS):
        juice_score += 2

    if "소주" in text:
        soju_score += 3
    if "주스" in text or "쥬스" in text:
        juice_score += 3

    if menu == "soju":
        soju_score += 1
    elif menu == "juice":
        juice_score += 1

    if soju_score > juice_score and soju_score >= 2:
        return "soju"
    if juice_score > soju_score and juice_score >= 2:
        return "juice"
    return menu


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

    emotion = str(payload.get("emotion", "")).strip()
    recommend_menu = str(payload.get("recommend_menu", "")).strip()
    reason = str(payload.get("reason", "")).strip()
    recommend_menu = _resolve_soju_juice_menu(
        str(payload.get("transcript", "")).strip(),
        recommend_menu,
    )
    transcript = _align_menu_transcript(
        str(payload.get("transcript", "")).strip(),
        recommend_menu,
    )

    return STTResult(
        text=transcript,
        emotion=emotion,
        recommend_menu=recommend_menu,
        reason=reason,
    )
