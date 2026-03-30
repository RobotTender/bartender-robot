import io
import os
from dataclasses import dataclass
from typing import Iterator

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

TTS_MODEL = "gpt-4o-mini-tts"

VOICES = [
    "nova",
    "alloy",
    "onyx",
]


@dataclass
class TTSResult:
    audio_bytes: bytes
    model: str
    voice: str


def _get_openai_client() -> OpenAI:
    try:
        settings_key = getattr(settings, "OPENAI_API_KEY", None)
    except ImproperlyConfigured:
        settings_key = None

    api_key = settings_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in Django settings or .env file.")

    return OpenAI(api_key=api_key)


def stream_speech(
    text: str,
    *,
    model: str = TTS_MODEL,
    voice: str = "nova",
    speed: float = 1.0,
    chunk_size: int = 4096,
) -> Iterator[bytes]:
    client = _get_openai_client()
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        response_format="wav",
        speed=speed,
    ) as response:
        for chunk in response.iter_bytes(chunk_size=chunk_size):
            yield chunk


def synthesize_speech(
    text: str,
    *,
    model: str = TTS_MODEL,
    voice: str = "nova",
    speed: float = 1.0,
) -> TTSResult:
    audio_buffer = io.BytesIO()
    for chunk in stream_speech(text, model=model, voice=voice, speed=speed):
        audio_buffer.write(chunk)

    return TTSResult(
        audio_bytes=audio_buffer.getvalue(),
        model=model,
        voice=voice,
    )
