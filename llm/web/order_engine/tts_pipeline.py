import io
import os
from dataclasses import dataclass

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


def synthesize_speech(
    text: str,
    *,
    model: str = TTS_MODEL,
    voice: str = "nova",
    speed: float = 1.0,
) -> TTSResult:
    try:
        settings_key = getattr(settings, "OPENAI_API_KEY", None)
    except ImproperlyConfigured:
        settings_key = None

    api_key = settings_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in Django settings or .env file.")

    client = OpenAI(api_key=api_key)

    audio_buffer = io.BytesIO()
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        response_format="wav",
        speed=speed,
    ) as response:
        for chunk in response.iter_bytes(chunk_size=4096):
            audio_buffer.write(chunk)

    return TTSResult(
        audio_bytes=audio_buffer.getvalue(),
        model=model,
        voice=voice,
    )
