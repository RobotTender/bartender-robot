import io
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from openai import OpenAI


load_dotenv()


@dataclass
class STTResult:
    text: str
    model: str
    language: str


def countdown_timer(duration: int) -> None:
    for i in range(duration, 0, -1):
        print(f"\r[Recording] Speak now! {i} seconds remaining...", end="", flush=True)
        time.sleep(1)
    print("\r[Status] Recording finished. Processing...          ")


def record_audio_bytes(duration: int = 5) -> bytes:
    process = subprocess.run(
        ["arecord", "-d", str(duration), "-t", "wav", "-f", "cd", "-q", "-"],
        capture_output=True,
        check=True,
    )
    return process.stdout


def transcribe_audio_bytes(
    audio_bytes: bytes,
    *,
    model: str = "gpt-4o-mini-transcribe",
    language: str = "ko",
    filename: str = "recording.webm",
) -> STTResult:
    try:
        settings_key = getattr(settings, "OPENAI_API_KEY", None)
    except ImproperlyConfigured:
        settings_key = None

    api_key = settings_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in Django settings or .env file.")

    client = OpenAI(api_key=api_key)
    audio_buffer = io.BytesIO(audio_bytes)
    audio_buffer.name = filename

    transcription = client.audio.transcriptions.create(
        file=audio_buffer,
        model=model,
        language=language,
    )
    return STTResult(
        text=(transcription.text or "").strip(),
        model=model,
        language=language,
    )


def record_and_transcribe(
    *,
    duration: int = 5,
    model: str = "gpt-4o-mini-transcribe",
    language: str = "ko",
) -> STTResult:
    timer_thread = threading.Thread(target=countdown_timer, args=(duration,))
    timer_thread.start()
    audio_bytes = record_audio_bytes(duration=duration)
    timer_thread.join()
    return transcribe_audio_bytes(
        audio_bytes,
        model=model,
        language=language,
        filename="memory_recording.wav",
    )


def main() -> None:
    try:
        result = record_and_transcribe()
        print("\n" + "=" * 40)
        print("       TRANSCRIPTION RESULT")
        print("=" * 40)
        print(result.text)
        print("=" * 40 + "\n")
    except subprocess.CalledProcessError:
        print("\n[Error] Failed to record. Is your microphone connected?")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[Error] An unexpected error occurred: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
