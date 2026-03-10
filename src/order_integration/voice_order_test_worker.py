#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

from order_integration.voice_order_runtime import build_voice_order_runtime
from order_integration.gemini_stt_pipeline import transcribe_audio_bytes


def _emit(event_type: str, **payload):
    body = {"type": str(event_type), **payload}
    print(json.dumps(body, ensure_ascii=False), flush=True)


def _read_payload() -> dict:
    raw = (sys.stdin.read() or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        # line-oriented fallback
        first_line = raw.splitlines()[0].strip()
        if not first_line:
            return {}
        return json.loads(first_line)


def _capture_stt_payload():
    events = []

    def _stage(stage: str, actor: str, message: str, data=None):
        body = {
            "type": "stage",
            "stage": str(stage),
            "actor": str(actor),
            "message": str(message),
        }
        if data is not None:
            body["data"] = data
        events.append(body)

    _stage("input_request", "voice_processor", "백엔드 음성 입력 요청 수신", {"request_stt": True})

    try:
        import speech_recognition as sr
    except Exception as e:
        events.append({"type": "error", "actor": "stt_pipeline", "message": f"speech_recognition 미설치: {e}"})
        return False, {}, events, "speech_recognition 미설치"

    try:
        recognizer = sr.Recognizer()
        _stage("stt_wait", "stt_pipeline", "마이크 입력 대기")
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=6.0, phrase_time_limit=8.0)
        _stage("stt_process", "stt_pipeline", "Gemini STT 처리 중")
        wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
        stt_result = transcribe_audio_bytes(wav_bytes, filename="recording.wav")
        text = str(stt_result.text or "").strip()
        if not text:
            events.append({"type": "error", "actor": "stt_pipeline", "message": "STT 결과가 비어 있습니다."})
            return False, {}, events, "STT 결과가 비어 있습니다."
        meta = {
            "stt_text": text,
            "emotion": str(stt_result.emotion or "").strip() or "neutral",
            "recommend_menu": str(stt_result.recommend_menu or "").strip(),
            "reason": str(stt_result.reason or "").strip(),
        }
        _stage("stt", "stt_pipeline", "Gemini STT 결과 반영", meta)
        return (
            True,
            {
                "text": text,
                "emotion": meta["emotion"],
                "recommend_menu": meta["recommend_menu"],
                "reason": meta["reason"],
            },
            events,
            "",
        )
    except sr.WaitTimeoutError:
        events.append({"type": "error", "actor": "stt_pipeline", "message": "마이크 입력 시간초과"})
        return False, {}, events, "마이크 입력 시간초과"
    except sr.UnknownValueError:
        events.append({"type": "error", "actor": "stt_pipeline", "message": "STT 인식 실패"})
        return False, {}, events, "STT 인식 실패"
    except sr.RequestError as e:
        msg = f"STT 서비스 요청 실패: {e}"
        events.append({"type": "error", "actor": "stt_pipeline", "message": msg})
        return False, {}, events, msg
    except Exception as e:
        msg = f"Gemini STT 처리 예외: {e}"
        events.append({"type": "error", "actor": "stt_pipeline", "message": msg})
        return False, {}, events, msg


def main() -> int:
    if load_dotenv is not None:
        project_root = SRC_ROOT.parent
        load_dotenv(dotenv_path=project_root / ".env", override=False)

    payload = _read_payload()
    input_text = str(payload.get("input_text", "") or "").strip()
    recommend_menu = str(payload.get("recommend_menu", "") or "").strip()
    allow_llm = bool(payload.get("allow_llm", True))
    request_stt = bool(payload.get("request_stt", False)) or (not bool(input_text))
    emotion = ""
    reason = ""

    if request_stt:
        ok_stt, stt_payload, stt_events, stt_msg = _capture_stt_payload()
        for event in stt_events:
            body = dict(event)
            evt_type = str(body.pop("type", "stage"))
            _emit(evt_type, **body)
        if not ok_stt:
            _emit(
                "result",
                worker_pid=os.getpid(),
                status="error",
                selected_menu="",
                selected_menu_label="",
                tts_text=str(stt_msg or "STT 실패"),
                recipe={},
                route="stt_error",
            )
            _emit("done", ok=False)
            return 1
        input_text = str((stt_payload or {}).get("text", "") or "").strip()
        if not recommend_menu:
            recommend_menu = str((stt_payload or {}).get("recommend_menu", "") or "").strip()
        emotion = str((stt_payload or {}).get("emotion", "") or "").strip()
        reason = str((stt_payload or {}).get("reason", "") or "").strip()

    output = build_voice_order_runtime(
        input_text=input_text,
        recommend_menu=recommend_menu,
        allow_llm=allow_llm,
        emotion=emotion,
        reason=reason,
    )
    for event in output.events:
        body = dict(event)
        evt_type = str(body.pop("type", "stage"))
        _emit(evt_type, **body)
    _emit("result", worker_pid=os.getpid(), **dict(output.result_payload))
    _emit("done", ok=bool(output.done_ok))
    return 0 if bool(output.done_ok) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - worker crash fallback
        _emit("error", actor="voice_worker", message=f"워커 예외 발생: {exc}")
        _emit("done", ok=False)
        raise
