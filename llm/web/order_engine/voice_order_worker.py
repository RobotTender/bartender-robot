#!/usr/bin/env python3
import base64
import json
import os
import sys
import time
from pathlib import Path


# Add necessary paths for standalone execution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
except Exception:
    pass

from stt_pipeline import transcribe_audio_bytes
from graph import create_graph_flow
from common import MENU_LABELS


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
        return {}


def _capture_stt_payload(audio_bytes=None, audio_filename="recording.webm"):
    events = []
    def _push_event(evt):
        events.append(evt)
        _emit(evt.get("type", "stage"), **evt)

    try:
        if audio_bytes:
            _push_event({"type": "stage", "actor": "stt_pipeline", "message": "오디오 페이로드 처리 중..."})
        else:
            # Fallback to microphone if no audio_bytes provided
            _push_event({"type": "stage", "actor": "stt_pipeline", "message": "마이크 입력을 기다리는 중..."})
            try:
                import speech_recognition as sr
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio = r.listen(source, timeout=10, phrase_time_limit=10)
                    audio_bytes = audio.get_wav_data()
                    audio_filename = "microphone.wav"
            except Exception as e:
                msg = f"마이크 녹음 실패: {e}"
                _push_event({"type": "error", "actor": "stt_pipeline", "message": msg})
                return False, {}, events, msg

        stt_result = transcribe_audio_bytes(audio_bytes, filename=audio_filename)
        text = str(stt_result.text or "").strip()
        meta = {
            "transcript": text,
            "emotion": str(stt_result.emotion or "neutral").strip(),
            "recommend_menu": str(stt_result.recommend_menu or "").strip(),
            "reason": str(stt_result.reason or "").strip(),
        }
        _push_event({"type": "stage", "actor": "stt_pipeline", "message": "STT 결과 반영", "data": meta})
        return True, meta, events, ""
    except Exception as e:
        msg = f"STT 처리 예외: {e}"
        _push_event({"type": "error", "actor": "stt_pipeline", "message": msg})
        return False, {}, events, msg


def main() -> int:
    payload = _read_payload()
    input_text = str(payload.get("input_text", "") or "").strip()
    recommend_menu = str(payload.get("recommend_menu", "") or "").strip()
    # allow_llm = bool(payload.get("allow_llm", True)) # LangGraph version might use this differently
    emotion = str(payload.get("emotion", "") or "").strip()
    reason = str(payload.get("reason", "") or "").strip()
    audio_filename = str(payload.get("audio_filename", "recording.webm") or "recording.webm").strip()
    audio_bytes = b""
    audio_base64 = str(payload.get("audio_base64", "") or "").strip()
    
    if audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64, validate=True)
        except Exception as e:
            _emit("error", actor="voice_worker", message=f"오디오 페이로드 디코딩 실패: {e}")
            _emit("done", ok=False)
            return 1

    request_stt = bool(payload.get("request_stt", False)) or (not bool(input_text)) or bool(audio_bytes)

    if request_stt:
        ok_stt, stt_payload, _, stt_msg = _capture_stt_payload(
            audio_bytes=audio_bytes if audio_bytes else None,
            audio_filename=audio_filename,
        )
        if not ok_stt:
            _emit("result", status="error", tts_text=stt_msg, ok=False)
            _emit("done", ok=False)
            return 1
        input_text = stt_payload.get("transcript", "")
        recommend_menu = stt_payload.get("recommend_menu", "")
        emotion = stt_payload.get("emotion", "")
        reason = stt_payload.get("reason", "")

    # Run LangGraph flow
    graph = create_graph_flow()
    initial_state = {
        "input_text": input_text,
        "recommend_menu": recommend_menu,
        "emotion": emotion,
        "reason": reason,
        "status": "init",
        "retry": False,
        "events": []
    }

    try:
        final_state = graph.invoke(initial_state)
        
        # Format result to match UI expectations
        selected_menu = final_state.get("selected_menu", "")
        selected_menu_label = MENU_LABELS.get(selected_menu, "")
        
        result = {
            "status": final_state.get("status", "unknown"),
            "selected_menu": selected_menu,
            "selected_menu_label": selected_menu_label,
            "tts_text": final_state.get("tts_text", ""),
            "llm_text": final_state.get("tts_text", ""), # UI sometimes uses llm_text
            "recipe": final_state.get("recipe", {}),
            "route": final_state.get("route", "langgraph"),
            "llm_used": final_state.get("llm_used", False),
            "llm_reason": final_state.get("llm_reason", ""),
        }
        
        _emit("result", **result)
        _emit("done", ok=(final_state.get("status") == "success"))
        return 0 if final_state.get("status") == "success" else 1
    except Exception as e:
        _emit("error", actor="voice_worker", message=f"Graph 실행 중 예외 발생: {e}")
        _emit("done", ok=False)
        return 1


if __name__ == "__main__":
    sys.exit(main())
