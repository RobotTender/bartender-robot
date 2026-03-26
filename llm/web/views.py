from pathlib import Path
import json
import sys
import logging

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views import View

from web.order_engine.common import MENU_LABELS
from web.order_engine.stt_pipeline import transcribe_audio_bytes
from web.order_engine.tts_pipeline import synthesize_speech
from web.order_engine.robot_topic import _start_robot_topic_publish

ORDER_ENGINE_DIR = Path(__file__).resolve().parent / "order_engine"
if str(ORDER_ENGINE_DIR) not in sys.path:
    sys.path.append(str(ORDER_ENGINE_DIR))

from web.order_engine.graph import create_graph_flow

graph = create_graph_flow()
logger = logging.getLogger(__name__)


ORDER_START_CUES = ("제조시작합니다", "주문 시작합니다")
ROBOT_STATUS_FILE = Path("/tmp/bartender_order_status.json")


def _read_robot_status() -> dict:
    try:
        if not ROBOT_STATUS_FILE.exists():
            return {"state": "unknown"}
        payload = json.loads(ROBOT_STATUS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read robot status: %s", exc)
        return {"state": "unknown"}

    return payload if isinstance(payload, dict) else {"state": "unknown"}



def home(request):
    return render(request, "web/home.html")


def order(request):
    # Start page -> order page 진입 시 이전 주문 세션 상태 초기화
    request.session["order_state"] = {}
    request.session["robot_job_active"] = False
    context = {
        "status": "init",
        "retry":False,
        "recommend_menu":"",
    }

    return render(request, "web/order.html", context)



@require_POST
def stt_transcribe(request):

    session_state = request.session.get("order_state", {})
    
    audio_file = request.FILES.get("audio")
    if not audio_file:
        return JsonResponse({"error": "audio file is required"}, status=400)

    audio_bytes = audio_file.read()
    if not audio_bytes:
        return JsonResponse({"error": "empty audio payload"}, status=400)

    if len(audio_bytes) < 2048:
        return JsonResponse({"error": "audio too short; please record a bit longer"}, status=400)

    try:
        result = transcribe_audio_bytes(
            audio_bytes,
            filename=audio_file.name or "recording.webm",
        )
    except Exception as exc:
        status_code = getattr(exc, "status_code", 500)
        if not isinstance(status_code, int):
            status_code = 500
        return JsonResponse({"error": str(exc)}, status=status_code)


    saved_status = session_state.get("status", "init")
    saved_retry = bool(session_state.get("retry", False))
    saved_recommend_menu = session_state.get("recommend_menu") or session_state.get("recommand_menu")
    saved_ratio = session_state.get("ratio", "")
    saved_ratio_reason = session_state.get("ratio_reason", "")
    if saved_recommend_menu:
        recommend_menu  = saved_recommend_menu
    else:
        recommend_menu = result.recommend_menu
        
    
  
    graph_state = {
        "input_text": result.text,
        "status": saved_status,
        "emotion": result.emotion,
        "recommend_menu": recommend_menu,
        "reason": result.reason,
        "retry": saved_retry,
        "ratio": saved_ratio,
        "ratio_reason": saved_ratio_reason,
    }

    try:
        graph_result = graph.invoke(graph_state)
    except Exception as exc:
        return JsonResponse({"error": f"graph invoke failed: {exc}"}, status=500)

    tts_text = graph_result.get("tts_text", "")
    selected_menu = graph_result.get("selected_menu", "")
    new_status = graph_result.get("status", "init")
    new_retry = bool(graph_result.get("retry", False))
    new_recommend_menu = graph_result.get("recommend_menu", "")
    new_ratio = graph_result.get("ratio", "")
    new_ratio_reason = graph_result.get("ratio_reason", "")
    recipe = graph_result.get("recipe",{})
    request.session["order_state"] = {
        "status": new_status,
        "retry": new_retry,
        "recommend_menu": new_recommend_menu,
        "ratio": new_ratio,
        "ratio_reason": new_ratio_reason,
    }

    logger.info(
        "Before robot publish: tts_text=%s selected_menu=%s ratio=%s ratio_reason=%s recipe=%s status=%s",
        tts_text,
        selected_menu,
        new_ratio,
        new_ratio_reason,
        recipe,
        new_status,
    )

    if recipe:
        command_payload = {
            "drinks": graph_result.get("drinks", selected_menu),
            "recipe": recipe,
            "status": new_status,
        }
        request.session["robot_job_active"] = True
        _start_robot_topic_publish(command_payload)
    else:
        request.session["robot_job_active"] = False

    return JsonResponse({
        "tts_text": tts_text,
        "status": new_status,
        "making": bool(recipe),
    })


def robot_status(request):
    status_payload = _read_robot_status()
    state = status_payload.get("state", "unknown")
    done = state in {"completed", "failed", "idle"}
    if done:
        request.session["robot_job_active"] = False

    return JsonResponse({
        "active": bool(request.session.get("robot_job_active", False)),
        "state": state,
        "done": done,
        "status": status_payload,
    })


class TTSView(View):
    def post(self, request):
        text = request.POST.get("text", "").strip()
        if not text:
            return JsonResponse({"error": "text is required"}, status=400)

        try:
            result = synthesize_speech(text)
        except Exception as exc:
            status_code = getattr(exc, "status_code", 500)
            if not isinstance(status_code, int):
                status_code = 500
            return JsonResponse({"error": str(exc)}, status=status_code)
        
        # 브라우저에서 직접 재생할 수 있도록 Content-Disposition을 inline으로 설정
        response = HttpResponse(result.audio_bytes, content_type="audio/wav")
        response["Content-Disposition"] = 'inline; filename="synthesized.wav"'
        return response
