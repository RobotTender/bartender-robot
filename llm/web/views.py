from pathlib import Path
import sys
import logging
import json

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
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




def home(request):
    order_start_enabled = request.session.get("order_start_enabled", True)
    return render(request, "web/home.html", {"order_start_enabled": order_start_enabled})


def order(request):
    # Check if ordering is enabled in session
    if not request.session.get("order_start_enabled", True):
        return redirect("home")
        
    # Start page -> order page 진입 시 이전 주문 세션 상태 초기화
    request.session["order_state"] = {}
    context = {
        "status": "init",
        "retry":False,
        "recommand_menu":"",
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
    saved_recommend_menu = session_state.get("recommend_menu")
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
        "retry": saved_retry
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
    recipe = graph_result.get("recipe",{})
    request.session["order_state"] = {
        "status": new_status,
        "retry": new_retry,
        "recommand_menu": new_recommend_menu,
    }

    if recipe:
        logger.info(
            "Recipe generated: selected_menu=%s recipe=%s status=%s",
            selected_menu,
            recipe,
            new_status,
        )
        command_payload = {
            "drinks": graph_result.get("drinks", selected_menu),
            "recipe": recipe,
            "status": new_status,
        }
        _start_robot_topic_publish(command_payload)

    return JsonResponse({
        "tts_text": tts_text,
        "status": new_status,
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


@csrf_exempt
@require_POST
def order_start_enabled(request):
    try:
        data = json.loads(request.body)
        enabled = bool(data.get("enabled", False))
        
        # 세션에 주문 시작 가능 상태 저장
        request.session["order_start_enabled"] = enabled
        
        return JsonResponse({
            "ok": True,
            "order_start_enabled": enabled
        })
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)
