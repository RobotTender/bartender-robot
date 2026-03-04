from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views import View

from web.order_engine.classify_order import generate_recommendation_text
from web.order_engine.common import MENU_LABELS
from web.order_engine.make_order import parse_reply
from web.order_engine.stt_pipeline import transcribe_audio_bytes
from web.order_engine.tts_pipeline import synthesize_speech

FORCE_CHOICE_TURN = 1


def home(request):
    context = {
        "submitted": False,
        "order_text": "",
    }

    if request.method == "POST":
        order_text = request.POST.get("order_text", "").strip()
        context["submitted"] = True
        context["order_text"] = order_text

    return render(request, "web/home.html", context)


@require_POST
def stt_transcribe(request):
    reject_count = int(request.POST.get("reject_count", 0))
    context_menu = request.POST.get("context_menu", "")

    # 선택지 제시 후에도 거부 시 첫 번째 메뉴로 강제 확정
    if reject_count >= FORCE_CHOICE_TURN:
        forced_menu = list(MENU_LABELS.keys())[0]
        return JsonResponse({
            "confirmed": True,
            "menu": forced_menu,
            "menu_label": MENU_LABELS[forced_menu],
        })

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

    # 2번째 턴부터는 사용자 응답으로 메뉴 확정 시도
    if context_menu:
        menu = parse_reply(result.text, context_menu)
        if menu:
            return JsonResponse({
                "confirmed": True,
                "menu": menu,
                "menu_label": MENU_LABELS.get(menu, menu),
                "text": result.text,
            })
        # 메뉴 미확정(거부) → 1번 거부 후 바로 선택지 제시
        if reject_count + 1 >= FORCE_CHOICE_TURN:
            options = [{"code": k, "label": v} for k, v in MENU_LABELS.items()]
            return JsonResponse({
                "force_choice": True,
                "options": options,
                "text": result.text,
            })

    recommendation_text = generate_recommendation_text(
        input_text=result.text,
        emotion=result.emotion,
        selected_menu=result.selected_menu,
        reason=result.reason,
    )

    return JsonResponse({
        "text": result.text,
        "transcript": result.text,
        "emotion": result.emotion,
        "selected_menu": result.selected_menu,
        "reason": result.reason,
        "recommendation_text": recommendation_text,
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
