from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.views import View

from web.order_engine.classify_order import generate_recommendation_text
from web.order_engine.stt_pipeline import transcribe_audio_bytes
from web.order_engine.tts_pipeline import synthesize_speech


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

    recommendation_text = generate_recommendation_text(
        input_text=result.text,
        emotion=result.emotion,
        selected_menu=result.selected_menu,
        reason=result.reason,
    )

    return JsonResponse(
        {
            "text": result.text,
            "inputtext": result.text,
            "transcript": result.text,
            "emotion": result.emotion,
            "selected_menu": result.selected_menu,
            "reason": result.reason,
            "recommendation_text": recommendation_text,
        }
    )


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
