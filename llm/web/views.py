from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from web.order_engine.stt_pipeline import transcribe_audio_bytes


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
            language="ko",
            filename=audio_file.name or "recording.webm",
        )
    except Exception as exc:
        status_code = getattr(exc, "status_code", 500)
        if not isinstance(status_code, int):
            status_code = 500
        return JsonResponse({"error": str(exc)}, status=status_code)

    return JsonResponse(
        {
            "text": result.text,
            "model": result.model,
            "language": result.language,
        }
    )
