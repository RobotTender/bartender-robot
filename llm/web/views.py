from django.shortcuts import render

from order_engine import process_user_text


def home(request):
    context = {
        "submitted": False,
        "order_text": "",
        "intent": "",
        "assistant_reply": "",
    }

    if request.method == "POST":
        order_text = request.POST.get("order_text", "").strip()
        engine_result = process_user_text(order_text)
        context["submitted"] = True
        context["order_text"] = order_text
        context["intent"] = engine_result.intent
        context["assistant_reply"] = engine_result.reply

    return render(request, "web/home.html", context)
