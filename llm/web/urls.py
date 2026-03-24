from django.urls import path

from .views import TTSView, order, home, stt_transcribe, order_start_enabled


urlpatterns = [
    path("", home, name="home"),
    path("order/", order, name="order"),
    path("stt/transcribe/", stt_transcribe, name="stt_transcribe"),
    path("tts/", TTSView.as_view(), name="tts"),
    path("api/control/order_start_enabled", order_start_enabled, name="order_start_enabled"),
]
