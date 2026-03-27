from django.urls import path

from .views import TTSView, order, home, robot_status, stt_transcribe


urlpatterns = [
    path("", home, name="home"),
    path("order/", order, name="order"),
    path("stt/transcribe/", stt_transcribe, name="stt_transcribe"),
    path("robot/status/", robot_status, name="robot_status"),
    path("tts/", TTSView.as_view(), name="tts"),
]
