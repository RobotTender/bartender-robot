from django.test import TestCase
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from unittest.mock import patch


class HomeViewTests(TestCase):
    def test_home_post_renders_input_text(self):
        response = self.client.post(reverse("home"), {"order_text": "모히또 한 잔 주세요"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "입력: 모히또 한 잔 주세요")


class STTViewTests(TestCase):
    @patch("web.views.transcribe_audio_bytes")
    def test_stt_transcribe_returns_json(self, mock_transcribe):
        from web.order_engine.stt_pipeline import STTResult

        mock_transcribe.return_value = STTResult(
            text="모히또 한 잔 주세요",
            model="gpt-4o-mini-transcribe",
            language="ko",
        )
        audio = SimpleUploadedFile("recording.webm", b"a" * 4096, content_type="audio/webm")
        response = self.client.post(
            reverse("stt_transcribe"),
            {"audio": audio},
        )
        self.assertEqual(response.status_code, 200)
        self.assertJSONEqual(
            response.content,
            {
                "text": "모히또 한 잔 주세요",
                "model": "gpt-4o-mini-transcribe",
                "language": "ko",
            },
        )

    def test_stt_transcribe_rejects_too_short_audio(self):
        audio = SimpleUploadedFile("recording.webm", b"tiny", content_type="audio/webm")
        response = self.client.post(reverse("stt_transcribe"), {"audio": audio})
        self.assertEqual(response.status_code, 400)
        self.assertJSONEqual(
            response.content,
            {"error": "audio too short; please record a bit longer"},
        )
