from django.test import TestCase
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from unittest.mock import patch

from web.order_engine.classify_order import generate_recommendation_text
from web.order_engine.make_order import parse_reply


class HomeViewTests(TestCase):
    def test_home_post_renders_input_text(self):
        response = self.client.post(reverse("home"), {"order_text": "모히또 한 잔 주세요"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "입력: 모히또 한 잔 주세요")


class STTViewTests(TestCase):
    @patch("web.views.generate_recommendation_text")
    @patch("web.views.transcribe_audio_bytes")
    def test_stt_transcribe_returns_json(self, mock_transcribe, mock_generate_recommendation):
        from web.order_engine.stt_pipeline import STTResult

        mock_generate_recommendation.return_value = "기분에 맞춰 맥주 한 잔 추천드려요."
        mock_transcribe.return_value = STTResult(
            text="모히또 한 잔 주세요",
            emotion="happy",
            selected_menu="beer",
            reason="상쾌한 주문 톤이라 가볍게 추천했습니다.",
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
                "transcript": "모히또 한 잔 주세요",
                "emotion": "happy",
                "selected_menu": "beer",
                "reason": "상쾌한 주문 톤이라 가볍게 추천했습니다.",
                "recommendation_text": "기분에 맞춰 맥주 한 잔 추천드려요.",
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

    def test_stt_force_confirmed_when_reject_count_exceeds_limit(self):
        # reject_count >= FORCE_CHOICE_TURN → 오디오 없이 첫 번째 메뉴로 강제 확정
        response = self.client.post(
            reverse("stt_transcribe"),
            {"reject_count": "1"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["confirmed"])
        self.assertEqual(data["menu"], "soju")

    @patch("web.views.parse_reply", return_value="soju")
    @patch("web.views.transcribe_audio_bytes")
    def test_stt_returns_confirmed_when_menu_resolved(self, mock_transcribe, _mock_parse_reply):
        from web.order_engine.stt_pipeline import STTResult

        mock_transcribe.return_value = STTResult(text="그걸로 줘", emotion="neutral", selected_menu="", reason="")
        audio = SimpleUploadedFile("recording.webm", b"a" * 4096, content_type="audio/webm")
        response = self.client.post(
            reverse("stt_transcribe"),
            {"audio": audio, "context_menu": "soju", "reject_count": "0"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["confirmed"])
        self.assertEqual(data["menu"], "soju")

    @patch("web.views.parse_reply", return_value="")
    @patch("web.views.transcribe_audio_bytes")
    def test_stt_returns_force_choice_on_rejection(self, mock_transcribe, _mock_parse_reply):
        from web.order_engine.stt_pipeline import STTResult

        mock_transcribe.return_value = STTResult(text="아니", emotion="neutral", selected_menu="", reason="")
        audio = SimpleUploadedFile("recording.webm", b"a" * 4096, content_type="audio/webm")
        response = self.client.post(
            reverse("stt_transcribe"),
            {"audio": audio, "context_menu": "beer", "reject_count": "0"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["force_choice"])
        self.assertEqual(len(data["options"]), 3)


class TTSViewTests(TestCase):
    @patch("web.views.synthesize_speech")
    def test_tts_returns_audio(self, mock_synthesize):
        from web.order_engine.tts_pipeline import TTSResult

        mock_synthesize.return_value = TTSResult(
            audio_bytes=b"fake-wav-bytes",
            model="gpt-4o-mini-tts",
            voice="nova",
        )
        response = self.client.post(
            reverse("tts"),
            {"text": "소주와 맥주를 섞은 소맥은 어떠신가요?"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "audio/wav")
        self.assertEqual(response.content, b"fake-wav-bytes")

    def test_tts_rejects_empty_text(self):
        response = self.client.post(reverse("tts"), {"text": ""})
        self.assertEqual(response.status_code, 400)
        self.assertJSONEqual(response.content, {"error": "text is required"})


class ClassifyOrderTests(TestCase):
    def test_generate_recommendation_direct_order_returns_confirmation(self):
        text = generate_recommendation_text(
            input_text="소주 말아줘",
            emotion="happy",
            selected_menu="beer",
            reason="요청 분위기에 맞음",
        )
        self.assertEqual(text, "소주 주문 확인했습니다.")

    @patch("web.order_engine.classify_order.os.getenv", return_value=None)
    def test_generate_recommendation_question_returns_recommendation(self, _mock_getenv):
        text = generate_recommendation_text(
            input_text="오늘 뭐 마시면 좋을까?",
            emotion="tired",
            selected_menu="somaek",
            reason="기분 전환이 필요해 보임",
        )
        self.assertEqual(text, "")


class ParseReplyTests(TestCase):
    def test_direct_menu_returns_menu_code(self):
        self.assertEqual(parse_reply("소주 줘"), "soju")

    def test_direct_menu_with_rejection_still_returns_menu(self):
        # "말고" 같은 거절 표현이 있어도 메뉴가 있으면 해당 메뉴 반환
        self.assertEqual(parse_reply("맥주 말고"), "beer")

    def test_plain_rejection_returns_empty(self):
        self.assertEqual(parse_reply("아니"), "")

    def test_confirm_cue_with_context_menu_returns_context_menu(self):
        self.assertEqual(parse_reply("그걸로 줘", context_menu="soju"), "soju")

    def test_confirm_cue_without_context_menu_falls_through(self):
        # context_menu 없으면 LLM 폴백 → API 키 없을 시 ""
        with patch("web.order_engine.make_order.os.getenv", return_value=None):
            self.assertEqual(parse_reply("그래"), "")

    def test_empty_input_returns_empty(self):
        self.assertEqual(parse_reply(""), "")
