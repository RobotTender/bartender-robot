from django.test import TestCase
from django.urls import reverse

from order_engine.llm import answer_user, user_classify


class OrderEngineTests(TestCase):
    def test_user_classify_order(self):
        self.assertEqual(user_classify("모히또 한 잔 만들어줘"), "order")

    def test_user_classify_recommend(self):
        self.assertEqual(user_classify("오늘 기분에 맞는 술 추천해줘"), "recommend")

    def test_user_classify_chit(self):
        self.assertEqual(user_classify("안녕 로보텐더"), "chit")

    def test_answer_user_recommend_has_choice_question(self):
        message = answer_user("피곤해서 추천해줘", "recommend")
        self.assertIn("아니면", message)
        self.assertIn("?", message)


class HomeViewTests(TestCase):
    def test_home_post_renders_engine_result(self):
        response = self.client.post(reverse("home"), {"order_text": "모히또 한 잔 주세요"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "분류: order")
        self.assertContains(response, "응답:")
