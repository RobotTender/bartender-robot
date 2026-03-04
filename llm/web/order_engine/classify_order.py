import logging
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
logger = logging.getLogger(__name__)
CLASSIFY_MODEL = "gpt-5-nano"


MENU_LABELS = {
    "soju": "소주",
    "beer": "맥주",
    "somaek": "소맥",
}
MENU_ALIASES = {
    "soju": ("소주",),
    "beer": ("맥주", "비어"),
    "somaek": ("소맥", "소주맥주", "소주 맥주"),
}
ORDER_CUES = ("줘", "주세요", "주문", "말아", "말아줘", "한잔", "한 잔", "주라", "내놔")
QUESTION_CUES = ("추천", "뭐", "무엇", "어때", "어떤", "가능", "있어", "있나요", "할까", "?", "왜")


def _detect_menu_from_text(text: str) -> str:
    for menu_code, aliases in MENU_ALIASES.items():
        if any(alias in text for alias in aliases):
            return menu_code
    return ""


def _is_direct_order(input_text: str) -> tuple[bool, str]:
    text = (input_text or "").strip()
    if not text:
        return False, ""

    if any(cue in text for cue in QUESTION_CUES):
        return False, ""

    menu_code = _detect_menu_from_text(text)
    has_order_cue = any(cue in text for cue in ORDER_CUES)
    if menu_code and has_order_cue:
        return True, menu_code
    return False, ""


def build_order_confirmation_text(input_text: str, selected_menu: str) -> str:
    _, detected_menu = _is_direct_order(input_text)
    menu_code = detected_menu or selected_menu
    menu_label = MENU_LABELS.get(menu_code, "해당 메뉴")
    return f"{menu_label} 주문 확인했습니다."


def fallback_recommendation_text(
    *,
    error_message: str,
) -> str:
    logger.error("Recommendation generation failed: %s", error_message)
    return ""


def _extract_response_text(response) -> str:
    text = (getattr(response, "output_text", "") or "").strip()
    if text:
        return text

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            content_type = getattr(content, "type", "")
            if content_type in ("output_text", "text"):
                candidate = (getattr(content, "text", "") or "").strip()
                if candidate:
                    return candidate
    return ""


def generate_recommendation_text(
    *,
    input_text: str,
    emotion: str,
    selected_menu: str,
    reason: str,
    model: str = CLASSIFY_MODEL,
) -> str:
    is_direct_order, _ = _is_direct_order(input_text)
    if is_direct_order:
        return build_order_confirmation_text(input_text, selected_menu)
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return fallback_recommendation_text(error_message="OPENAI_API_KEY is missing.")

    menu_label = MENU_LABELS.get(selected_menu, selected_menu or "추천 메뉴")
    user_text = input_text or "(미인식)"

    prompt = f"""당신은 한국어 바텐더입니다.
        아래 정보를 종합해 사용자에게 전달할 추천 멘트 1문장을 작성하세요.
        - 사용자 발화: {user_text}
        - 감정: {emotion or 'neutral'}
        - 추천 메뉴: {menu_label}
        - 추천 이유: {reason }

        규칙:
        - 한국어 존댓말 1문장만 출력
        - 25~55자 이내
        - 과장 없이 자연스럽게
        - 메뉴명을 반드시 포함
    """

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=prompt,
        )
        generated = _extract_response_text(response)
        if generated:
            return generated

        logger.warning(
            "Empty recommendation text. status=%s incomplete_details=%s",
            getattr(response, "status", None),
            getattr(response, "incomplete_details", None),
        )
    except Exception as exc:
        return fallback_recommendation_text(error_message=str(exc))

    return fallback_recommendation_text(error_message="Model returned empty recommendation text.")
