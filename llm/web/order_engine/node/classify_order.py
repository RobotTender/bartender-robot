import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from common import MODEL, MENU_LABELS, build_order_confirmation_text, build_recommendation_text, detect_menu_from_text
from ratio_utils import select_ratio_with_llm
from state import GraphState

load_dotenv()
logger = logging.getLogger(__name__)

ORDER_CUES = ("줘", "주세요", "시킬게", "주문", "말아", "말아줘", "한잔", "한 잔", "주라", "내놔",'부탁해', '먹고싶어', '마시고싶어', '먹고싶다', '마시고싶다')


def _is_direct_order(input_text: str) -> tuple[bool, str]:
    menu_code = detect_menu_from_text(input_text)
    has_order_cue = any(cue in input_text for cue in ORDER_CUES)
    if menu_code and has_order_cue:
        return True, menu_code
    return False, ""


def fallback_recommendation_text(
    *,
    error_message: str,
    selected_menu: str = "",
    ratio: str = "",
    ratio_reason: str = "",
) -> GraphState:
    logger.error("Recommendation generation failed: %s", error_message)
    return {
        "tts_text": build_recommendation_text(selected_menu, ratio, ratio_reason) if selected_menu else "",
        "status": "processing",
        "selected_menu": selected_menu,
        "ratio": ratio,
        "ratio_reason": ratio_reason,
    }


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


def classify_node(state: GraphState) -> GraphState:
    input_text = state.get("input_text", "")

    if not input_text or not input_text.strip():
        raise ValueError("input_text is required and cannot be empty.")

    is_direct_order, selected_menu = _is_direct_order(input_text)
    ratio = state.get("ratio", "")
    ratio_reason = state.get("ratio_reason", "")

    if is_direct_order:
        if selected_menu in ("somaek", "koktail") and not ratio:
            try:
                ratio, ratio_reason = select_ratio_with_llm(
                    selected_menu,
                    input_text,
                    state.get("emotion", "neutral"),
                    state.get("user_profile", {}),
                )
            except Exception as exc:
                return fallback_recommendation_text(
                    error_message=str(exc),
                    selected_menu=selected_menu,
                    ratio=ratio,
                    ratio_reason=ratio_reason,
                )
        tts_text = build_order_confirmation_text(selected_menu, ratio)

    else:
        api_key = os.getenv("OPENAI_API_KEY")
        recommend_menu = state.get("recommend_menu", "") or state.get("recommand_menu", "")
        selected_menu = recommend_menu
        emotion = state.get("emotion", "neutral")
        reason = state.get("reason", "")
        menu_label = MENU_LABELS.get(recommend_menu, "")

        if not selected_menu or not menu_label:
            return fallback_recommendation_text(error_message="recommend_menu is missing or invalid.")

        if selected_menu in ("somaek", "koktail") and not ratio:
            try:
                ratio, ratio_reason = select_ratio_with_llm(
                    selected_menu,
                    input_text,
                    emotion,
                    state.get("user_profile", {}),
                )
            except Exception as exc:
                return fallback_recommendation_text(
                    error_message=str(exc),
                    selected_menu=selected_menu,
                    ratio=ratio,
                    ratio_reason=ratio_reason,
                )

        if not api_key:
            return fallback_recommendation_text(
                error_message="OPENAI_API_KEY is missing.",
                selected_menu=selected_menu,
                ratio=ratio,
                ratio_reason=ratio_reason,
            )

        ratio_rule = ""
        if selected_menu in ("somaek", "koktail") and ratio:
            ratio_rule = (
                f'- 비율 표현은 반드시 "{ratio}"만 그대로 포함\n'
                f'- 왜 그 비율을 추천하는지 "{ratio_reason}" 의미를 자연스럽게 반드시 포함\n'
            )

        prompt = f"""당신은 한국어 바텐더입니다.
            아래 정보를 종합해 사용자에게 전달할 추천 멘트 1문장을 작성하세요.
            - 사용자 발화: {input_text}
            - 감정: {emotion}
            - 추천 메뉴: {menu_label}
            - 추천 이유: {reason}
            - 비율 추천 이유: {ratio_reason}

            규칙:
            - 한국어 존댓말 1문장만 출력
            - 25~80자 이내
            - 과장 없이 자연스럽게
            - 메뉴명은 반드시 "{menu_label}" 문자열만 그대로 포함
            - "마가리타 칵테일"처럼 세부 메뉴명으로 확장하지 말 것
            {ratio_rule}- 소맥과 칵테일이 아닐 때는 비율을 말하지 말 것
        """

        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=MODEL,
                input=prompt,
            )
            tts_text = _extract_response_text(response)
            if not tts_text:
                return fallback_recommendation_text(
                    error_message="Model returned empty recommendation text.",
                    selected_menu=selected_menu,
                    ratio=ratio,
                    ratio_reason=ratio_reason,
                )
        except Exception as exc:
            return fallback_recommendation_text(
                error_message=str(exc),
                selected_menu=selected_menu,
                ratio=ratio,
                ratio_reason=ratio_reason,
            )

    return {
        **state,
        "tts_text": tts_text,
        "status": "success" if is_direct_order else "processing",
        "selected_menu": selected_menu,
        "ratio": ratio,
        "ratio_reason": ratio_reason,
    }
