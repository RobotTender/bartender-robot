import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from common import (
    MODEL,
    MENU_LABELS,
    build_order_confirmation_text,
    build_ratio_recommendation_prompt_text,
    build_recommendation_text,
    detect_menu_from_text,
)
from ratio_utils import extract_ratio_from_text, select_ratio_with_llm
from state import GraphState

load_dotenv()
logger = logging.getLogger(__name__)

ORDER_CUES = (
    "줘",
    "주세요",
    "시킬게",
    "시킬건데",
    "시키고",
    "주문",
    "말아",
    "말아줘",
    "한잔",
    "한 잔",
    "주라",
    "내놔",
    "부탁해",
    "먹고싶어",
    "마시고싶어",
    "먹고싶다",
    "마시고싶다",
)


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


def _resolve_selected_menu(state: GraphState, input_text: str) -> str:
    detected_menu = detect_menu_from_text(input_text)
    if detected_menu:
        return detected_menu
    return state.get("recommend_menu", "") or state.get("recommand_menu", "")


def _build_state_update(
    state: GraphState,
    *,
    tts_text: str,
    status: str,
    selected_menu: str,
    ratio: str = "",
    ratio_reason: str = "",
    recommend_menu: str | None = None,
) -> GraphState:
    resolved_recommend_menu = selected_menu if recommend_menu is None else recommend_menu
    return {
        **state,
        "tts_text": tts_text,
        "status": status,
        "selected_menu": selected_menu,
        "recommend_menu": resolved_recommend_menu,
        "ratio": ratio,
        "ratio_reason": ratio_reason,
    }


def _resolve_ratio(
    *,
    state: GraphState,
    selected_menu: str,
    input_text: str,
    ratio: str,
    ratio_reason: str,
) -> tuple[str, str]:
    if selected_menu not in ("somaek", "koktail") or ratio:
        return ratio, ratio_reason

    explicit_ratio = extract_ratio_from_text(input_text)
    if explicit_ratio:
        return explicit_ratio, "사용자가 직접 선택한 비율"

    return select_ratio_with_llm(
        selected_menu,
        input_text,
        state.get("emotion", "neutral"),
        state.get("user_profile", {}),
    )


def _build_recommendation_prompt(
    *,
    input_text: str,
    emotion: str,
    menu_label: str,
    reason: str,
    selected_menu: str,
    ratio: str,
    ratio_reason: str,
) -> str:
    ratio_rule = ""
    if selected_menu in ("somaek", "koktail") and ratio:
        ratio_rule = (
            f'- 비율 표현은 반드시 "{ratio}"만 그대로 포함\n'
            f'- 왜 그 비율을 추천하는지 "{ratio_reason}" 의미를 자연스럽게 반드시 포함\n'
        )

    return f"""당신은 한국어 바텐더입니다.
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


def _build_llm_recommendation(
    *,
    input_text: str,
    emotion: str,
    reason: str,
    selected_menu: str,
    ratio: str,
    ratio_reason: str,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    menu_label = MENU_LABELS.get(selected_menu, "")
    if not selected_menu or not menu_label:
        raise ValueError("recommend_menu is missing or invalid.")

    prompt = _build_recommendation_prompt(
        input_text=input_text,
        emotion=emotion,
        menu_label=menu_label,
        reason=reason,
        selected_menu=selected_menu,
        ratio=ratio,
        ratio_reason=ratio_reason,
    )
    client = OpenAI(api_key=api_key)
    response = client.responses.create(model=MODEL, input=prompt)
    tts_text = _extract_response_text(response)
    if not tts_text:
        raise ValueError("Model returned empty recommendation text.")
    if menu_label not in tts_text:
        if selected_menu in ("somaek", "koktail") and ratio:
            return build_ratio_recommendation_prompt_text(selected_menu, ratio, ratio_reason)
        return build_recommendation_text(selected_menu, ratio, ratio_reason)
    if selected_menu in ("somaek", "koktail") and ratio and ratio not in tts_text:
        return build_ratio_recommendation_prompt_text(selected_menu, ratio, ratio_reason)
    return tts_text


def _handle_direct_order(
    state: GraphState,
    *,
    input_text: str,
    selected_menu: str,
    ratio: str,
    ratio_reason: str,
) -> GraphState:
    try:
        resolved_ratio, resolved_ratio_reason = _resolve_ratio(
            state=state,
            selected_menu=selected_menu,
            input_text=input_text,
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

    if selected_menu in ("somaek", "koktail") and not ratio and resolved_ratio_reason != "사용자가 직접 선택한 비율":
        return _build_state_update(
            state,
            tts_text=build_ratio_recommendation_prompt_text(selected_menu, resolved_ratio, resolved_ratio_reason),
            status="processing",
            selected_menu=selected_menu,
            ratio=resolved_ratio,
            ratio_reason=resolved_ratio_reason,
        )

    return _build_state_update(
        state,
        tts_text=build_order_confirmation_text(selected_menu, resolved_ratio),
        status="success",
        selected_menu=selected_menu,
        ratio=resolved_ratio,
        ratio_reason=resolved_ratio_reason,
    )


def _handle_recommendation_flow(
    state: GraphState,
    *,
    input_text: str,
    ratio: str,
    ratio_reason: str,
) -> GraphState:
    selected_menu = _resolve_selected_menu(state, input_text)
    try:
        resolved_ratio, resolved_ratio_reason = _resolve_ratio(
            state=state,
            selected_menu=selected_menu,
            input_text=input_text,
            ratio=ratio,
            ratio_reason=ratio_reason,
        )
        tts_text = _build_llm_recommendation(
            input_text=input_text,
            emotion=state.get("emotion", "neutral"),
            reason=state.get("reason", ""),
            selected_menu=selected_menu,
            ratio=resolved_ratio,
            ratio_reason=resolved_ratio_reason,
        )
    except Exception as exc:
        return fallback_recommendation_text(
            error_message=str(exc),
            selected_menu=selected_menu,
            ratio=ratio,
            ratio_reason=ratio_reason,
        )

    return _build_state_update(
        state,
        tts_text=tts_text,
        status="processing",
        selected_menu=selected_menu,
        ratio=resolved_ratio,
        ratio_reason=resolved_ratio_reason,
    )


def classify_node(state: GraphState) -> GraphState:
    input_text = state.get("input_text", "")

    if not input_text or not input_text.strip():
        raise ValueError("input_text is required and cannot be empty.")

    is_direct_order, selected_menu = _is_direct_order(input_text)
    ratio = state.get("ratio", "")
    ratio_reason = state.get("ratio_reason", "")

    if is_direct_order:
        return _handle_direct_order(
            state,
            input_text=input_text,
            selected_menu=selected_menu,
            ratio=ratio,
            ratio_reason=ratio_reason,
        )

    return _handle_recommendation_flow(
        state,
        input_text=input_text,
        ratio=ratio,
        ratio_reason=ratio_reason,
    )
