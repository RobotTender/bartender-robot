import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from common import MODEL, MENU_LABELS, detect_menu_from_text,build_order_confirmation_text
from state import GraphState
load_dotenv()
logger = logging.getLogger(__name__)

ORDER_CUES = ("줘", "주세요", "주문", "말아", "말아줘", "한잔", "한 잔", "주라", "내놔")

def _is_direct_order(input_text: str) -> tuple[bool,str]:
    
    menu_code = detect_menu_from_text(input_text)
    has_order_cue = any(cue in input_text for cue in ORDER_CUES)
    if menu_code and has_order_cue:
        return True, menu_code
    
    return False, ""


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


def classify_node(state: GraphState) -> GraphState:
    
    input_text=state.get("input_text", "")
    
    if not input_text or not input_text.strip():
        raise ValueError("input_text is required and cannot be empty.")

    is_direct_order, selected_menu = _is_direct_order(input_text)
    

    if is_direct_order:
        tts_text = build_order_confirmation_text(selected_menu)
    
    else:
    
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return fallback_recommendation_text(error_message="OPENAI_API_KEY is missing.")
    
        recommend_menu = state.get("recommend_menu", "") or state.get("recommand_menu", "")
        menu_label = MENU_LABELS.get(recommend_menu, "")
        emotion = state.get("emotion", "neutral")
        reason = state.get("reason", "")
        selected_menu = recommend_menu
        
        prompt = f"""당신은 한국어 바텐더입니다.
            아래 정보를 종합해 사용자에게 전달할 추천 멘트 1문장을 작성하세요.
            - 사용자 발화: {input_text}
            - 감정: {emotion}
            - 추천 메뉴: {menu_label}
            - 추천 이유: {reason}

            규칙:
            - 한국어 존댓말 1문장만 출력
            - 25~55자 이내
            - 과장 없이 자연스럽게
            - 메뉴명을 반드시 포함
        """

        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=MODEL,
                input=prompt,
            )
            tts_text = _extract_response_text(response)
            if not tts_text:
                return fallback_recommendation_text(error_message="Model returned empty recommendation text.")
        except Exception as exc:
            return fallback_recommendation_text(error_message=str(exc))
        
    
    return {
                **state,
                "tts_text":tts_text,
                "status": "success" if is_direct_order else "processing",
                "selected_menu": selected_menu
    }
