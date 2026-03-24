import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from common import MODEL, MENU_LABELS, detect_menu_from_text,build_order_confirmation_text
from state import GraphState
load_dotenv()
logger = logging.getLogger(__name__)

CONFIRM_CUES = ("응", "어", "그래", "좋아", "맞아", "그걸로", "그거", "줘", "주세요", "할게", "할게요", "부탁")
REJECT_CUES = ("아니", "말고", "싫어", "괜찮아", "됐어")


def _resolve_with_llm(text, recommend_menu) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is missing.")
        logger.info("_resolve_with_llm result=%r reason=%r text=%r context_menu=%r", "", "missing_api_key", text, context_menu)
        return ""

    prompt = f"""
        {text}는 바텐더가 직전에 추천한 메뉴를 듣고 사용자가 바텐더에게 한 응답입니다.
        
    
        사용자가 바텐더에게 한 말: "{text}"
        바텐더가 직전에 추천한 메뉴: {recommend_menu}
        바텐더가 직전에 추천한 메뉴를 듣고 사용자가 바텐더에게 한말입니다.
        이를 분석하여, 사용자가 주문하려는 메뉴를 아래 중 하나로만 답하세요.
        선택지: {", ".join(MENU_LABELS.keys())}
        판별 불가 또는 부정이면 빈 문자열만 출력하세요. 다른 말은 출력하지 마세요.
    """

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(model=MODEL, input=prompt)
        raw_result = (getattr(response, "output_text", "") or "").strip()
        result = raw_result if raw_result in MENU_LABELS else ""
        return result
    except Exception as exc:
        logger.error("LLM resolve failed: %s", exc)
        logger.info("_resolve_with_llm result=%r reason=%r text=%r context_menu=%r", "", "exception", text, context_menu)
        return ""


def make_order_node(state: GraphState) -> GraphState:
    """
    사용자 발화를 분석해 주문 메뉴 코드 반환.
    """
    text = str(state.get("input_text", "") or "").strip()
    recommend_menu = state.get("recommend_menu", "")
    retry = state.get("retry", False)
    
    selected_menu = ""
    status = "retry"
    route = "unknown"
    llm_used = False
    llm_reason = ""

    # 1. Direct Keyword Check
    detected = detect_menu_from_text(text)
    if detected:
        selected_menu = detected
        status = "success"
        route = "direct_keyword_in_make"
        tts_text = build_order_confirmation_text(selected_menu)
    
    # 2. Confirm Cue Check
    elif any(cue in text for cue in CONFIRM_CUES):
        if recommend_menu in MENU_LABELS:
            selected_menu = recommend_menu
            status = "success"
            route = "confirm_recommendation"
            tts_text = build_order_confirmation_text(selected_menu)
        else:
            status = "retry"
            route = "confirm_without_context"
            tts_text = "네, 무엇을 드릴까요?"

    # 3. Reject Cue Check
    elif any(cue in text for cue in REJECT_CUES):
        status = "retry"
        route = "reject_cue"
        tts_text = "알겠습니다. 다른 것은 어떠신가요? 소주, 맥주, 주스, 소맥이 있습니다."

    # 4. LLM Resolve
    else:
        llm_match = _resolve_with_llm(text, recommend_menu)
        llm_used = True
        if llm_match in MENU_LABELS:
            selected_menu = llm_match
            status = "success"
            route = "llm_resolve_success"
            llm_reason = f"LLM matched to {selected_menu}"
            tts_text = build_order_confirmation_text(selected_menu)
        else:
            status = "retry"
            route = "llm_resolve_fail"
            llm_reason = "LLM could not resolve menu"
            tts_text = "죄송합니다. 잘 이해하지 못했습니다. 다시 말씀해 주시겠어요?"

    if status == "retry" and retry:
        status = "error" # Too many retries
        tts_text = "죄송합니다. 주문을 도와드리기 어렵네요. 나중에 다시 시도해주세요."

    return {
        **state,
        "retry": True if status == "retry" else retry,
        "status": status,
        "selected_menu": selected_menu,
        "tts_text": tts_text,
        "llm_text": tts_text,
        "route": route,
        "llm_used": llm_used,
        "llm_reason": llm_reason
    }
