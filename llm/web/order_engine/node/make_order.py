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

    1. 명시적 거절 + 메뉴 없음 → ""
    2. 텍스트에 메뉴가 있으면 해당 메뉴 코드 반환
    3. 긍정 반응 + context_menu 있으면 context_menu 반환
    4. 모호한 표현 → LLM으로 판별
    - 판별 불가 시 "" 반환
    """
    status = state.get("status")
    text = str(state.get("input_text", "") or "").strip()
    detected = detect_menu_from_text(text)
    recommend_menu = detected if detected else MENU_LABELS.get(state.get("recommend_menu"))
    retry = state.get("retry")
    selected_menu = state.get("selected_menu", "")
    

    if any(cue in text for cue in CONFIRM_CUES):
        selected_menu = recommend_menu
        tts_text = build_order_confirmation_text(selected_menu)
    
    elif any(cue in text for cue in REJECT_CUES):
        if retry:
            status = "end"
            tts_text = "죄송합니다. 이해하지 못했습니다."
            
        else:
            retry=True
            tts_text = "어떤 것을 그럼 원하시나요? 저희는 소주, 맥주, 소맥이 준비 되어있습니다."
 
    else:
        selected_menu = _resolve_with_llm(text, recommend_menu)
        if not selected_menu:
            if retry:
                status = "end"
                tts_text = "다시 한번 말씀해주시겠어요?"
                
            retry = True
            tts_text = "다시 한번 말씀해주시겠어요?"
        else:
            tts_text = build_order_confirmation_text(selected_menu)
            
    return {
        **state,
        "retry": retry,
        "status": status,
        "selected_menu":selected_menu,
        "tts_text":tts_text
    }
