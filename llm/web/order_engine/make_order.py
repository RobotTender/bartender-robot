import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from .common import MODEL, MENU_LABELS, detect_menu_from_text

load_dotenv()
logger = logging.getLogger(__name__)

CONFIRM_CUES = ("응", "어", "그래", "좋아", "맞아", "그걸로", "그거", "줘", "주세요", "할게", "할게요", "부탁")
REJECT_CUES = ("아니", "말고", "싫어", "괜찮아", "됐어")


def _resolve_with_llm(text: str, context_menu: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is missing.")
        return ""

    context_label = MENU_LABELS.get(context_menu, "없음")
    prompt = f"""사용자가 바텐더에게 한 말: "{text}"
        바텐더가 직전에 추천한 메뉴: {context_label}

        사용자가 주문하려는 메뉴를 아래 중 하나로만 답하세요.
        선택지: {", ".join(MENU_LABELS.keys())}
        판별 불가 또는 부정이면 빈 문자열만 출력하세요. 다른 말은 출력하지 마세요.
    """

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(model=MODEL, input=prompt)
        result = (getattr(response, "output_text", "") or "").strip()
        return result if result in MENU_LABELS else ""
    except Exception as exc:
        logger.error("LLM resolve failed: %s", exc)
        return ""


def parse_reply(input_text: str, context_menu: str = "") -> str:
    """
    사용자 발화를 분석해 주문 메뉴 코드 반환.

    1. 명시적 거절 + 메뉴 없음 → ""
    2. 텍스트에 메뉴가 있으면 해당 메뉴 코드 반환
    3. 긍정 반응 + context_menu 있으면 context_menu 반환
    4. 모호한 표현 → LLM으로 판별
    - 판별 불가 시 "" 반환
    """
    text = (input_text or "").strip()

    detected = detect_menu_from_text(text)

    if any(cue in text for cue in REJECT_CUES) and not detected:
        return ""

    if detected:
        return detected

    if context_menu and any(cue in text for cue in CONFIRM_CUES):
        return context_menu

    return _resolve_with_llm(text, context_menu)
