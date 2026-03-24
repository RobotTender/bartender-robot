from typing import TypedDict

class GraphState(TypedDict, total=False):
    input_text: str
    status: str  # 상태 success, processing, retry, error, canceled
    emotion: str # AI 분석 사용자 감정
    recommend_menu: str # AI 추천 메뉴 
    reason: str # 이유
    retry: bool # 반복 여부
    selected_menu: str # 주문 메뉴
    selected_menu_label: str # 주문 메뉴 한글명
    tts_text: str
    llm_text: str
    recipe: dict
    route: str
    llm_used: bool
    llm_reason: str
    events: list[dict]
