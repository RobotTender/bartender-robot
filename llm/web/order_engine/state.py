from typing import TypedDict

class GraphState(TypedDict, total=False):
    input_text: str
    status: str  # 상태 success or processing
    emotion: str # AI 분석 사용자 감정
    user_profile: dict
    recommend_menu: str # AI 추천 메뉴 
    reason: str # 이유
    retry: bool # 반복횟수
    selected_menu: str # 주문 메뉴
    drinks: str
    tts_text: str
    ratio: str
    ratio_reason: str
    recipe: dict[str]
    
