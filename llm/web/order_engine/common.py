import os

MODEL = os.environ.get("VOICE_ORDER_MODEL", "gpt-5-nano")

MENU_LABELS = {
    "soju": "소주",
    "beer": "맥주",
    "juice": "주스",
    "somaek": "소맥",
}

MENU_ALIASES = {
    "soju": ("소주",),
    "beer": ("맥주", "비어"),
    "juice": ("주스", "쥬스"),
    "somaek": ("소맥", "소주맥주", "소주 맥주"),
}

RECIPE_BY_MENU = {
    "soju": {"soju": 50},
    "beer": {"beer": 100},
    "juice": {"juice": 100},
    "somaek": {"soju": 60, "beer": 140},
}

ORDER_CUES = ("줘", "주세요", "주문", "말아", "말아줘", "한잔", "한 잔", "주라", "내놔")
CONFIRM_CUES = ("응", "어", "그래", "좋아", "맞아", "그걸로", "그거", "할게", "할게요", "부탁")
REJECT_CUES = ("아니", "말고", "싫어", "괜찮아", "됐어")
NEGATIVE_CONFIRM_BLOCKERS = (
    "안 좋아",
    "안좋아",
    "좋지 않아",
    "좋지않아",
    "기분이 안 좋아",
    "기분이 안좋아",
)


def build_order_confirmation_text(selected_menu: str) -> str:
    menu_label = MENU_LABELS.get(selected_menu, "해당 메뉴")
    return f"{menu_label} 주문 확인했습니다. 제조시작합니다."


def detect_menu_from_text(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    for menu_code, aliases in MENU_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return menu_code
    return ""


def build_recipe(selected_menu: str) -> dict:
    return dict(RECIPE_BY_MENU.get(str(selected_menu or ""), {}))
