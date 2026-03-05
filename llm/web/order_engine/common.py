MODEL = "gpt-5-nano"

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

def build_order_confirmation_text(selected_menu: str) -> str:
    menu_label = MENU_LABELS.get(selected_menu, "해당 메뉴")
    return f"{menu_label} 주문 확인했습니다. 제조시작합니다."


def detect_menu_from_text(text: str) -> str:
    for menu_code, aliases in MENU_ALIASES.items():
        if any(alias in text for alias in aliases):
            return menu_code
    return ""
