import os
import re


MODEL = os.getenv("ORDER_LLM_MODEL", "gpt-4o-mini")

MENU_LABELS = {
    "soju": "소주",
    "beer": "맥주",
    "somaek": "소맥",
    "juice": "주스",
    "koktail": "칵테일",
}
MENU_ALIASES = {
    "soju": ("소주", "소쥬", "쏘주", "수주"),
    "beer": ("맥주", "비어"),
    "somaek": ("소맥", "소주맥주", "소주 맥주"),
    "juice": ("주스", "쥬스", "주수", "쥬수", "논알콜", "논알콜", "논 알콜", "논알코올", "술 없는 음료","무알콜", "무알코올"),
    "koktail": ("칵테일", "코크테일", "코크테일"),
}

RATIO_INGREDIENTS = {
    "somaek": ("소주", "맥주"),
    "koktail": ("소주", "주스"),
}

RATIO_PATTERN = re.compile(r"1\s*[:：대]\s*(1|2|5)")


def build_ratio_description(selected_menu: str, ratio: str) -> str:
    ingredients = RATIO_INGREDIENTS.get(selected_menu)
    if not ingredients or not ratio:
        return ""
    first_name, second_name = ingredients
    return f"{first_name}와 {second_name}를 {ratio} 비율로"


def build_recommendation_text(selected_menu: str, ratio: str = "", ratio_reason: str = "") -> str:
    menu_label = MENU_LABELS.get(selected_menu, "해당 메뉴")
    ratio_desc = build_ratio_description(selected_menu, ratio)
    if ratio_desc:
        reason_suffix = f" {ratio_reason}" if ratio_reason else ""
        return f"{menu_label}은 {ratio_desc} 추천드려요.{reason_suffix}".strip()
    return f"{menu_label} 추천드려요."


def build_ratio_recommendation_prompt_text(
    selected_menu: str,
    ratio: str = "",
    ratio_reason: str = "",
) -> str:
    recommendation = build_recommendation_text(selected_menu, ratio, ratio_reason)
    if not recommendation:
        return ""
    return f"{recommendation} 괜찮으실까요?"


def build_order_confirmation_text(selected_menu: str, ratio: str = "") -> str:
    menu_label = MENU_LABELS.get(selected_menu, "해당 메뉴")
    ratio_desc = build_ratio_description(selected_menu, ratio)
    if ratio_desc:
        return f"{menu_label} 주문 확인했습니다. {ratio_desc} 제조시작합니다."
    return f"{menu_label} 주문 확인했습니다. 제조시작합니다."


def _contains_any_alias(text: str, menu_code: str) -> bool:
    return any(alias in text for alias in MENU_ALIASES.get(menu_code, ()))


def _has_explicit_ratio(text: str) -> bool:
    return bool(RATIO_PATTERN.search(text or ""))


def detect_menu_from_text(text: str) -> str:
    content = text or ""

    # Prefer explicit composite-menu names over ingredient mentions.
    for menu_code in ("somaek", "koktail"):
        if _contains_any_alias(content, menu_code):
            return menu_code

    # If the user mentions an ingredient pair with a ratio, treat it as the composite menu.
    has_ratio = _has_explicit_ratio(content)
    if has_ratio and _contains_any_alias(content, "soju") and _contains_any_alias(content, "beer"):
        return "somaek"
    if has_ratio and _contains_any_alias(content, "soju") and _contains_any_alias(content, "juice"):
        return "koktail"

    for menu_code, aliases in MENU_ALIASES.items():
        if any(alias in text for alias in aliases):
            return menu_code
    return ""
