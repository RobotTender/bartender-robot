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

def detect_menu_from_text(text: str) -> str:
    for menu_code, aliases in MENU_ALIASES.items():
        if any(alias in text for alias in aliases):
            return menu_code
    return ""
