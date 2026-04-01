import json
import logging
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from common import MODEL

load_dotenv()
logger = logging.getLogger(__name__)

TOTAL_VOLUME_ML = 300
ALLOWED_RATIOS = ("1:1", "1:2", "1:5")
RATIO_MAP = {
    "1:1": (1, 1),
    "1:2": (1, 2),
    "1:5": (1, 5),
}
RATIO_INGREDIENTS = {
    "somaek": ("소주", "맥주"),
    "koktail": ("소주", "주스"),
}


def extract_ratio_from_text(text: str) -> str:
    normalized = (text or "").replace(" ", "")
    match = re.search(r"(1)\s*[:：대]\s*(1|2|5)", normalized)
    if not match:
        return ""
    ratio = f"{match.group(1)}:{match.group(2)}"
    return ratio if ratio in RATIO_MAP else ""


def select_ratio_with_llm(menu: str, text: str, emotion: str, user_profile: dict | None) -> tuple[str, str]:
    if menu not in RATIO_INGREDIENTS:
        return "", "비율 선택이 필요 없는 메뉴"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Ratio selection LLM cannot run.")

    ingredient_desc = ":".join(RATIO_INGREDIENTS[menu])
    user_profile_json = json.dumps(user_profile or {}, ensure_ascii=False)
    prompt = f"""
        사용자 맞춤 음료 비율 선택기입니다.
        메뉴 종류와 사용자 표현을 보고 허용된 비율 중 하나만 선택하세요.

        메뉴: {menu}
        재료 비율 의미: {ingredient_desc}
        사용자 발화: {text}
        사용자 감정: {emotion}
        사용자 프로필(JSON): {user_profile_json}
        허용 비율: {", ".join(ALLOWED_RATIOS)}

        출력은 반드시 아래 JSON 한 개만 반환하세요.
        {{
        "ratio": "1:1",
        "reason": "짧은 이유"
        }}
"""
    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(model=MODEL, input=prompt)
        raw = (getattr(response, "output_text", "") or "").strip()
        data = json.loads(raw)
        ratio = str(data.get("ratio", "")).strip()
        reason = str(data.get("reason", "")).strip() or "LLM이 선택한 비율"
        if ratio in RATIO_MAP:
            return ratio, reason
        raise ValueError(f"Invalid ratio from LLM response: {raw!r}")
    except Exception as exc:
        logger.error("Ratio selection LLM failed: %s", exc)
        raise


def _calc_pair_recipe(first_name: str, second_name: str, ratio: str) -> dict[str, int]:
    first_ratio, second_ratio = RATIO_MAP[ratio]
    total_ratio = first_ratio + second_ratio
    first_ml = round(TOTAL_VOLUME_ML * first_ratio / total_ratio)
    second_ml = TOTAL_VOLUME_ML - first_ml
    return {first_name: first_ml, second_name: second_ml}


def build_recipe(menu: str, ratio: str) -> dict[str, int]:
    if menu == "soju":
        return {"soju": 50}
    if menu == "beer":
        return {"beer": 300}
    if menu == "juice":
        return {"juice": 350}
    if menu == "somaek":
        return _calc_pair_recipe("soju", "beer", ratio or "1:2")
    if menu == "koktail":
        return _calc_pair_recipe("soju", "juice", ratio or "1:2")
    return {}
