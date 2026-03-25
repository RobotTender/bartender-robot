from ratio_utils import build_recipe, select_ratio_with_llm
from state import GraphState


def menu_detail_node(state: GraphState) -> GraphState:
    menu = state.get("selected_menu", "")
    text = state.get("input_text", "")
    emotion = state.get("emotion", "")
    user_profile = state.get("user_profile", {})
    ratio = state.get("ratio", "")
    ratio_reason = state.get("ratio_reason", "")

    if menu in ("somaek", "koktail") and not ratio:
        ratio, ratio_reason = select_ratio_with_llm(menu, text, emotion, user_profile)

    recipe = build_recipe(menu, ratio)

    return {
        **state,
        "drinks": menu,
        "ratio": ratio,
        "ratio_reason": ratio_reason,
        "recipe": recipe,
    }
