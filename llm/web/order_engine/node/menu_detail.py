from common import build_recipe
from state import GraphState


def menu_detail_node(state: GraphState) -> GraphState:
    # 메뉴 레시피(사용자에 따른 비율 산정을 할경우 LLM 쓸 가능성 있음)
    menu = state.get("selected_menu", "")
    recipe = build_recipe(menu)
    return {
        **state,
        "drinks": menu,
        "recipe": recipe,
    }
