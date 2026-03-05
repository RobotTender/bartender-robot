from state import GraphState


def menu_detail_node(state: GraphState) -> GraphState:
    # 메뉴 레시피(사용자에 따른 비율 산정을 할경우 LLM 쓸 가능성 있음)
    menu = state.get("selected_menu")
    if menu =="soju":
        recipe = {"soju": 50}
    elif menu == "beer":
        recipe = {"beer":200}
    elif menu == "somaek":
        recipe = {"soju": 60, "beer":140}
    return {
        **state,
        "recipe" : recipe
    }
