from typing import TypedDict

from langgraph.graph import END, StateGraph

from node.classify_order import classify_node
from node.make_order import make_order_node
from node.menu_detail import menu_detail_node
from state import GraphState

    
def route_entry(state: GraphState) -> str:
    status = state.get("status")
    return {
        "init": "classify_order",
        "processing": "makeorder",
        "success": "menu_detail"
    }.get(status, END)  # 기본값

def route_to_detail(state: GraphState) -> str:
    if state.get("status") == "success":
        return "menu_detail"
    return END


def create_graph_flow():
    graph_builder = StateGraph(GraphState)

    graph_builder.add_node("classify_order", classify_node)
    graph_builder.add_node("makeorder", make_order_node)
    graph_builder.add_node("menu_detail", menu_detail_node)

    graph_builder.set_conditional_entry_point(route_entry)
    graph_builder.add_conditional_edges("classify_order", route_to_detail)
    graph_builder.add_conditional_edges("makeorder", route_to_detail)
    graph_builder.add_edge("menu_detail", END)

    return graph_builder.compile()
