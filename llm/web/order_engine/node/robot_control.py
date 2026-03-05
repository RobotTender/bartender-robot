from typing import Any, Mapping


def robot_control_node(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **state,
        "robot_action": "dispatch_order",
    }
