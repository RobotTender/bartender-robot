#!/usr/bin/env python3
import json
import logging
import os
import shlex
import subprocess
import threading

logger = logging.getLogger(__name__)

ROBOT_ORDER_TOPIC = "/bartender/order_detail"


def _build_order_payload(payload: dict) -> str:
    order_payload = {
        "drinks": payload.get("drinks") or payload.get("selected_menu") or "",
        "recipe": payload.get("recipe", {}),
    }
    return json.dumps(order_payload, ensure_ascii=False, separators=(",", ":"))


def _publish_robot_order(payload: dict) -> None:
    order_payload = _build_order_payload(payload)
    msg = {"data": order_payload}
    full_cmd = (
        "source /opt/ros/jazzy/setup.bash && "
        f"ros2 topic pub --once {ROBOT_ORDER_TOPIC} std_msgs/msg/String "
        f"{shlex.quote(json.dumps(msg, ensure_ascii=False, separators=(',', ':')))}"
    )

    try:
        subprocess.run(["bash", "-lc", full_cmd], check=True, env=os.environ.copy())
    except Exception:
        logger.exception("Failed to publish robot order topic. topic=%s payload=%s", ROBOT_ORDER_TOPIC, order_payload)


def _start_robot_topic_publish(payload: dict) -> None:
    threading.Thread(target=_publish_robot_order, args=(payload,), daemon=True).start()
