#!/usr/bin/env python3
import json
import logging
import shlex
import subprocess
import threading

logger = logging.getLogger(__name__)

COMMANDS = {
    "soju": "ros2 run bartender_test move_to home",
    "beer": "ros2 run bartender_test gripper_command open",
    "somaek": "ros2 run bartender_test measure_weight tare",
}


def run_ros_command(cmd: str) -> None:
    full_cmd = (
        "source /opt/ros/jazzy/setup.bash && "
        "source ~/bartender-robot/robot/install/setup.bash && "
        f"{cmd}"
    )
    subprocess.run(["bash", "-lc", full_cmd], check=True)


def _build_command_from_payload(payload: dict) -> str:
    selected_menu = str(payload.get("selected_menu", "") or "")

    base_cmd = COMMANDS.get(selected_menu)
    if not base_cmd:
        logger.warning("Unknown selected_menu for robot command: %r", selected_menu)
        return ""

    # payload 전체를 JSON으로 1개 인자 전달
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"{base_cmd} --recipe {shlex.quote(payload_json)}"


def _run_robot_job(payload: dict) -> None:
    cmd = _build_command_from_payload(payload)
    if not cmd:
        return

    try:
        run_ros_command(cmd)
    except Exception:
        logger.exception("Robot command execution failed. cmd=%r", cmd)


def _start_robot_job(payload: dict) -> None:
    threading.Thread(target=_run_robot_job, args=(payload,), daemon=True).start()







