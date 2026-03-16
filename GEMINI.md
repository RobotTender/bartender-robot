# GEMINI.md - Persistent Mandates & Working Log (robot-llm-combine-junsung)

## ⚠️ SCOPE BOUNDARY (CRITICAL)
This project is part of a larger workspace. **You MUST ONLY modify files within `robot/src/bartender_test/`, `llm/`, `detection/` and this `GEMINI.md` file.** Your expertise is strictly limited to the "Robotender" bartender robot logic and its integration with LLM/Detection modules.

## Working Context (March 16, 2026 - Branch Reset)
- **Current Branch:** `robot-llm-combine-junsung`
- **Architecture Status:** 
    - **Persistent Node + Service** structure is the standard.
    - `robotender_action` (`action_node.py`): Handles complex motions (Pour, Warmup) with Snap Recovery.
    - `robotender_gripper` (`gripper.py`): Controls the Robotis RH-P12-RN via Autonomous DRL Injection (One-Shot).
    - `robotender_monitor` (`monitor.py`): Unified telemetry (Joints, Pose, Force, Weight).
    - `robotender_trigger` (`trigger.py`): Spacebar listener for manual SNAP/Recovery.
    - **LLM Integration:** `llm/web/order_engine/` contains the logic for order classification and making.

## Core Mandates
- **Architecture Priority:** Maintain the **Persistent Node + Service** structure. Do not revert to one-shot scripts for core operations.
- **Gripper Implementation:** The gripper **MUST** be controlled via autonomous DRL injection (One-Shot Modbus) from the `robotender_gripper` node.
- **Auto-Rebuild:** Every time a code change is made to `bartender_test`, you **MUST** automatically run:
  `colcon build --symlink-install --packages-select bartender_test && source install/setup.bash`
- **Safety First:** Always use `movesj` or `movej` with cautious velocities (default 60, max 250 for recovery) to prevent hardware collisions.
- **Commit Rule:** Commit every change immediately.
- **Python 3 DRL:** Always use `bytes(modbus_send_make([...list...]))`.
- **Camera Logs:** Hidden by default. Ask user to unhide at session start if needed.

## Project Organization (Node-Service Structure)
- `robot/src/bartender_test/`: The primary ROS 2 Python package.
    - `bartender_test/action_node.py` (**`robotender_action`**):
        - Service: `/dsr01/pour/start` (Pouring with Snap Recovery).
        - Service: `/dsr01/warmup/start` (Pose verification).
        - Subscription: `pouring_trigger` (Manual SNAP).
    - `bartender_test/gripper.py` (**`robotender_gripper`**):
        - Service: `/dsr01/gripper/open` (Autonomous DRL injection).
        - Service: `/dsr01/gripper/close` (Autonomous DRL injection).
    - `bartender_test/monitor.py` (**`robotender_monitor`**): Real-time display of joints, TCP pose, and weight.
    - `bartender_test/trigger.py` (**`robotender_trigger`**): Physical interrupt node (Spacebar listener).
    - `bartender_test/defines.py`: Central repository for Poses (`posj`, `posx`) and physical constants.

## Hardware Specification
- **Robot:** Doosan Robotics dsr01 e0509 (6-DOF)
- **Gripper:** Robotis RH-P12-RN (Force-controlled)
- **Bottle:** Standard Soju (500ml / ~500g)
- **Gripper Force:** 400 (Standard), 800 (Closing/Grasp)

## Operational Commands
### Virtual Mode (Simulation with RViz)
```bash
ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=virtual model:=e0509 gui:=true gripper:=rh_p12_rn object:=bottle
```

### Real Mode (Physical Robot)
```bash
ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=real host:=110.120.1.68 model:=e0509
```

### Full Stack (Launch All Nodes)
```bash
python3 scripts/start_order_stack.py --with-bringup --bringup-cmd "ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=real host:=110.120.1.68 model:=e0509"
```

## Snapping/Recovery Status
- SPACEBAR trigger implemented in `trigger.py` -> `action_node.py`.
- 3Hz joint-space recording buffer and high-speed backtracking (250 vel/acc) for safe recovery after pouring interruption.
