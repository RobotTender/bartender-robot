## Working Context (March 24, 2026 - Developer UI Migration)
- **Current Branch:** `robot-llm-detection-ui-combine-junsung`
- **Architecture Status:** 
    - **UI Reorganized:** Moved to `./ui/frontend` (Scripts) and `./ui/backend` (Logic).
    - **Developer UI:** Path-refactored to find assets in `./ui/assets` and backend in `./ui/backend`.
    - **Manager Integration:** Ready for manual ROS 2 bridge implementation.

## Achievements (March 24, 2026)
- **Branch Renamed:** Switched to `robot-llm-detection-ui-combine-junsung`.
- **UI Structure Refactored:** 
    - `./ui/frontend/`: Contains `developer_frontend.py`, `user_frontend.py`.
    - `./llm/web/order_engine/`: Unified voice/LLM logic (STT, TTS, LangGraph).
    - `./ui/backend/`: Contains `task_backend_node.py` and `bartender_action/`.
    - `./ui/assets/`: Contains `.ui` layout files.
- **Path Refactoring:**
    - `developer_frontend.py`: Updated to correctly load `RobotBackend`, `.ui` files, and use `llm/web/order_engine` for TTS.
    - `user_frontend.py`: Updated `sys.path` to find the `llm/web/order_engine` module.
    - `task_backend_node.py`: Migrated voice order processing to `llm/web/order_engine`.
- **Cleanup:** Removed redundant `ui/frontend/order_integration/` directory in favor of `llm/web/order_engine/`.
- **Recipe Sync:** Updated `llm/web/order_engine/common.py` to match full recipe list (Soju, Beer, Juice, Somaek).
- **Environment Template:** Created `./ui/.env` for API keys and configuration.

## Developer Frontend Status & Next Steps
The `developer_frontend.py` is now structurally ready to run but currently relies on the migrated `RobotBackend` class. 

### Migration Tasks for VS Code:
1. **ROS 2 Topic Mapping:** Map the UI's status indicators to the local topics:
    - Manager Mode: `/dsr01/robotender_manager/mode`
    - Robot State: `/dsr01/msg/robot_state` (or equivalent)
2. **Action Triggering:** Connect the UI "Start" buttons to the `/dsr01/robotender_manager/pick_bottle` (and other) services.
3. **Log Capture:** The UI captures `stdout`. Ensure the local manager nodes are logging to stdout or use the ROS 2 log subscriber already present in `developer_frontend.py`.

## Core Mandates
- **Orchestration Pattern:** The Manager node MUST handle the decision-making and sequence logic.
- **Real-time Control:** Mode overrides MUST be handled via topics to allow flexibility without restarting.
- **Safety:** Always return to `POSJ_HOME` or a safe state before halting an auto-sequence.

## Operational Commands (VSCode / Terminal)

### 1. Developer UI (Frontend)
Run this in a dedicated terminal to monitor and control the robot:
```bash
clear && ./.venv/bin/python3 ui/frontend/developer_frontend.py
```

### 2. Full System Stack (Bringup + Logic + Web)
Run this to start the complete bartender robot system:
```bash
clear && ./scripts/cleanup.sh && python3 scripts/start_order_stack.py --with-bringup
```

### 3. User Web UI (Standalone - Optional)
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/ui/frontend
python3 ui/frontend/user_frontend.py --enabled 1
```

## Hardware Specification
- **Robot:** Doosan Robotics dsr01 e0509 (6-DOF)
- **Gripper:** Robotis RH-P12-RN (Force-controlled)
- **Bottle:** Standard Soju, Beer, Juice
