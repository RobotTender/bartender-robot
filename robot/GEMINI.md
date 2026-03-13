# Robotics Project: Robotender (Hand-off Documentation)

## ⚠️ SCOPE BOUNDARY (CRITICAL)
This project is part of a larger workspace. **You MUST ONLY modify files within `src/bartender_test/` and this `GEMINI.md` file.** Do not touch other ROS 2 packages, system configurations, or root-level files unless explicitly instructed by the user. Your expertise is strictly limited to the "Robotender" bartender robot logic.

## Core Mandates
- **Architecture Priority:** Maintain the **Persistent Node + Service** structure. Do not revert to one-shot scripts for core operations. 
- **Gripper Implementation:** The gripper **MUST** be controlled via autonomous DRL injection (One-Shot Modbus) from the `robotender_gripper` node. This avoids the need for manual background DRL scripts on the Teach Pendant.
- **Auto-Rebuild:** Every time a code change is made to `bartender_test`, you **MUST** automatically run:
  `colcon build --symlink-install --packages-select bartender_test && source install/setup.bash`
- **Safety First:** Always use `movesj` or `movej` with cautious velocities (default 60, max 250 for recovery) to prevent hardware collisions.

## Project Organization (Node-Service Structure)
- `src/bartender_test/`: The primary ROS 2 Python package.
    - `bartender_test/action_node.py` (**`robotender_action`**): Persistent brain node.
        - Service: `/dsr01/pour/start` (Pouring with Snap Recovery).
        - Service: `/dsr01/warmup/start` (Pose verification).
    - `bartender_test/gripper.py` (**`robotender_gripper`**): Persistent hardware driver.
        - Service: `/dsr01/gripper/open` (Autonomous DRL injection).
        - Service: `/dsr01/gripper/close` (Autonomous DRL injection).
    - `bartender_test/monitor.py` (**`robotender_monitor`**): Telemetry Hub. Real-time display of joints, TCP pose, and weight.
    - `bartender_test/trigger.py` (**`trigger_node`**): Physical interrupt node (Spacebar listener).
    - `bartender_test/defines.py`: Central repository for Poses (`posj`, `posx`) and physical constants.

## Hardware Specification
- **Robot:** Doosan Robotics dsr01 e0509 (6-DOF)
- **Gripper:** Robotis RH-P12-RN (Force-controlled)
- **Bottle:** Standard Soju (500ml / ~500g)
- **Gripper Force:** 400 (Standard for stable grasp)

## Implementation Status (Phase 7: Autonomous Persistent Services)
We have successfully migrated from ephemeral scripts to background drivers:
1.  **Launch-Ready:** All core services are started via `ros2 launch bartender_test robotender.launch.py`.
2.  **Autonomous Gripper:** `robotender_gripper` node injects full Modbus DRL code on-demand. No Teach Pendant interaction is required.
3.  **Snap Recovery:** Integrated into the `robotender_action` node. Uses 3Hz joint-space recording and tiered backtracking (max 3-4 points) for high-speed (250 deg/s) reversal.

## How to Use
1.  **Startup (Terminal 1):**
    ```bash
    ros2 launch bartender_test robotender.launch.py
    ```
2.  **Monitor (Terminal 2):**
    ```bash
    ros2 run bartender_test monitor
    # Provides live telemetry and 'T' to Tare.
    ```
3.  **Manual Trigger (Terminal 3):**
    ```bash
    ros2 run bartender_test trigger
    # Press SPACEBAR during pour to snap back.
    ```
4.  **Execute (Terminal 4):**
    ```bash
    # Warmup
    ros2 service call /dsr01/warmup/start std_srvs/srv/Trigger {}
    # Pour
    ros2 service call /dsr01/pour/start std_srvs/srv/Trigger {}
    ```
