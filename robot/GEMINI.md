# Robotics Project: Robotender (Hand-off Documentation)

## ⚠️ SCOPE BOUNDARY (CRITICAL)
This project is part of a larger workspace. **You MUST ONLY modify files within `src/bartender_test/` and this `GEMINI.md` file.** Do not touch other ROS 2 packages, system configurations, or root-level files unless explicitly instructed by the user. Your expertise is strictly limited to the "Robotender" bartender robot logic.

## Core Mandates
- **Auto-Rebuild:** Every time a code change is made to `bartender_test`, you **MUST** automatically run:
  `colcon build --symlink-install --packages-select bartender_test && source install/setup.bash`
- **Validation:** Always verify that the build succeeds and the ROS nodes can start before reporting completion.
- **Safety First:** Always use `movesj` or `movej` with cautious velocities (default 60, max 250 for recovery) to prevent hardware collisions.

## Project Organization
- `src/bartender_test/`: The primary ROS 2 Python package.
    - `bartender_test/action.py`: **Main Entry Point.** Contains pouring logic, trajectory definitions, and the "Snapping" recovery system.
    - `bartender_test/monitor.py`: **Telemetry Hub.** Real-time display of joint angles, TCP pose, motor torques, and weight estimation.
    - `bartender_test/trigger.py`: **Interrupt Node.** Listens for Spacebar input to trigger immediate stop and recovery.
    - `bartender_test/defines.py`: Central repository for Poses (`posj`, `posx`) and physical constants.
    - `bartender_test/gripper_controller.py`: Helper class for the Robotis RH-P12-RN gripper.
    - `bartender_test/pose.py`: Utility for moving to specific named poses.

## Hardware Specification
- **Robot:** Doosan Robotics dsr01 e0509 (6-DOF)
- **Gripper:** Robotis RH-P12-RN (Force-controlled)
- **Bottle:** Standard Soju (500ml / ~500g)
- **Gripper Force:** 250 (Recommended for Soju bottle stability)

## Implementation Status (Phase 6: Snapping/Recovery)
We have successfully implemented the "Snapping" mechanism to minimize over-pouring:
1.  **Non-Blocking Trigger:** `trigger.py` publishes to `/dsr01/pouring_trigger`.
2.  **Path Recording:** `action.py` uses a background thread to record Joint-Space (`posj`) points at 3Hz during pouring.
3.  **Interrupt Logic:** Upon trigger, the robot immediately executes `move_stop(DR_SSTOP)`.
4.  **Reverse Recovery:** The system calculates a thinned backtrack path (max ~4 points) from the recorded history and executes a high-speed `movesj` (250 deg/s) back to the "Cheers" (safe) position.

## How to Use
1.  **Build:**
    ```bash
    colcon build --symlink-install --packages-select bartender_test
    source install/setup.bash
    ```
2.  **Monitor (Window 1):**
    ```bash
    ros2 run bartender_test monitor
    # Press 'T' to tare weight when bottle is held vertically.
    ```
3.  **Trigger (Window 2):**
    ```bash
    ros2 run bartender_test trigger
    # Press SPACEBAR to stop flow.
    ```
4.  **Action (Window 3):**
    ```bash
    ros2 run bartender_test action pour 1
    # Robot will move to contact -> pour. Use Trigger to snap back.
    ```
