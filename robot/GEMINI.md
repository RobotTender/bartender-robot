# Robotics Project: Robotender

## Core Mandates
- **Auto-Rebuild:** Every time a code change is made to a ROS 2 package (e.g., `bartender_test`), the agent **MUST** automatically run `colcon build --symlink-install --packages-select <package_name>` and `source install/setup.bash` to ensure the workspace is in a valid state.
- **Validation:** Always verify that the build succeeds before reporting completion.

## Hardware
- **Robot:** Doosan Robotics dsr01 e0509
- **Gripper:** Robotis RH-P12-RN
- **Bottle (Standard):** "Soju" (Max Volume: 500ml / 500g)
- **Gripper Force (Soju):** 250 (Recommended for firm grasp during weighing)

## Project Status: Phase 6 - Snapping (Recovery)
- **Goal:** Achieve rapid recovery to stop pouring flow immediately upon command.
- **Current Challenge:** Minimizing over-pouring by reversing the exact pouring path at high acceleration.

## Implementation Phases
- **Phase 1-4 (Sensing & Basic Control):** [COMPLETED]
- **Phase 5 (Unified Telemetry):** [COMPLETED]
- **Phase 6 (Snapping / Recovery):** [IN PROGRESS - IMPLEMENTATION COMPLETE]
    - **Step 1: Keyboard Trigger:** [COMPLETED] Non-blocking Spacebar listener using background thread.
    - **Step 2: Interrupt Logic:** [COMPLETED] Call `motion/move_stop(DR_SSTOP)` via dedicated listener node.
    - **Step 3: Path Recording:** [COMPLETED] 4Hz background Joint-Space (`posj`) recording via `joint_states` subscription.
    - **Step 4: Reverse Recovery:** [COMPLETED] Adaptive thinning (max ~10 points) + `movesj` at 250 deg/s / 250 deg/s² for high-speed backtracking.
    - **Step 5: Code Cleanup:** [COMPLETED] Removed obsolete calibration and predictive modeling nodes.
    - **Step 6: Validation:** [TODO] Perform final snapping tests with real robot and liquid to verify ±2g accuracy.

## Commands (DO NOT DELETE)
- **Real Mode:** `ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=real host:=110.120.1.59 model:=e0509`
- **Virtual Mode:** `ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=virtual model:=e0509 gripper:=rh_p12_rn object:=bottle`
- **Pouring & Snapping:** `ros2 run bartender_test action pour` (Spacebar to trigger snap)
- **Warmup Move:** `ros2 run bartender_test action warmup`
- **Telemetry:** `ros2 run bartender_test monitor`
- **Manual Trigger:** `ros2 run bartender_test trigger`
- **Gripper Control:** `ros2 run bartender_test gripper [open | close]`
- **Pose Utility:** `ros2 run bartender_test pose`
