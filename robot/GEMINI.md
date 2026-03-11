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
- **Current Challenge:** Implementing a high-acceleration "snap" back to upright position (CHEERS_POSE) to minimize over-pouring.

## Implementation Phases
- **Phase 1-4 (Sensing & Basic Control):** [COMPLETED]
- **Phase 5 (Unified Telemetry):** [COMPLETED]
- **Phase 6 (Snapping / Recovery):** [IN PROGRESS - SNAP ACTIVE]
    - **Step 1: Keyboard Trigger:** [COMPLETED] Non-blocking listener using background thread.
    - **Step 2: Interrupt Logic:** [COMPLETED] Call `motion/move_stop(DR_SSTOP)` via dedicated listener node.
    - **Step 3: High-Acceleration Return:** [COMPLETED] Execute `movej(CHEERS_POSE)` at 250 deg/s / 250 deg/s² upon interrupt.
    - **Step 4: Path Revision:** [TODO] Revise snap motion to follow the reverse path of the pouring trajectory to avoid collisions.
    - **Step 5: Validation:** [PENDING] Verify stop-to-upright transition is <200ms.

## Later / Future Work (Predictive Flow Modeling)
- **Goal:** Achieve ±2g accuracy by predicting "In-Flight" volume to trigger early recovery.
- **Challenge:** Liquid remains in the air after the robot stops tilting; must stop *before* target is reached.
- **Steps:**
    - **Flow Calibration:** Measure flow rate (g/s) at various tilt offsets and fullness levels.
    - **Flow Model:** Derive Q(delta_theta, W) = flow rate prediction.
    - **In-Flight Modeling:** Calculate fall time (t_fall) + robot response latency.
    - **Predictive Stop:** Implement stop threshold: Poured >= Target - (Q * t_total).

## Commands
- **Dynamic Pouring:** `ros2 run bartender_test action pour auto <target_grams>`
- **Snapping Test:** `ros2 run bartender_test action snap_test` (Manual Spacebar trigger)
- **Telemetry:** `ros2 run bartender_test monitor`
- **Calibration Tool:** `ros2 run bartender_test model`

### Reference Models
- **First Drop Angle:** Ry = -0.0433 * Weight + 102.3888 (Phase 3)
- **In-Flight Weight (Initial):** Estimated at 2g-5g depending on flow rate.
