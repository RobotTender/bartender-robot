## Working Context (March 20, 2026 - Manager-Pour Action Orchestration)
- **Current Branch:** `robot-llm-detection-combine-junsung`
- **Architecture Status:** 
    - **Full Manager-Orchestration** for Pick, Pour, and Place is now active.
    - `robotender_manager` (**`manager.py`**): Orchestrates the full sequence (Pick -> Pour -> Place).
    - `robotender_pour` (**`pour.py`**): **ROS 2 Action Server** (Vision-triggered pouring with spline motion).
    - **Node Isolation Standard:** All motion nodes (Pick, Pour, Place) now use an isolated `doosan_node` internal architecture to prevent deadlocks.

## Achievements (March 20, 2026)
- **Liquid Detection Refinement:** Switched from mask-based to bounding-box-based calibration for the cup bottom in `realsense_cam2.py`. This resolves the issue where the liquid mask "cut" the cup mask, leading to incorrect `0.0ml` readings.
- **Debug Diagnostics:** Added a throttled debug logger (1Hz) to `realsense_cam2.py` to monitor `Cup_Total_Px`, `Liq_Height_Px`, and `Ratio` during the pour.
- **Pour Action Server:** Converted Pour node to Action Server with real-time feedback (Moving to cheers, approaching contact, pouring spline, recovering path).
- **Manager-Pour Integration:** Added `/dsr01/robotender_manager/pour_bottle` service. Manager now passes the ordered bottle name to the Pour action.
- **Post-Pour Orchestration:** Manager automatically commands the robot to `POSJ_HOME` after a successful pour, matching the Pick pattern.
- **Global Node Isolation:** Applied the isolated internal node pattern to the `Pour` node, resolving the "halt on second run" issue.
- **Resource Optimization:** Disabled heartbeat timers and logging in `Pick` and `Pour` nodes to save CPU resources while maintaining executor health via `ReentrantCallbackGroup`.

## Next Steps: Verification & Todo
- **Todo: Test Snap and Recovery:** Verify the "Snap" (interruption) logic during the pour motion to ensure the hybrid recovery spline works as expected.
- **Verify Cycle 2:** Confirm the Pour node continues to respond after multiple consecutive runs.
- **Monitor Latency:** Check if the removal of heartbeat timers affects long-term visibility into node health.

## Ongoing Issue: Gripper Velocity Slowdown
- **Problem:** After multiple pick/place cycles, the gripper's physical movement velocity appears to decrease. 
- **Current Status:** In `gripper.py`, the DRL template has been simplified to explicitly set `Goal PWM (270)` to 800 and `Goal Velocity (276)` to 500 in every call.

## Core Mandates
- **Orchestration Pattern:** The Manager node MUST handle the decision-making while individual nodes handle execution.
- **Action-Feedback:** All motion-heavy tasks (Pick, Pour, Place) MUST use Actions to provide real-time feedback.
- **Memory Management:** The Manager is responsible for storing and passing task-specific data (bottle names, coordinates).
- **Auto-Rebuild:** `colcon build --symlink-install --packages-select robotender_msgs bartender_test && source install/setup.bash`

## Operational Commands

### 1. System Startup (Full Stack)
```bash
# Start all hardware, logic nodes, and web server in one command
python3 scripts/start_order_stack.py --with-bringup
```

### 2. Manual Order Injection
```bash
# Inject a manual order (e.g., 1 Beer) directly to the system
ros2 topic pub --once /bartender/order_detail std_msgs/msg/String "{data: '{\"recipe\": {\"beer\": 1}}'}"
```

### 3. Orchestrated Execution (Manager Node)
**Trigger motions via the Manager to ensure data persistence and correct sequencing:**
```bash
# 1. Trigger Pick (Uses vision to find bottle)
ros2 service call /dsr01/robotender_manager/pick_bottle std_srvs/srv/Trigger {}

# 2. Trigger Pour (Uses stored bottle name and vision for volume)
ros2 service call /dsr01/robotender_manager/pour_bottle std_srvs/srv/Trigger {}

# 3. Trigger Place (Returns bottle to original location)
ros2 service call /dsr01/robotender_manager/place_bottle std_srvs/srv/Trigger {}
```

### 4. Manual Node Execution (Debug)
```bash
# If running nodes individually for debugging:
ros2 run bartender_test manager
ros2 run bartender_test pick
ros2 run bartender_test pour
ros2 run bartender_test place
```

## Hardware Specification
- **Robot:** Doosan Robotics dsr01 e0509 (6-DOF)
- **Gripper:** Robotis RH-P12-RN (Force-controlled)
- **Bottle:** Standard Soju (500ml / ~500g)
