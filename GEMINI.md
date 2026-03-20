## Working Context (March 20, 2026 - Manager-Pour Action Orchestration)
- **Current Branch:** `robot-llm-detection-combine-junsung`
- **Architecture Status:** 
    - **Full Manager-Orchestration** for Pick, Pour, and Place is now active.
    - `robotender_manager` (**`manager.py`**): Orchestrates the full sequence (Pick -> Pour -> Place).
    - `robotender_pour` (**`pour.py`**): **ROS 2 Action Server** (Vision-triggered pouring with spline motion).
    - **Node Isolation Standard:** All motion nodes (Pick, Pour, Place) now use an isolated `doosan_node` internal architecture to prevent deadlocks.

## Achievements (March 20, 2026)
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
### 1. Manual Node Execution
```bash
# Manager Node
ros2 run bartender_test manager
# Pick Node
ros2 run bartender_test pick
# Pour Node
ros2 run bartender_test pour
# Place Node
ros2 run bartender_test place
```

### 2. Interaction & Testing
**Orchestrated Flow**
```bash
# 1. Trigger Pick
ros2 service call /dsr01/robotender_manager/pick_bottle std_srvs/srv/Trigger {}
# 2. Trigger Pour (Uses stored bottle name)
ros2 service call /dsr01/robotender_manager/pour_bottle std_srvs/srv/Trigger {}
# 3. Trigger Place (Uses stored coordinates)
ros2 service call /dsr01/robotender_manager/place_bottle std_srvs/srv/Trigger {}
```

## Hardware Specification
- **Robot:** Doosan Robotics dsr01 e0509 (6-DOF)
- **Gripper:** Robotis RH-P12-RN (Force-controlled)
- **Bottle:** Standard Soju (500ml / ~500g)
