## Working Context (March 19, 2026 - Manager Orchestration & Action Control)
- **Current Branch:** `robot-llm-detection-combine-junsung`
- **Architecture Status:** 
    - **Manager-Orchestration based control** is the current focus.
    - `robotender_manager` (**`manager.py`**): Orchestrates the entire pick-pour-place sequence.
    - `robotender_pick` (**`pick.py`**): Converted to **ROS 2 Action Server** for feedback and coordination.
    - `robotender_pour` (**`pour.py`**): Service-based motion control.
    - `robotender_place` (**`place.py`**): Service-based placement.
    - **Pose Migration:** All joint poses renamed to `POSJ_...` format in `defines.py`.

## Core Mandates
- **Orchestration Pattern:** The Manager node MUST handle the decision-making (what/when) while individual nodes handle execution.
- **Action-Feedback:** Long-running tasks like Picking MUST use Actions to provide real-time feedback to the Manager.
- **Standby Readiness:** The Manager MUST execute a physical standby sequence (Home + Close Gripper) before enabling order subscriptions.
- **Mock Mode:** The Pick node supports `use_mock_vision:=True` for testing in virtual environments without a camera.
- **Architecture Priority:** Maintain the **Persistent Node + Service/Action** structure. 
- **Auto-Rebuild:** Every time a code change is made to `bartender_test`, you **MUST** automatically run:
  `colcon build --symlink-install --packages-select bartender_test && source install/setup.bash`
- **Commit Rule:** commit and push when explicitly asked

## Project Organization (Node-Service Structure)
- `robot/src/bartender_test/`: The primary ROS 2 Python package.
    - `bartender_test/manager.py` (**`robotender_manager`**): Orchestrator client.
    - `bartender_test/pick.py` (**`robotender_pick`**): Action Server for detection and grasping.
    - `bartender_test/pour.py` (**`robotender_pour`**): Service for pouring logic.
    - `bartender_test/place.py` (**`robotender_place`**): Service for bottle placement.
    - `bartender_test/defines.py`: Central repository for Poses (`POSJ_...`) and offsets.

# Future Tasks
- **Test Manager-Pick Control:** Verify the full handshake between Manager and Pick Action Server in both virtual (Mock) and real modes.
- **Integrate Pour/Place into Manager:** Chain the success of the Pick Action into the Pour and Place service calls.
- **Recovery Motion Redesign:** Verify that waypoint-based tracking is consistent with the new orchestration flow.

## Launch Section
### Virtual Orchestration Test
```bash
# Terminal 1: Bringup
ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=virtual model:=e0509 gui:=true gripper:=rh_p12_rn object:=bottle
# Terminal 2: Gripper
ros2 run bartender_test gripper
# Terminal 3: Pick (Mock)
ros2 run bartender_test pick --ros-args -p use_mock_vision:=True
# Terminal 4: Manager
ros2 run bartender_test manager
# Terminal 5: Order Injection
ros2 topic pub --once /bartender/order_detail std_msgs/msg/String "{data: '{\"recipe\": {\"soju\": 1}}'}"
```

## Hardware Specification
- **Robot:** Doosan Robotics dsr01 e0509 (6-DOF)
- **Gripper:** Robotis RH-P12-RN (Force-controlled)
- **Bottle:** Standard Soju (500ml / ~500g)

## Operational Commands (Manual)
### Manual Control (One-Shot Commands)
```bash
# Move to predefined poses
ros2 run bartender_test pose home
ros2 run bartender_test pose cheers
ros2 run bartender_test pose ready

# Gripper control (Service calls to persistent robotender_gripper node)
ros2 service call /dsr01/robotender_gripper/open std_srvs/srv/Trigger {}
ros2 service call /dsr01/robotender_gripper/close std_srvs/srv/Trigger {}
```
