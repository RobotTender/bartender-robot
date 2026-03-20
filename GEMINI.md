## Working Context (March 20, 2026 - Manager-Place Action Orchestration)
- **Current Branch:** `robot-llm-detection-combine-junsung`
- **Architecture Status:** 
    - **Full Manager-Orchestration** for Pick and Place is now active.
    - `robotender_manager` (**`manager.py`**): Stores `last_picked_pose` from Pick action and coordinates the Place action.
    - `robotender_pick` (**`pick.py`**): ROS 2 Action Server (Vision-based detection + Grasping).
    - `robotender_place` (**`place.py`**): **ROS 2 Action Server** (8-step motion sequence with real-time feedback).
    - **Action Standardization:** Motion tasks moved from Services to Actions for feedback and cancellation support.

## Achievements (March 20, 2026)
- **Place Action Server:** Converted Place node to Action Server with real-time feedback (Approaching, Lifting, Lowering, Releasing, Retreating).
- **Coordinate Handover:** Manager now automatically remembers the exact coordinates where a bottle was picked and sends them to the Place node.
- **Manual Control Service:** Added `/dsr01/robotender_manager/place_bottle` to manually trigger the saved placement sequence.
- **Improved Reliability:** Implemented "Fire & Forget" gripper patterns to prevent ROS 2 service deadlocks during physical motion.
- **Callback Starvation Fix:** 
    - Reverted `async/await` in `pick.py` and `place.py` to pure synchronous methods with `time.sleep()`.
    - Set `self._default_callback_group = ReentrantCallbackGroup()` in both nodes to ensure system-internal state updates are not blocked by high-latency vision/motion logic.
    - Removed `callback_group` parameter from `message_filters.Subscriber` to avoid `TypeError`.

## Next Steps: Verification
- **Verify Cycle 2:** Run two consecutive pick/place cycles to ensure the Pick node no longer halts.
- **Monitor Latency:** Check if the `ReentrantCallbackGroup` causes any race conditions in the vision synchronization.

## Core Mandates
- **Orchestration Pattern:** The Manager node MUST handle the decision-making (what/when) while individual nodes handle execution.
- **Action-Feedback:** All motion-heavy tasks (Pick, Place) MUST use Actions to provide real-time feedback to the Manager.
- **Memory Management:** The Manager is responsible for storing and passing task-specific data (like picked coordinates) between workers.
- **Auto-Rebuild:** Every time a code change is made to `bartender_test`, you **MUST** automatically run:
  `colcon build --symlink-install --packages-select robotender_msgs bartender_test && source install/setup.bash`

## Operational Commands
### 1. Full Stack (Real Mode)
```bash
python3 scripts/start_order_stack.py --with-bringup
```

### 2. Manual Node Execution
*Note: Always source the venv and workspace in each terminal.*
```bash
source .venv/bin/activate && source robot/install/setup.bash

# Manager Node
ros2 run bartender_test manager

# Pick Node (Vision/YOLO)
ros2 run bartender_test pick

# Place Node (Motion)
ros2 run bartender_test place
```

### 3. Interaction & Testing
**Workflow A: Orchestrated Flow (Recommended)**
```bash
# 1. Trigger Pick (via Order Topic)
ros2 topic pub --once /bartender/order_detail std_msgs/msg/String "{data: '{\"recipe\": {\"soju\": 1}}'}"

# 2. Trigger Place (via Manager Service - uses stored coordinates)
ros2 service call /dsr01/robotender_manager/place_bottle std_srvs/srv/Trigger {}
```

**Workflow B: Direct Action Testing (Individual Workers)**
```bash
# Direct Pick
ros2 action send_goal --feedback /dsr01/robotender_pick/execute robotender_msgs/action/PickBottle "{bottle_name: 'soju'}"

# Direct Place (Requires manual coordinates)
ros2 action send_goal --feedback /dsr01/robotender_place/execute robotender_msgs/action/PlaceBottle "{picked_pose: [450.0, 50.0, 120.0]}"
```

## Hardware Specification
- **Robot:** Doosan Robotics dsr01 e0509 (6-DOF)
- **Gripper:** Robotis RH-P12-RN (Force-controlled)
- **Bottle:** Standard Soju (500ml / ~500g)
