## Working Context (March 17, 2026 - Manual Command Setup)
- **Current Branch:** `robot-llm-combine-junsung`
- **Architecture Status:** 
    - **Persistent Node + Service** structure is the standard.
    - `robotender_pour` (`pour.py`): Handles complex motions (Pour, Warmup) with Snap Recovery.
    - `robotender_gripper` (`gripper.py`): Controls the Robotis RH-P12-RN via Autonomous DRL Injection (One-Shot).
    - `robotender_monitor` (`monitor.py`): Unified telemetry (Joints, Pose, Force, Weight).
    - `robotender_snap` (`snap.py`): Spacebar listener for manual SNAP/Recovery.
    - **One-Shot Nodes:** `pose`, `movej`, `movel` for manual control.
    - **LLM Integration:** `llm/web/order_engine/` contains the logic for order classification and making.

## Core Mandates
- **Architecture Priority:** Maintain the **Persistent Node + Service** structure. Do not revert to one-shot scripts for core operations.
- **Gripper Implementation:** The gripper **MUST** be controlled via autonomous DRL injection (One-Shot Modbus) from the `robotender_gripper` node.
- **Auto-Rebuild:** Every time a code change is made to `bartender_test`, you **MUST** automatically run:
  `colcon build --symlink-install --packages-select bartender_test && source install/setup.bash`
- **Safety First:** Always use `movesj` or `movej` with cautious velocities (default 60, max 250 for recovery) to prevent hardware collisions.
- **Commit Rule:** Commit every change immediately, only push when explicitly asked

## Project Organization (Node-Service Structure)
- `robot/src/bartender_test/`: The primary ROS 2 Python package.
    - `bartender_test/pour.py` (**`robotender_pour`**):
        - Service: `/dsr01/robotender_pour/start` (Pouring with Snap Recovery).
        - Service: `/dsr01/robotender_pour/warmup` (Pose verification).
        - Subscription: `robotender_snap/trigger` (Manual SNAP).
    - `bartender_test/gripper.py` (**`robotender_gripper`**):
        - Service: `/dsr01/robotender_gripper/open` (Autonomous DRL injection).
        - Service: `/dsr01/robotender_gripper/close` (Autonomous DRL injection).
    - `bartender_test/monitor.py` (**`robotender_monitor`**): Real-time display of joints, TCP pose, and weight.
    - `bartender_test/snap.py` (**`robotender_snap`**): Physical interrupt node (Spacebar listener).
    - `bartender_test/pick.py` (**`robotender_pick`**): Vision-based picking and task coordination.
    - `bartender_test/pose.py` (**`pose`**): One-shot move to predefined poses.
    - `bartender_test/movej.py` (**`movej`**): One-shot joint-space movement.
    - `bartender_test/movel.py` (**`movel`**): One-shot linear-space movement.
    - `bartender_test/defines.py`: Central repository for Poses (`posj`, `posx`) and physical constants.


## Hardware Specification
- **Robot:** Doosan Robotics dsr01 e0509 (6-DOF)
- **Gripper:** Robotis RH-P12-RN (Force-controlled)
- **Bottle:** Standard Soju (500ml / ~500g)

## Operational Commands
### Virtual Mode (Simulation with RViz)
```bash
ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=virtual model:=e0509 gui:=true gripper:=rh_p12_rn object:=bottle
```

### Real Mode (Physical Robot)
```bash
ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=real host:=110.120.1.xx model:=e0509
```

### Full Stack (Launch All Nodes)
```bash
python3 scripts/start_order_stack.py --with-bringup --bringup-cmd "ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=real host:=110.120.1.xx model:=e0509"
```

### Manual Control (One-Shot Commands)
```bash
# Move to predefined poses (home, cheers, contact, horizontal, diagonal, vertical, pole)
ros2 run bartender_test pose cheers

# Joint-space movement (relative or absolute)
ros2 run bartender_test movej j4 rel 10
ros2 run bartender_test movej 0 0 90 0 90 0

# Linear-space movement (dx, dy, dz in cm)
ros2 run bartender_test movel 0 0 +5

# Gripper control (Service calls to persistent robotender_gripper node)
ros2 service call /dsr01/robotender_gripper/open std_srvs/srv/Trigger {}
ros2 service call /dsr01/robotender_gripper/close std_srvs/srv/Trigger {}
```