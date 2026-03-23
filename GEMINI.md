## Working Context (March 23, 2026 - Real-time Mode Orchestration)
- **Current Branch:** `robot-llm-detection-combine-junsung`
- **Architecture Status:** 
    - **Full Manager-Orchestration** for Pick, Pour, and Place.
    - **Auto/Manual Mode:** Real-time switching via ROS topic.
    - **Node Isolation Standard:** All motion nodes use isolated internal nodes.

## Achievements (March 23, 2026)
- **Time-Deterministic Pouring (Snap Strategy):** Implemented a high-precision snap trigger based on **Flow Detection + Timing**, replacing the slower filtered-volume trigger.
    - **Camera Side:** Detects the "First Drop" by identifying a waterline jump above the baseline. Confirms flow after 5 consecutive frames.
    - **Robot Side:** Receives `flow_started` signal and runs a calculated timer based on `pour_target_ml` and `flow_rate_ml_s`.
- **Real-time Mode Selection:** Added `/dsr01/robotender_manager/mode` topic (`std_msgs/msg/String`) to toggle between `auto` and `manual`.
- **Automatic Sequence Chaining:** In `auto` mode, receiving an order now triggers the full **Pick -> Pour -> Place** sequence automatically.
- **Interruptible Auto-Sequence:** The manager checks the execution mode before every major step (Pick, Pour, Place). If switched to `manual` mid-sequence, the robot completes the current step to a safe position (Home) and stops.
- **Simplified Launch:** Removed redundant `manual`/`auto` command-line arguments. The system defaults to `auto` mode on startup.
- **Syntax Fix:** Resolved `SyntaxWarning` regarding invalid escape sequences in regex strings within `startup.py` and `start_order_stack.py`.

## Achievements (March 20, 2026)
- **Liquid Detection Refinement:** Switched from mask-based to bounding-box-based cup calibration.
- **Pour Action Server:** Converted Pour node to Action Server with real-time feedback.
- **Global Node Isolation:** Applied the isolated internal node pattern to all motion nodes.

## Tuning & Calibration Guide (Snap Logic)
The system calculates the pour duration as:
`Wait Time = (Target Volume / Flow Rate) - Reaction Offset`

All tuning is done in `robot/src/bartender_test/bartender_test/defines.py`:

### 1. If the cup OVERFILLS:
- **Increase `flow_rate_ml_s`**: Tells the robot liquid is faster -> stops earlier.
- **Increase `REACTION_TIME_OFFSET`**: Accounts for mechanical braking lag (current: 0.7s).

### 2. If the cup UNDERFILLS:
- **Decrease `flow_rate_ml_s`**: Tells the robot liquid is slower -> pours longer.
- **Decrease `REACTION_TIME_OFFSET`**: Reduces the "head start" the robot takes to stop.

### 3. Stability Tuning (`realsense_cam2.py`):
- **`FLOW_STABILITY_THRESHOLD`**: Currently **5 frames**. Increase if reflections cause false starts; decrease for faster response.

## Core Mandates
- **Orchestration Pattern:** The Manager node MUST handle the decision-making and sequence logic.
- **Real-time Control:** Mode overrides MUST be handled via topics to allow flexibility without restarting.
- **Safety:** Always return to `POSJ_HOME` or a safe state before halting an auto-sequence.

## Operational Commands

### 1. System Startup (Full Stack)
```bash
# Start all hardware, logic nodes, and web server
python3 scripts/start_order_stack.py --with-bringup
```

### 2. Mode Switching (Real-time)
```bash
# Switch to Manual Mode (Orders won't start automatically)
ros2 topic pub --once /dsr01/robotender_manager/mode std_msgs/msg/String "{data: 'manual'}"

# Switch to Auto Mode (Chain Pick->Pour->Place automatically)
ros2 topic pub --once /dsr01/robotender_manager/mode std_msgs/msg/String "{data: 'auto'}"
```

### 3. Orchestrated Execution (Manager Node)
**Manual triggers (Use when in Manual Mode or for debugging):**
```bash
# 1. Trigger Pick
ros2 service call /dsr01/robotender_manager/pick_bottle std_srvs/srv/Trigger {}

# 2. Trigger Pour
ros2 service call /dsr01/robotender_manager/pour_bottle std_srvs/srv/Trigger {}

# 3. Trigger Place
ros2 service call /dsr01/robotender_manager/place_bottle std_srvs/srv/Trigger {}
```

## Hardware Specification
- **Robot:** Doosan Robotics dsr01 e0509 (6-DOF)
- **Gripper:** Robotis RH-P12-RN (Force-controlled)
- **Bottle:** Standard Soju (500ml / ~500g)
