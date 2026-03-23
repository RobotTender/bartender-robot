## Working Context (March 23, 2026 - Real-time Mode Orchestration)
- **Current Branch:** `robot-llm-detection-combine-junsung`
- **Architecture Status:** 
    - **Full Manager-Orchestration** for Pick, Pour, and Place.
    - **Auto/Manual Mode:** Real-time switching via ROS topic.
    - **Node Isolation Standard:** All motion nodes use isolated internal nodes.

## Achievements (March 23, 2026)
- **Real-time Mode Selection:** Added `/dsr01/robotender_manager/mode` topic (`std_msgs/msg/String`) to toggle between `auto` and `manual`.
- **Automatic Sequence Chaining:** In `auto` mode, receiving an order now triggers the full **Pick -> Pour -> Place** sequence automatically.
- **Interruptible Auto-Sequence:** The manager checks the execution mode before every major step (Pick, Pour, Place). If switched to `manual` mid-sequence, the robot completes the current step to a safe position (Home) and stops.
- **Simplified Launch:** Removed redundant `manual`/`auto` command-line arguments. The system defaults to `auto` mode on startup.
- **Syntax Fix:** Resolved `SyntaxWarning` regarding invalid escape sequences in regex strings within `startup.py` and `start_order_stack.py`.

## Achievements (March 20, 2026)
- **Liquid Detection Refinement:** Switched from mask-based to bounding-box-based cup calibration.
- **Pour Action Server:** Converted Pour node to Action Server with real-time feedback.
- **Global Node Isolation:** Applied the isolated internal node pattern to all motion nodes.

## Next Steps: Verification & Todo
- **Verify Real-time Interrupts:** Test switching to manual at various points in the auto-sequence to ensure safe halting.
- **Test Snap and Recovery:** Verify the "Snap" (interruption) logic during the pour motion.

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
