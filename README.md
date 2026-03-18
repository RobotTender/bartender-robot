# RobotTender: Autonomous Bartender Robot Project

This repository contains the implementation of an autonomous bartender system using a Doosan Robotics E0509 robot, a Robotis RH-P12-RN gripper, and RealSense depth cameras. This report summarizes the implementation details and improvements made in the current branch (`robot-llm-detection-combine-junsung`).

## Project Structure Overview

The project is divided into three main components:
1.  **LLM (Order Engine)**: Handles natural language processing for orders (maintained from previous branches).
2.  **Detection**: Vision-based system for bottle picking and liquid volume measurement.
3.  **Robot**: ROS 2-based control system for robot motions and gripper operations.

---

## 1. Detection System

### Volume Measurement Improvements
In previous versions (specifically the `od-realsense` branch), measuring liquid volume was inaccurate because it relied on a **fixed lookup table** mapping absolute pixel height (`height_px`) directly to volume (`ml`). 

**Verification of Original Method:**
Analysis of the codebase (specifically `realsense_cam2.py`) confirms that the original version used fixed arrays and lacked advanced temporal filtering for boundaries:
*   `known_heights_px`: `[0, 26.3, 50, 67.4, 84, 100, 115.7, 131.9, 140.1, 158, 174]`
*   `known_volumes_ml`: `[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]`
*   **No Boundary EMA**: The original branch did not employ Exponential Moving Average (EMA) filters for detecting or locking the cup's boundary. It relied on absolute pixel values, making it highly sensitive to noise and slight camera/cup movements.

**Key Changes in Current Branch:**
*   **Ratio-Based Lookup Table**: We transitioned to a ratio-based approach. The system now uses a lookup table that maps the **Ratio** (`liquid_height_px / total_cup_height_px`) to **Volume** (`ml`). This normalization allows the system to remain accurate even if the cup's perceived size changes.
*   **EMA Filtering**: Added Exponential Moving Average (EMA) filters for liquid height (`height_px_ema`) and volume estimation to smooth out sensor noise and provide more stable readings.
*   **Initial Calibration (Auto-Tare)**: A calibration step was added to capture the cup's state when it is first placed. The system waits for a stable detection of an empty cup (stability-based trigger) to lock the `total_cup_height_px` and the `fixed_bottle_bottom_y` coordinate. This allows for flexible cup positioning within the detection area.
*   **Auto-Snap Trigger**: Once the estimated volume reaches the target (e.g., 100ml), the detection node automatically publishes a trigger signal to the robot to stop pouring and initiate recovery.

### Snap Recovery & Motion Interruption
A key challenge was interrupting the "blocking" pouring motion (`movesx`) when the target volume is reached or the spacebar is pressed.

---

## 2. Robot Control System

The robot system follows a **Persistent Node + Service** architecture, ensuring stable communication and real-time responsiveness.

### Initialization Sequence
The system initialization is handled by `startup.py`, which:
1.  Performs a system readiness check.
2.  Resets the robot's safety state and ensures it is in `STANDBY`.
3.  Stops any lingering DRL (Doosan Robot Language) scripts to ensure a clean slate.

### Node-Service Architecture

| Node | Responsibility | Primary Services / Topics |
| :--- | :--- | :--- |
| `robotender_gripper` | Controls the Robotis RH-P12-RN gripper via autonomous DRL injection (One-Shot Modbus). | `/open` (Trigger), `/close` (Trigger) |
| `robotender_pick` | Vision-based (YOLOv8 + cam_1) picking. Transforms coordinates and coordinates the task. | Sub: `/bartender/order_detail`, Pub: `last_pose` |
| `robotender_pour` | Executes pouring motions with **Snap Recovery**. Handles trajectory recording and backtracking. | `/start` (Trigger), `/warmup` (Trigger) |
| `robotender_place` | Returns the bottle to its original pick location using coordinates from the pick node. | `/start` (Trigger) |
| `robotender_snap` | Physical interrupt listener (Spacebar) and vision trigger monitor. | Pub: `robotender_snap/trigger` |
| `robotender_monitor` | Unified telemetry display (Joints, Pose, Force, Weight). | Sub: `/joint_states`, etc. |

### 2.1 Gripper Node (`gripper.py`)
The gripper is controlled via the `robotender_gripper` node, which provides ROS 2 Trigger services.
*   **One-Shot DRL Injection**: Instead of a persistent DRL script, it uses the `/dsr01/drl/drl_start` service to inject raw DRL code for each command. This prevents the robot controller from being "occupied" by a gripper script.
*   **Modbus Communication**: The injected DRL handles high-frequency Modbus FC06/FC16 commands via the robot's flange serial port (57600 baud) to control the Robotis RH-P12-RN.

### 2.2 Pick Node (`pick.py`)
The `robotender_pick` node manages the initial phase of the bartender task.
*   **Vision-to-Robot Transformation**: It synchronizes RGB and Depth frames from Camera 1, detects bottles using YOLOv8, and transforms pixel coordinates into robot base coordinates using a pre-calibrated Rotation (R) and Translation (t) matrix.
*   **Coordinate Memory**: Crucially, when a bottle is picked, the node publishes its exact X, Y, Z coordinates to the `robotender_pick/last_pose` topic. This allows the system to return the bottle exactly where it was found later.

### 2.3 Pour Node (`pour.py`)
The `robotender_pour` node handles the trajectory execution for pouring and the critical **Snap Recovery** logic.

**Parallel Interruption Flow:**
1.  **Thread A (Motion)**: Sends a command to the Robot Controller: "Start moving (`movesx`) and block until finished."
2.  **Thread B (Monitor)**: Independently listens for a "Snap" message from the vision system or spacebar.
3.  **Snap Event**: Thread B receives the trigger and immediately sends a `MoveStop` command (Stop Mode 2) to the Robot Controller.
4.  **Controller Action**: The Robot Controller kills the active motion trajectory instantly.
5.  **Motion Release**: The Robot Controller signals back to Thread A that the requested motion has ended (interrupted).
6.  **Recovery Phase**: Thread A "wakes up," detects the interruption, and immediately executes the **Recovery Path** (Backtracking).

**Recovery Execution:**
1.  **Real-time Path Recording**: While pouring (`self.recording = True`), a background timer records the robot's joint poses (`posj`) at 3Hz into a buffer.
2.  **Dynamic Backtracking**: Once the motion is stopped, the node calculates a recovery path by reversing the recorded buffer. To ensure smooth motion, the buffer is downsampled to a target size (1-3 points depending on duration) and the current pose and `CHEERS_POSE` are added.
3.  **High-Velocity Snap**: The robot executes this reverse path using `movesj` at high velocity (200 deg/s) to quickly "snap" back to a safe posture (`CHEERS_POSE`), effectively preventing liquid overspill.

### 2.4 Place Node (`place.py`)
The `robotender_place` node is responsible for returning the bottle after pouring is complete.
*   **Shared Memory via ROS Topics**: The node subscribes to `robotender_pick/last_pose`. It updates its internal `last_pick_pose` state every time the Pick node identifies a target.
*   **Dynamic Return Trajectory**: When the `/dsr01/robotender_place/start` service is called, it uses the stored coordinates to generate an approach and placement path, ensuring the bottle is returned to its original position without manual programming.

### 2.5 Common Configuration (`defines.py`)
All nodes share a centralized repository for:
*   **Predefined Poses**: Joint angles for `HOME`, `CHEERS`, `CONTACT`, etc.
*   **Cartesian Markers**: Fixed XYZ coordinates for pouring trajectory waypoints.
*   **Implementation Note**: The current implementation relies on hardcoded offsets within the motion logic (e.g., approach/retreat distances) to ensure immediate reliability across different bottle types without requiring complex geometry calculations.

---

## 3. Full Stack Execution Sequence

To run the complete system, follow these steps:

1.  **Bringup the Robot**:
    ```bash
    ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=real host:=110.120.1.xx model:=e0509
    ```
2.  **Verify Startup**:
    ```bash
    ros2 run bartender_test startup
    ```
3.  **Launch Order Stack** (Starts all persistent nodes):
    ```bash
    python3 scripts/start_order_stack.py
    ```
4.  **Launch Detection Nodes**:
    *   Pick Camera: `python3 detection/realsense_cam1.py`
    *   Volume Camera: `python3 detection/realsense_cam2.py`
5.  **Inject Order**:
    ```bash
    python3 scripts/test_post_order.py [soju|beer|juice]
    ```

---

## 4. Future Improvements

### Detection Enhancements
*   **Temporal Filtering for Boundaries**: Implement Exponential Moving Average (EMA) filters for the cup's boundary (utilizing the prepared `bottle_area_ema` logic) to further stabilize the reference scale during calibration and pouring.
*   **Dynamic Target Volume**: Integrate the detection node with the LLM/Order Engine to set the `target_volume_ml` dynamically based on recipe details instead of a hardcoded value.
*   **Support for Transparent/Reflective Vessels**: Improve segmentation accuracy for glass or reflective cups using specialized depth-aware models.

### Robot Control & Safety
*   **Coordinate Tuning**: Refine the pick coordinate transformation logic to correct the slight horizontal offset currently observed during gripper approach.
*   **Class-Specific Offsets**: Define and implement specific Z-axis offsets for each bottle class (Soju, Beer, Juice) to optimize grasp points.
*   **Pose Transition Optimization**: Smooth the joint-space transitions between `HOME` -> `CHEERS` -> `CONTACT` to reduce mechanical stress and improve fluidity.
*   **Advanced Force Control**: Implement dynamic gripper force control to effectively and safely grasp opened bottles with varying structural integrity.
*   **Dynamic Obstacle Avoidance**: Utilize RealSense depth data to detect unexpected obstacles in the workspace and plan real-time collision-free trajectories.
*   **Enhanced Calibration Protocols**: Automate the extrinsic calibration between cameras and the robot base for faster setup in new environments.

---

*This report reflects the status as of March 18, 2026.*
