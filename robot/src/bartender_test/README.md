# Bartender Test Package (Robotender)

A ROS 2 package for the Doosan Robotics dsr01 e0509 robot and Robotis RH-P12-RN gripper, focused on achieving precision pouring through high-speed "Snapping" (recovery) motions.

---

## Getting Started

### 1. Execution Modes

The project can be run in either a simulated environment or on the physical hardware.

#### **Virtual Mode (Simulation)**
```bash
ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=virtual model:=e0509 gripper:=rh_p12_rn object:=bottle
```

#### **Real Mode (Hardware)**
Ensure the robot controller is on the network and replace `xx` with the specific ID of your robot (e.g., `110.120.1.59`).
```bash
ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=real host:=110.120.1.xx model:=e0509
```

---

## Command Reference

| Command | Subcommand | Description |
| :--- | :--- | :--- |
| `action` | `pour [repeat=n]` | **Main Task:** Executes the pouring orbit. Press **Spacebar** to trigger an immediate snap recovery. |
| `action` | `warmup [repeat=n]` | Exercises the robot through all key poses (Home, Cheers, Contact, Pouring). |
| `monitor` | *(None)* | **Telemetry:** Real-time dashboard showing Joint/TCP positions, Net Forces, and Unified Weight. |
| `trigger` | *(None)* | **Listener:** Required for snapping. Listens for the Spacebar to trigger the recovery motion. |
| `gripper` | `open / close` | Manual control of the Robotis gripper. |
| `pose` | `<pose_name>` | **Movement:** Moves the robot to a named pose (e.g., `home`, `cheers`, `contact`, `pour_horizontal`). |
| `movej` | *(None)* | Basic point-to-point Joint-space movement test script. |
| `check_tcp` | *(None)* | Verifies the current Tool Center Point configuration. |
| `register_tool` | *(None)* | Registers the bottle dimensions and weight on the robot controller. |

---

## Implementation Details

### **Pouring Action (`action pour`)**
The pouring logic is designed to be smooth and continuous, primarily utilizing the Doosan **`movesx`** (Spline Motion) command. 

1.  **Path Generation:** The path is constructed as a spline through 4 key Cartesian points (Contact -> Horizontal -> Diagonal -> Vertical). 
2.  **Forward Motion:** Executed at a controlled speed (100mm/s) to maintain a steady flow.

### **Snapping (Recovery) Path**
The "Snap" is a high-speed backtrack triggered when the user (or future weight sensor) detects that the target volume has been reached.

1.  **Interruption:** Upon a Spacebar event, the system immediately calls the `move_stop(DR_SSTOP)` service to halt forward momentum.
2.  **Path Recording:** During the pour, the robot's joint states are recorded at **3Hz** into a circular buffer.
3.  **Adaptive Thinning:** To ensure high-speed stability, the recorded path is dynamically "thinned" down to 1–4 key waypoints based on the duration of the pour.
4.  **Reverse Execution:** The robot executes a **`movesj`** (Joint Spline) command at maximum acceleration (**250 deg/s / 250 deg/s²**) to snap back to the "Cheers" (upright) position, effectively cutting the flow of liquid instantly.

---

## File Descriptions

| File | Purpose |
| :--- | :--- |
| `action.py` | Orchestrates the high-level pouring and recovery logic. |
| `monitor.py` | Unified telemetry node for weighing and pose monitoring. |
| `trigger.py` | Non-blocking keyboard listener. |
| `defines.py` | Centralized hardware offsets and pose constants. |
| `gripper_command.py` | Manual control of the Robotis gripper. |
| `pose.py` | Movement utility for named target poses. |
