# Robotics Project: Robotender

## Hardware
- **Robot:** Doosan Robotics dsr01 e0509
- **Gripper:** Robotis RH-P12-RN

## Software
- **OS:** Ubuntu 24.04 LTS
- **ROS 2:** Jazzy Jalisco

## Project Status
- **Project Name:** Robotender (Robot Bartender)
- **Current Task:** Continuous Orbit Motion using Spline
- **Active Package:** `bartender_test`
- **Recent Updates:** 
  - Implemented `orbit` command using `movesx` for a smooth 5-point path.
  - Added `monitor` command for live tracking of Joint and Cartesian coordinates.
  - Optimized trajectory to ensure consistent TCP orientation throughout the motion.
  - Standardized pouring poses: `POUR_HORIZONTAL`, `POUR_DIAGONAL`, and `POUR_VERTICAL`.

## Commands
- **Working Directory:** `cd ~/ros2_ws` (or your workspace root)
- **First-time Build:** 
  - `rosdep install --from-paths src --ignore-src -y` (Install dependencies)
  - `colcon build --symlink-install --allow-overriding dsr_description2`
  - `source install/setup.bash`
- **Incremental Build (Recommended):** `colcon build --symlink-install --packages-select bartender_test dsr_description2 --allow-overriding dsr_description2`
- **Real Mode (Connect to Hardware):** `ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=real host:=110.120.1.59 model:=e0509`
- **Virtual Mode (Simulation):** 
  - `ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=virtual model:=e0509` (Basic)
  - `ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=virtual model:=e0509 gripper:=rh_p12_rn object:=bottle` (With Gripper & Bottle)
- **Actions & Pouring:** 
  - `ros2 run bartender_test action warmup [repeat=<n>]`
  - `ros2 run bartender_test action pour move [horizontal|vertical] [repeat=<n>]` (CHEERS -> POUR_VERTICAL)
  - `ros2 run bartender_test action pour orbit [<start_idx>] [<end_idx>] [repeat=<n>]` (Integrated Spline)
- **Monitoring (1Hz Refresh):**
  - `ros2 run bartender_test monitor xyz` (Cartesian coordinates)
  - `ros2 run bartender_test monitor j` (Joint angles)
  - `ros2 run bartender_test check_tcp` (Current TCP position)

> **CRITICAL SYNC RULE:** If `POS1_XYZ` through `POS5_XYZ` are updated in `defines.py`, the corresponding hardcoded values in `src/doosan-robot2/dsr_description2/xacro/e0509.urdf.xacro` **MUST** also be updated manually. A full simulation restart is required to see updated markers in RViz.

- **Named Poses:** `ros2 run bartender_test pose [home|cheers|contact|pour_horizontal|pour_diagonal|pour_vertical|pole]`
- **Joint Control:** `ros2 run bartender_test movej j<num> [rel|abs] <val>` (e.g., `movej j1 abs 90`)
- **Gripper Control:** `ros2 run bartender_test gripper [open|close <force>]` (Close defaults to Max Force: 800)

## Simulation Control
- **Pause Motion:** `ros2 service call /dsr01/motion/move_pause dsr_msgs2/srv/MovePause`
- **Resume Motion:** `ros2 service call /dsr01/motion/move_resume dsr_msgs2/srv/MoveResume`
