# Bartender Test Package

This package provides a suite of ROS 2 nodes for controlling a Doosan E0509 robot equipped with a Robotis RH-12-RN gripper. It includes specialized functions for precise movement, gripper control, and high-precision object weighing using joint torques.

## Prerequisites

- **OS:** Ubuntu 22.04 or 24.04
- **ROS 2:** Jazzy Jalisco
- **Hardware:** Doosan E0509 Robot + Robotis RH-12-RN Gripper
- **Driver:** [Doosan Robot ROS 2 Driver](https://github.com/doosan-robotics/doosan-robot2) (Must be installed in the same workspace).

## Installation

1. Create a ROS 2 workspace:
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   ```

2. Clone the Doosan Driver and this package:
   ```bash
   # Clone Doosan driver (if not already present)
   git clone https://github.com/doosan-robotics/doosan-robot2.git
   
   # Clone this package
   git clone <your-repository-url> bartender_test
   ```

## Building

From the root of your workspace (`~/ros2_ws`):

```bash
# Build the package
colcon build --packages-select bartender_test --symlink-install

# Source the workspace
source install/setup.bash
```

## Usage

### 1. Robot Movement (`move_to`)
Move the robot to pre-defined poses or specific joint values.

```bash
# Pre-defined poses
ros2 run bartender_test move_to home      # Standard home position
ros2 run bartender_test move_to cheers    # Bottle-holding/Cheers position
ros2 run bartender_test move_to pole      # Vertical/Pole position

# Relative joint moves
ros2 run bartender_test move_to j1 rel 10 # Joint 1 +10 degrees

# Absolute joint moves
ros2 run bartender_test move_to j3 abs 45 # Set Joint 3 to 45 degrees
```

### 2. Gripper Control (`gripper_command`)
Open or close the gripper. **Safety Note:** `close` requires a force value.

```bash
# Open the gripper
ros2 run bartender_test gripper_command open

# Close the gripper (Strict order: close <force>)
ros2 run bartender_test gripper_command close 150  # Success
ros2 run bartender_test gripper_command close      # No action (Safe)
```

### 3. Precision Weighing (`measure_weight`)
Measure the weight of a grasped object using multi-joint torque sensing (J2, J3, J5).

```bash
# Step 1: Establish baseline (Automated mechanical reset via Home)
ros2 run bartender_test measure_weight tare

# Step 2: Grab and weigh object (Active stabilization monitoring)
ros2 run bartender_test measure_weight first 150

# Step 3: Re-measure (Fast check without disturbing the grasp)
ros2 run bartender_test measure_weight again
```

## Features
- **Active Stabilization:** Automatically detects when mechanical vibrations settle (typically 3-5s) instead of using fixed timers.
- **Gold Standard Workflow:** Automated Home-to-Pose resets during `tare` to eliminate harmonic drive hysteresis.
- **Multi-Joint Sensing:** Uses torques from J2, J3, and J5 for high-precision payload estimation.
- **Adaptive Wait:** Fast 2s wait for repeated measurements, 10s for initial movements.
