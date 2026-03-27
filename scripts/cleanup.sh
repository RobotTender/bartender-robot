#!/bin/bash

# A script to forcefully clean up all ROS 2 and robot-related processes

echo "--- Cleaning up all Bartender Robot processes ---"

# 1. Kill the main stack script if running
echo "Stopping main execution scripts..."
pkill -9 -f "start_order_stack.py"
pkill -9 -f "test_post_order.py"

# 2. Kill individual ROS 2 nodes and project-specific libs
echo "Stopping ROS 2 project nodes..."
pkill -9 -f "python3 -m bartender_test\.(gripper|pick|pour|place|monitor|snap|startup)"
pkill -9 -f "/lib/bartender_test/"
pkill -9 -f "ros2 run bartender_test"
pkill -9 -f "bartender_test."
pkill -9 -f "bartender_test/"
pkill -9 -f "python3 -m bartender_test"

# 3. Kill Core ROS 2 and Doosan Robot Driver processes
echo "Stopping Core ROS 2 and Robot Driver processes..."
pkill -9 -f "dsr_"
pkill -9 -f "robot_state_publisher"
pkill -9 -f "controller_manager"
pkill -9 -f "ros2_control_node"
pkill -9 -f "virtual_node"
pkill -9 -f "run_emulator"
pkill -9 -f "/opt/ros/"

# 4. Kill Realsense hardware processes
echo "Stopping Realsense hardware processes..."
pkill -9 -f "realsense_cam"

# 5. Kill Django web server if running
echo "Stopping Django web server..."
pkill -9 -f "manage.py runserver"

# 6. Restart the ROS 2 daemon to clear the node list
echo "Restarting ROS 2 daemon..."
ros2 daemon stop
ros2 daemon start

# Wait a moment for processes to release resources
sleep 1

echo "--- Cleanup Complete. Checking for remaining nodes... ---"
ros2 node list

echo "Done."
