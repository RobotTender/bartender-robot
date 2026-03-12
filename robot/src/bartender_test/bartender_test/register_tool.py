#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import DR_init
import sys
import time

def main():
    rclpy.init()
    ROBOT_ID = "dsr01"
    ROBOT_MODEL = "e0509"
    node = Node('register_tool_node', namespace=ROBOT_ID)
    DR_init.__dsr__id, DR_init.__dsr__model, DR_init.__dsr__node = ROBOT_ID, ROBOT_MODEL, node

    from DSR_ROBOT2 import config_create_tool, set_current_tool, wait, set_robot_mode, ROBOT_MODE_AUTONOMOUS

    try:
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        wait(0.5)

        # 1. Create the Tool Model
        # name: "RH_P12_RN"
        # weight: 0.85 kg (approx for this gripper)
        # cog: [0, 0, 50] (Center of gravity offset in mm from flange)
        # inertia: [0,0,0,0,0,0] (Simplified)
        tool_name = "RH_P12_RN"
        weight = 0.85 
        cog = [0.0, 0.0, 50.0]
        inertia = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        print(f"Registering tool: {tool_name} ({weight}kg)...")
        if config_create_tool(tool_name, weight, cog, inertia):
            print("Successfully created tool model.")
        else:
            print("Failed to create tool model (it might already exist).")

        # 2. Set it as the active tool
        print(f"Setting {tool_name} as active tool...")
        if set_current_tool(tool_name):
            print("Successfully activated tool.")
            print("\n!!! TOOL REGISTERED !!!")
            print("The robot now 'knows' the gripper is there.")
            print("Force sensing should now be much more sensitive to the bottle.")
        else:
            print("Failed to activate tool.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
