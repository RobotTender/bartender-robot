import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Robotender Gripper (Hardware Driver)
    # This persistent node manages the gripper services.
    gripper_node = Node(
        package='bartender_test',
        executable='gripper',
        name='robotender_gripper',
        namespace='dsr01',
        output='screen'
    )

    # 2. Robotender Action (Pouring & Warmup Brain)
    # This persistent node manages the pouring services and recovery logic.
    action_node = Node(
        package='bartender_test',
        executable='action',
        name='robotender_action',
        namespace='dsr01',
        output='screen'
    )

    return LaunchDescription([
        gripper_node,
        action_node
    ])
