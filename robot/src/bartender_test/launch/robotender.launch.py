import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Robotender Gripper (Hardware Driver)
    gripper_node = Node(
        package='bartender_test',
        executable='gripper',
        name='robotender_gripper',
        namespace='dsr01',
        output='screen'
    )

    # 2. Robotender Pour (Pouring & Snap Recovery)
    pour_node = Node(
        package='bartender_test',
        executable='pour',
        name='robotender_pour',
        namespace='dsr01',
        output='screen'
    )

    # 3. Robotender Place (Bottle Placement)
    place_node = Node(
        package='bartender_test',
        executable='place',
        name='robotender_place',
        namespace='dsr01',
        output='screen'
    )

    # 4. Robotender Snap (Manual Interrupt)
    snap_node = Node(
        package='bartender_test',
        executable='snap',
        name='robotender_snap',
        namespace='dsr01',
        output='screen'
    )

    return LaunchDescription([
        gripper_node,
        pour_node,
        place_node,
        snap_node
    ])
