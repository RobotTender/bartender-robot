# 
#  dsr_bringup2
#  Author: Minsoo Song (minsoo.song@doosan.com)
#  
#  Copyright (c) 2025 Doosan Robotics
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# 

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # This launch file is a wrapper for dsr_bringup2_rviz.launch.py
    # to provide a default entry point for the package.

    # Declare arguments that we want to pass through
    # These are the same arguments as in dsr_bringup2_rviz.launch.py
    ARGUMENTS = [
        DeclareLaunchArgument('name',       default_value = 'dsr01',          description = 'NAME_SPACE'     ),
        DeclareLaunchArgument('host',       default_value = '127.0.0.1',      description = 'ROBOT_IP'       ),
        DeclareLaunchArgument('port',       default_value = '12345',          description = 'ROBOT_PORT'     ),
        DeclareLaunchArgument('mode',       default_value = 'virtual',        description = 'OPERATION MODE' ),
        DeclareLaunchArgument('model',      default_value = 'm1013',          description = 'ROBOT_MODEL'    ),
        DeclareLaunchArgument('color',      default_value = 'white',          description = 'ROBOT_COLOR'    ),
        DeclareLaunchArgument('gui',        default_value = 'false',          description = 'Start RViz2'    ),
        DeclareLaunchArgument('gz',         default_value = 'false',          description = 'USE GAZEBO SIM' ),
        DeclareLaunchArgument('rt_host',    default_value = '192.168.137.50', description = 'ROBOT_RT_IP'    ),
        DeclareLaunchArgument('remap_tf',   default_value = 'false',          description = 'REMAP TF'       ),
        DeclareLaunchArgument('gripper',    default_value = 'none',           description = 'GRIPPER'        ),
        DeclareLaunchArgument('object',     default_value = 'none',           description = 'OBJECT'         ),
    ]

    # Path to the rviz launch file
    rviz_launch_file = PathJoinSubstitution([
        FindPackageShare('dsr_bringup2'),
        'launch',
        'dsr_bringup2_rviz.launch.py'
    ])

    # Include the rviz launch file and pass all arguments
    included_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rviz_launch_file),
        launch_arguments={
            'name':     LaunchConfiguration('name'),
            'host':     LaunchConfiguration('host'),
            'port':     LaunchConfiguration('port'),
            'mode':     LaunchConfiguration('mode'),
            'model':    LaunchConfiguration('model'),
            'color':    LaunchConfiguration('color'),
            'gui':      LaunchConfiguration('gui'),
            'gz':       LaunchConfiguration('gz'),
            'rt_host':  LaunchConfiguration('rt_host'),
            'remap_tf': LaunchConfiguration('remap_tf'),
            'gripper':  LaunchConfiguration('gripper'),
            'object':   LaunchConfiguration('object'),
        }.items()
    )

    return LaunchDescription(ARGUMENTS + [included_launch])
