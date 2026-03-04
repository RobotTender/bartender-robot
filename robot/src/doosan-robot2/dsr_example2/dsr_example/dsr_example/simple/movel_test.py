#!/usr/bin/env python3
# ============================================================
#  Copyright (c) 2025 HumanAILab
#
#  Author: HumanAILab
#  Contact: kimrujin32@gmail.com
#
#  All rights reserved.
#
#  This program is protected by copyright law.
#  Unauthorized copying, modification, distribution, or use
#  of this software, via any medium, is strictly prohibited
#  without prior written permission from the copyright holder.
# ============================================================

import rclpy
import DR_init
import sys

def main(args=None):
    rclpy.init(args=args)

    ROBOT_ID = "dsr01"
    ROBOT_MODEL = "e0509"
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    node = rclpy.create_node('example_py', namespace=ROBOT_ID)

    DR_init.__dsr__node = node

    from DSR_ROBOT2 import movej, movel, posj, posx, set_robot_mode, ROBOT_MODE_AUTONOMOUS

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)

    target_pos = posj(0, 0, 90.0, 0, 90.0, 0)
    x1 = [300, 0, 600.0, 0.0, 180.0, 0.0]
    x2 = [600, 0, 300.0, 0.0, 180.0, 0.0]
    x3 = [600, 300, 300.0, 0.0, 180.0, 0.0]
    while rclpy.ok():

        movej(target_pos, vel=100, acc=100)
        movel(x1, vel=100, acc=100)
        movel(x2, vel=100, acc=100)
        movel(x3, vel=100, acc=100)

    print("Example complete")
    rclpy.shutdown()

if __name__ == '__main__':
    main()