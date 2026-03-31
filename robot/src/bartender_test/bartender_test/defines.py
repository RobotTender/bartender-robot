# Named Joint Poses (Joint angles in degrees)
# Order: J1, J2, J3, J4, J5, J6
POSJ_HOME = [69.5, -43.0, 102.0, 101.0, -72.0, -213.0]
POSJ_PICK_READY = [28.00, -36.23, 103.71, 78.25, 62.47, -156.73]
POSJ_PLACE_READY = POSJ_PICK_READY
POSJ_CHEERS = [45.0, 0.0, 135.0, 90.0, -90.0, -135.0]

# Final snap pose
POSJ_SNAP = [45.00, 26.25, 131.50, 90.00, -90.00, -112.23]

# Bottle Configurations: [JUICE, BEER, SOJU]
# Each bottle has its own gripper settings and pouring trajectory points
BOTTLE_CONFIG = {
    'juice': {
        'id': 0,
        'gripper_pos': 800,
        'gripper_force': 400,
        'posx_contact': [289.07, 15.11, 266.97, -90.00, 29.99, -135.02],
        'posj_contact': [39.36, 1.43, 144.89, 85.32, -86.88, -93.56],
        'posx_horizontal': [304.40, 24.05, 241.55, -76.45, 90.00, -120.88],
        'posj_horizontal': [38.62, 9.27, 143.31, 83.80, -86.90, -27.29],
        'posx_diagonal': [277.64, -6.28, 294.52, 90.00, 40.00, 45.00],
        'posj_diagonal': [36.62, -8.25, 146.69, 83.71, -84.45, 8.74],
        'posx_vertical': [233.47, -40.50, 323.69, 90.00, -1.0, 45.00],
        'posj_vertical': [36.16, -26.64, 151.69, 84.90, -82.78, 36.38],
        'pour_velocity': 6,
        'pour_acc': 1.5,
        'pour_wait_time': [ 0.0, 1.1, 2.25, 3.25, 6.0, -1 ],
        'grasp_wait_time': 1.5
    },
    'beer': {
        'id': 1,
        'gripper_pos': 570,
        'gripper_force': 90,
        'posx_contact': [279.07, 5.10, 277.00, -90.00, 30.00, -135.01],
        'posj_contact': [38.97, -3.01, 146.55, 85.16, -86.43, -96.32],
        'posx_horizontal': [284.80, 20.82, 261.63, 35.3, 89.99, -9.8],
        'posj_horizontal': [41.25, 2.70, 145.06, 86.83, -88.01, -32.18],
        'posx_diagonal': [257.63, -6.31, 294.54, 90.01, 40.01, 45.00],
        'posj_diagonal': [40.45, -11.27, 148.94, 86.64, -86.94, 7.75],
        'posx_vertical': [228.47, -45.50, 333.69, 90.00, -1.00, 44.99],
        'posj_vertical': [35.69, -30.28, 151.43, 85.15, -82.05, 32.48],
        'pour_velocity': 6,
        'pour_acc': 1.5,
        'pour_wait_time': [ 2.0, 4.8, 6.0, 7.5, 9.0, 11.0 ],
        'grasp_wait_time': 4
    },
    'soju': {
        'id': 2,
        'gripper_pos': 425,
        'gripper_force': 200,
        'posx_contact': [319.05, 45.09, 257.00, -90.00, 29.99, -135.01],
        'posj_contact': [40.29, 9.21, 138.60, 86.02, -87.50, -92.1],
        'posx_horizontal': [344.75, 80.77, 251.54, -84.30, 89.98, -129.30],
        'posj_horizontal': [42.31, 15.26, 131.76, 87.74, -88.53, -32.98],
        'posx_diagonal': [297.65, 33.72, 344.53, 90.01, 40.01, 45.00],
        'posj_diagonal': [41.55, -9.33, 137.00, 87.89, -87.27, -2.28],
        'posx_vertical': [223.47, -30.50, 373.71, 90.00, -1.00, 44.99],
        'posj_vertical': [42.03, -34.12, 146.10, 88.89, -87.25, 23.01],
        'pour_velocity': 4,
        'pour_acc': 1.0,
        'pour_wait_time': [ 0.0, 1.25, 3.0, 4.25, 6.5, 7.8 ],
        'grasp_wait_time': 4
    }
}

# Helper for ID-based lookup (YOLO classes)
BOTTLE_ID_MAP = {str(cfg['id']): name for name, cfg in BOTTLE_CONFIG.items()}

# Backward Compatibility & Easy Access
#INDEX_JUICE = 0
#INDEX_BEER = 1
#INDEX_SOJU = 2

# Gripper Constants (Position 0-1100, Force 0-1000)
GRIPPER_POSITION_OPEN = 0
GRIPPER_FORCE_DEFAULT = 500

# Global Motion Constants
PICK_PLACE_Z = 650
PICK_PLACE_X_OFFSET = 20.0
PICK_PLACE_Y_OFFSET = -60.0

# Snap Recovery Constants
SNAP_VELOCITY = 150.0
SNAP_ACCELERATION = 150.0

VEL_READY, ACC_READY = 100.0, 100.0
VEL_APPROACH, ACC_APPROACH = 100.0, 100.0
VEL_LIFT, ACC_LIFT = 75.0, 75.0
VEL_RETREAT, ACC_RETREAT = 100.0, 100.0