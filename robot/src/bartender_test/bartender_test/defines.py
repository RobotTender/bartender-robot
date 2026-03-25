# Named Joint Poses (Joint angles in degrees)
# Order: J1, J2, J3, J4, J5, J6
POSJ_HOME = [69.5, -43.0, 102.0, 101.0, -72.0, -213.0]
POSJ_PICK_PLACE_READY = [28.00, -36.23, 103.71, 78.25, 62.47, -156.73]
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
        'pour_target_ml': 100.0,
        'posx_contact': [319.04, 45.08, 197.00, -90.00, 29.99, -135.02],
        'posj_contact': [40.29, 21.94, 139.45, 85.55, -88.50, -78.56],
        'posx_horizontal': [304.78, 20.82, 251.55, -88.05, 89.99, -133.0],
        'posj_horizontal': [37.99, 7.03, 143.01, 83.92, -86.51, -29.79],
        'posx_diagonal': [277.66, -6.29, 284.56, 90.00, 40.00, 45.00],
        'posj_diagonal': [36.62, -6.09, 147.33, 83.45, -84.76, 11.53],
        'posx_vertical': [233.48, -40.48, 333.73, 90.00, -1.01, 44.99],
        'posj_vertical': [36.16, -28.40, 150.63, 85.26, -82.54, 33.54],
        'pour_velocity': 10,
        'pour_acc': 5,
        'pour_wait_time': [ 0.0, 0.25, 0.5, 0.75, 1.0, 1.25 ]
    },
    'beer': {
        'id': 1,
        'gripper_pos': 550,
        'gripper_force': 75,
        'pour_target_ml': 100.0,
        'posx_contact': [319.06, 45.10, 177.00, -90.00, 30.00, -135.01],
        'posj_contact': [40.29, 26.46, 139.23, 85.45, -88.84, -74.27],
        'posx_horizontal': [294.76, 10.80, 261.57, -73.25, 89.99, -118.25],
        'posj_horizontal': [37.54, 2.90, 144.85, 83.68, -86.03, -32.04],
        'posx_diagonal': [267.63, -16.31, 284.54, 90.01, 40.01, 45.00],
        'posj_diagonal': [35.96, -8.67, 149.38, 82.99, -84.29, 11.05],
        'posx_vertical': [228.47, -45.50, 333.69, 90.00, -1.00, 44.99],
        'posj_vertical': [35.69, -30.28, 151.43, 85.15, -82.05, 32.48],
        'pour_velocity': 5,
        'pour_acc': 5,
        'pour_wait_time': [ 0.0, 0.25, 0.5, 0.75, 1.0, 1.25 ]
    },
    'soju': {
        'id': 2,
        'gripper_pos': 400,
        'gripper_force': 175,
        'pour_target_ml': 50.0,
        'posx_contact': [329.05, 55.09, 117.00, -90.00, 29.99, -135.01],
        'posj_contact': [40.53, 40.36, 134.88, 85.56, -89.63, -64.75],
        'posx_horizontal': [354.75, 70.77, 231.54, -77.55, 89.98, -122.55],
        'posj_horizontal': [39.63, 18.89, 132.00, 85.30, -87.39, -29.02],
        'posx_diagonal': [307.65, 23.72, 334.53, 90.01, 40.01, 45.00],
        'posj_diagonal': [38.12, -7.68, 137.65, 85.57, -84.73, 0.16],
        'posx_vertical': [233.47, -40.50, 373.69, 90.00, -1.00, 44.99],
        'posj_vertical': [36.16, -33.75, 145.95, 86.64, -81.83, 23.45],
        'pour_velocity': 6,
        'pour_acc': 1.5,
        'pour_wait_time': [ 0.0, 1.2, 2.0, 0.75, 1.0, 1.25 ]
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
