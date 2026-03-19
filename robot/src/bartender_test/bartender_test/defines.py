# Named Joint Poses (Joint angles in degrees)
# Order: J1, J2, J3, J4, J5, J6

HOME_POSE = [69.5, -43.0, 102.0, 101.0, -72.0, -213.0]
PICK_PLACE_READY = [28.00, -36.23, 103.71, 78.25, 62.47, -156.73]
CHEERS_POSE = [45.0, 0.0, 135.0, 90.0, -90.0, -135.0]
POLE_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Cartesian Markers (mm)
POS_CHEERS = [278.48, 34.52, 313.75]   # Red (Entry position - Shared)

# Bottle Configurations: [JUICE, BEER, SOJU]
# Each bottle has its own gripper settings and pouring trajectory points
BOTTLE_CONFIG = {
    'juice': {
        'id': 0,
        'gripper_pos': 800,
        'gripper_force': 400,
        'posx_contact': [319.06, 45.10, 177.00, -90.00, 30.00, -135.01],
        'posj_contact': [40.29, 26.46, 139.23, 85.45, -88.84, -74.27],
        'posx_horizontal': [304.78, 20.82, 251.55, -88.05, 89.99, -133.0],
        'posj_horizontal': [37.99, 7.03, 143.01, 83.92, -86.51, -29.79],
        'posx_diagonal': [277.66, -6.29, 284.56, 90.00, 40.00, 45.00],
        'posj_diagonal': [36.62, -6.09, 147.33, 83.45, -84.76, 11.53],
        'posx_vertical': [233.48, -40.48, 333.73, 90.00, -1.01, 44.99],
        'posj_vertical': [36.16, -28.40, 150.63, 85.26, -82.54, 33.54]
    },
    'beer': {
        'id': 1,
        'gripper_pos': 550,
        'gripper_force': 150,
        'posx_contact': [319.06, 45.10, 177.00, -90.00, 30.00, -135.01],
        'posj_contact': [40.29, 26.46, 139.23, 85.45, -88.84, -74.27],
        'posx_horizontal': [304.78, 20.82, 251.55, -88.05, 89.99, -133.0],
        'posj_horizontal': [37.99, 7.03, 143.01, 83.92, -86.51, -29.79],
        'posx_diagonal': [277.66, -6.29, 284.56, 90.00, 40.00, 45.00],
        'posj_diagonal': [36.62, -6.09, 147.33, 83.45, -84.76, 11.53],
        'posx_vertical': [233.48, -40.48, 333.73, 90.00, -1.01, 44.99],
        'posj_vertical': [36.16, -28.40, 150.63, 85.26, -82.54, 33.54]
    },
    'soju': {
        'id': 2,
        'gripper_pos': 400,
        'gripper_force': 175,
        'posx_contact': [319.06, 45.10, 177.00, -90.00, 30.00, -135.01],
        'posj_contact': [40.29, 26.46, 139.23, 85.45, -88.84, -74.27],
        'posx_horizontal': [304.78, 20.82, 251.55, -88.05, 89.99, -133.0],
        'posj_horizontal': [37.99, 7.03, 143.01, 83.92, -86.51, -29.79],
        'posx_diagonal': [277.66, -6.29, 284.56, 90.00, 40.00, 45.00],
        'posj_diagonal': [36.62, -6.09, 147.33, 83.45, -84.76, 11.53],
        'posx_vertical': [233.48, -40.48, 333.73, 90.00, -1.01, 44.99],
        'posj_vertical': [36.16, -28.40, 150.63, 85.26, -82.54, 33.54]
    }
}

# Helper for ID-based lookup (YOLO classes)
BOTTLE_ID_MAP = {str(cfg['id']): name for name, cfg in BOTTLE_CONFIG.items()}

# Backward Compatibility & Easy Access
INDEX_JUICE = 0
INDEX_BEER = 1
INDEX_SOJU = 2

# Gripper Constants (Position 0-1100, Force 0-1000)
GRIPPER_POSITION_OPEN = 0
GRIPPER_FORCE_OPEN = 75

ORDER_TOPIC = "/bartender/order_detail"

# Global Motion Constants
PICK_PLACE_Z = 650
PICK_PLACE_X_OFFSET = 20.0
PICK_PLACE_Y_OFFSET = -60.0
