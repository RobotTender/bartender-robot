# Named Joint Poses (Joint angles in degrees)
# Order: J1, J2, J3, J4, J5, J6

HOME_POSE = [69.5, -43.0, 102.0, 101.0, -72.0, -213.0]
PICK_PLACE_READY = [28.00, -36.23, 103.71, 78.25, 62.47, -156.73]
CHEERS_POSE = [45.0, 0.0, 135.0, 90.0, -90.0, -135.0]
CONTACT_POSE = [45.00, 43.58, 134.19, 90.01, -90.00, -62.23]
POUR_HORIZONTAL = [42.43, 21.08, 129.85, 87.75, -88.75, -29.06]
POUR_DIAGONAL = [41.83, -5.00, 134.35, 87.99, -87.55, -0.61]
POUR_VERTICAL = [38.76, -35.80, 146.74, 87.76, -84.18, 22.06]
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
        'pos_contact': [314.06, 70.10, 101.99],
        'pos_horizontal': [354.78, 90.82, 226.54],
        'pos_diagonal': [312.68, 48.73, 339.57],
        'pos_vertical': [223.48, -40.48, 373.73]
    },
    'beer': {
        'id': 1,
        'gripper_pos': 550,
        'gripper_force': 150,
        'pos_contact': [319.06, 45.1, 167.0],
        'pos_horizontal': [304.78, 20.82, 251.55],
        'pos_diagonal': [277.66, -6.29, 284.56],
        'pos_vertical': [233.48, -40.48, 333.73]
    },
    'soju': {
        'id': 2,
        'gripper_pos': 400,
        'gripper_force': 175,
        'pos_contact': [314.06, 70.10, 101.99],
        'pos_horizontal': [354.78, 90.82, 226.54],
        'pos_diagonal': [312.68, 48.73, 339.57],
        'pos_vertical': [223.48, -40.48, 373.73]
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
