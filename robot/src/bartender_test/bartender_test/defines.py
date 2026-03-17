# Named Joint Poses (Joint angles in degrees)
# Order: J1, J2, J3, J4, J5, J6

HOME_POSE = [69.5, -43.0, 102.0, 101.0, -72.0, -213.0]
PICK_PLACE_READY = [28.0, -35.0, 100.0, 77.0, 63.0, -154.0]
CHEERS_POSE = [45.0, 0.0, 135.0, 90.0, -90.0, -135.0]
CONTACT_POSE = [45.00, 43.58, 134.19, 90.01, -90.00, -62.23]
POUR_HORIZONTAL = [42.43, 21.08, 129.85, 87.75, -88.75, -29.06]
POUR_DIAGONAL = [41.83, -5.00, 134.35, 87.99, -87.55, -0.61]
POUR_VERTICAL = [38.76, -35.80, 146.74, 87.76, -84.18, 22.06]
POLE_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Bottle & Grasp Parameters
BOTTLE_HEIGHT = 240.0        # mm
BOTTLE_FULL_LIQUID = 500.0   # g
BOTTLE_EMPTY_WEIGHT = 0.0    # g (PET bottle is below sensor threshold)
GRASP_Z_FROM_BOTTOM = 80.0   # mm

# Target Pouring
DEFAULT_TARGET_POUR = 50.0   # g

# Cartesian Markers (mm)
POS1_XYZ = [278.48, 34.52, 313.75]   # Red
POS2_XYZ = [314.06, 70.10, 101.99]   # Orange (was mid_p1p2)
POS3_XYZ = [354.78, 90.82, 226.54]   # Yellow (was pos2)
POS4_XYZ = [312.68, 48.73, 339.57]   # Green  (was mid_p2p3)
POS5_XYZ = [223.48, -40.48, 373.73]  # Blue   (was pos3)
