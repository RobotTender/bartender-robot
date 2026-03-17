# Named Joint Poses (Joint angles in degrees)
# Order: J1, J2, J3, J4, J5, J6

HOME_POSE = [69.5, -43.0, 102.0, 101.0, -72.0, -213.0]
PICK_PLACE_READY = [28.0, -35.0, 100.0, 77.0, 63.0, -154.0]
PMID_POSE = [61.0, -20.0, 97.0, 96.0, -63.0, -195.0]
CHEERS_POSE = [45.0, 0.0, 135.0, 90.0, -90.0, -135.0]
CONTACT_POSE = [45.00, 31.00, 138.90, 90.01, -90.00, -70.10]
POUR_HORIZONTAL = [45.00, 13.63, 137.92, 90.00, -90.00, -28.46]
POUR_DIAGONAL = [45.00, -5.81, 143.54, 90.00, -90.00, 7.72]
POUR_VERTICAL = [45.00, -26.99, 149.99, 90.00, -90.00, 34.01]
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
POS2_XYZ = [304.04, 60.08, 156.99]   # Orange (was mid_p1p2)
POS3_XYZ = [309.78, 65.83, 239.06]   # Yellow (was pos2)
POS4_XYZ = [272.68, 28.72, 299.58]   # Green  (was mid_p2p3)
POS5_XYZ = [223.48, -20.48, 333.75]  # Blue   (was pos3)
