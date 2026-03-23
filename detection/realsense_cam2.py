import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, Float32, String
from std_srvs.srv import Trigger

import cv2
import numpy as np
import time
from ultralytics import YOLO


# ✅ YOLO Segmentation 모델로 교체하세요 (예: yolo11n-seg.pt 또는 커스텀 best.pt)
# model = YOLO("yolo11n-seg.pt")
model = YOLO("/home/fastcampus/bartender-robot/detection/weights/cam_2.pt")  # Updated path to match workspace

# 클래스별 색 (BGR)
CLASS_COLORS = {
    0: (0, 0, 255),   # 빨강
    1: (255, 0, 0),   # 파랑
}


class DepthReader(Node):
    def __init__(self):
        super().__init__('depth_reader')
        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None

        self.depth_scale = 0.001  # 16UC1(mm) -> m 스케일 (환경에 맞게)

        # class name
        self.cup_class_name = "cup"

        # --- Ratio-Based Volume Lookup Table ---
        self.ratio_to_ml_table = np.array([
            [0.0000, 0],
            [0.1186, 50.0],
            [0.2260, 100.0],
            [0.3169, 150.0],
            [0.4002, 200.0],
            [0.4844, 250.0],
            [0.5657, 300.0],
            [0.6348, 350.0],
            [0.6891, 400.0],
            [0.7748, 450.0],
            [0.8434, 500.0]
        ])
        # --------------------------------------

        self.locked_total_cup_px = None # Reference scale locked during Tare
        self.bottom_y_locked = False
        self.is_pouring_active = False # MASTER GATE: Only measure when robot says so

        # height EMA
        self.height_ema_alpha = 0.2
        self.height_px_ema = None
        self.current_height_px_ema = None

        # 고정 bottom_y
        self.fixed_bottle_bottom_y = None

        # EMA filter
        self.ema_alpha = 0.2
        self.estimated_ml_ema = None
        self.last_total_cup_px = 1.0 # Initialize to 1 to avoid div by zero

        # --- Automatic Snap Trigger Logic ---
        self.trigger_pub = self.create_publisher(Empty, '/dsr01/robotender_snap/trigger', 10)
        self.volume_pub = self.create_publisher(Float32, '/dsr01/robotender/liquid_volume', 10)
        
        # Subscriber for dynamic target volume (increment)
        self.target_ml_sub = self.create_subscription(Float32, '/detection/cup_target_volume', self.target_volume_callback, 10)
        
        # HANDSHAKE: Service to prepare pouring
        self.prepare_srv = self.create_service(Trigger, '/dsr01/robotender/prepare_pouring', self.prepare_pouring_callback)
        # HANDSHAKE: Topic to end pouring
        self.status_sub = self.create_subscription(String, '/dsr01/robotender/pouring_status', self.pouring_status_callback, 10)

        self.target_volume_ml = 0.0     # Incremental target (ml)
        self.start_volume_ml = 0.0     # Volume at the start of the pour
        self.target_total_volume_ml = 0.0 # start_volume + target_volume (Absolute Goal)
        self.target_line_y = None      # Y-coordinate for the target visualization line
        
        self.snap_triggered = False    # To ensure we only trigger once per pour
        self.low_volume_count = 0      # Used to reset the trigger flag when cup is empty
        
        # --- Stability Logic ---
        self.liquid_stability_count = 0 
        self.STABILITY_THRESHOLD = 5   # number of frames for stabilzation, avoid reflect glitches
        
        # --- Tare Safety Logic ---
        self.tare_stability_count = 0  # To ensure we only tare on a truly empty cup
        self.no_cup_count = 0          # To reset state when cup is removed
        self.TARE_STABILITY_THRESHOLD = 90 # ~3 seconds at 30fps
        self.NO_CUP_THRESHOLD = 45         # 1.5 seconds at 30fps
        
        # Store last known cup bbox for line visualization
        self.last_cup_bbox = None
        self.log_counter = 0

        # --- Forward Snapping Parameters ---
        self.SNAP_RATIO = 0.1 # Trigger snap when reaching 10% of the target volume increment
        # -----------------------------------

        # 구독자 설정
        self.create_subscription(Image, '/camera/camera_2/color/image_raw', self.color_callback, 10)
        self.create_subscription(Image, '/camera/camera_2/aligned_depth_to_color/image_raw', self.depth_callback, 10)

    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def apply_height_ema(self, value: float):
        if value is None:
            return self.height_px_ema
        if self.height_px_ema is None:
            self.height_px_ema = value
        else:
            self.height_px_ema = (self.height_ema_alpha * value + (1.0 - self.height_ema_alpha) * self.height_px_ema)
        return self.height_px_ema

    def estimate_height_px(self, liquid_mask: np.ndarray):
        ys_l, _ = np.where(liquid_mask > 0)
        if len(ys_l) == 0: return None, None
        if self.fixed_bottle_bottom_y is None: return None, None

        waterline_y = int(np.min(ys_l))
        height_px = self.fixed_bottle_bottom_y - waterline_y
        if height_px < 0: return 0.0, waterline_y
        return float(height_px), waterline_y

    def height_px_to_volume_ml(self, height_px: float):
        if self.locked_total_cup_px is None or self.locked_total_cup_px <= 0 or height_px <= 0:
            return 0.0
        current_ratio = height_px / self.locked_total_cup_px
        ratios = self.ratio_to_ml_table[:, 0]
        volumes = self.ratio_to_ml_table[:, 1]
        vol_ml = np.interp(current_ratio, ratios, volumes)
        return float(vol_ml)

    def apply_liquid_ml_ema(self, value: float):
        if value is None: return self.estimated_ml_ema
        if self.estimated_ml_ema is None:
            self.estimated_ml_ema = value
        else:
            self.estimated_ml_ema = (self.ema_alpha * value + (1.0 - self.ema_alpha) * self.estimated_ml_ema)
        return self.estimated_ml_ema

    def _update_target_line_y(self):
        if (self.target_total_volume_ml > 0 and self.locked_total_cup_px is not None and self.fixed_bottle_bottom_y is not None):
            ratios = self.ratio_to_ml_table[:, 0]
            volumes = self.ratio_to_ml_table[:, 1]
            target_ratio = np.interp(self.target_total_volume_ml, volumes, ratios)
            target_h_px = target_ratio * self.locked_total_cup_px
            self.target_line_y = int(self.fixed_bottle_bottom_y - target_h_px)
        else:
            self.target_line_y = None

    def target_volume_callback(self, msg):
        val = float(msg.data)
        if val <= 0.0:
            self.target_volume_ml = 0.0
            return
        self.target_volume_ml = val
        self.get_logger().info(f"Target Volume Set: +{self.target_volume_ml}ml (Waiting for Prepare Service)")

    def prepare_pouring_callback(self, request, response):
        """Service handler: Robot is at CHEERS and wants to start pouring."""
        self.get_logger().info("PREPARE POURING Service Called. Calibrating baseline...")
        
        # 1. Wait for stable cup detection if not locked
        wait_start = time.time()
        while self.locked_total_cup_px is None and (time.time() - wait_start < 2.0):
            time.sleep(0.1)
            
        if self.locked_total_cup_px is None:
            self.get_logger().error("Prepare Failed: Cup not detected/stable.")
            response.success = False
            response.message = "Cup not found."
            return response

        # 2. Lock scale and snapshot baseline
        self.bottom_y_locked = True
        self.start_volume_ml = self.estimated_ml_ema if self.estimated_ml_ema is not None else 0.0
        self.target_total_volume_ml = self.start_volume_ml + self.target_volume_ml
        
        # 3. Finalize goal visualization
        self._update_target_line_y()
        self.snap_triggered = False
        self.liquid_stability_count = 0 # Reset for new pour
        self.is_pouring_active = True # MASTER GATE OPEN
        
        self.get_logger().info(f"Prepare Successful. Baseline: {self.start_volume_ml:.1f}ml, Goal: {self.target_total_volume_ml:.1f}ml")
        response.success = True
        response.message = f"Ready. Goal: {self.target_total_volume_ml:.1f}ml"
        return response

    def pouring_status_callback(self, msg):
        """Topic handler: Robot says it is finished."""
        if msg.data == 'done':
            self.get_logger().info("Pouring DONE signal received. Resetting state.")
            self.is_pouring_active = False
            self.bottom_y_locked = False
            self.target_volume_ml = 0.0
            self.target_total_volume_ml = 0.0
            self.target_line_y = None
            self.liquid_stability_count = 0

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        if self.color_image is None: return
        img_vis = self.color_image.copy()

        results = model.predict(source=img_vis, conf=0.6, iou=0.5, retina_masks=True, verbose=False) # 0327: 0.4 -> 0.6
        overlay = img_vis.copy()
        bottle_mask_current = None
        liquid_mask_current = None

        for result in results:
            boxes = result.boxes
            masks = result.masks
            if boxes is None or len(boxes) == 0 or masks is None: continue
            mdata = masks.data.detach().cpu().numpy() if hasattr(masks.data, 'detach') else np.asarray(masks.data)
            H, W = img_vis.shape[:2]

            for i, b in enumerate(boxes):
                cls_id = int(b.cls[0].cpu().numpy())
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                mi = mdata[i]
                if mi.shape != (H, W): mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_bin = (mi > 0.5).astype(np.uint8)
                mask_area = int(np.count_nonzero(mask_bin))
                class_name = model.names.get(cls_id, cls_id)

                if class_name == self.cup_class_name:
                    if bottle_mask_current is None or mask_area > int(np.count_nonzero(bottle_mask_current)):
                        bottle_mask_current = mask_bin.copy()
                        self.last_cup_bbox = (x1, y1, x2, y2)
                if class_name != self.cup_class_name:
                    if liquid_mask_current is None or mask_area > int(np.count_nonzero(liquid_mask_current)):
                        liquid_mask_current = mask_bin.copy()

                color = CLASS_COLORS.get(cls_id, (255,255,255))
                MASK_ALPHA = 0.2
                colored = np.zeros_like(overlay, dtype=np.uint8)
                colored[:, :] = color
                overlay = np.where(mask_bin[:, :, None].astype(bool), ((1 - MASK_ALPHA) * overlay + MASK_ALPHA * colored).astype(np.uint8), overlay)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, f"{class_name}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Calibration Logic (Always run while Idle)
        if bottle_mask_current is not None and self.last_cup_bbox is not None:
            self.no_cup_count = 0 
            if not self.bottom_y_locked:
                self.tare_stability_count += 1
                if self.tare_stability_count >= self.TARE_STABILITY_THRESHOLD:
                    cx1, cy1, cx2, cy2 = self.last_cup_bbox
                    self.fixed_bottle_bottom_y, self.locked_total_cup_px = int(cy2), float(cy2 - cy1)
                    self.last_total_cup_px = self.locked_total_cup_px
                    self._update_target_line_y()
        else:
            self.no_cup_count += 1
            if self.no_cup_count >= self.NO_CUP_THRESHOLD:
                self.fixed_bottle_bottom_y, self.locked_total_cup_px, self.bottom_y_locked = None, None, False
                self.snap_triggered, self.height_px_ema, self.estimated_ml_ema, self.target_line_y = False, None, None, None
                self.is_pouring_active = False
                self.liquid_stability_count = 0
        
        # Liquid Logic (Temporal Stability + Active Pouring Check)
        if liquid_mask_current is not None and self.fixed_bottle_bottom_y is not None:
            self.liquid_stability_count += 1
            if self.liquid_stability_count >= self.STABILITY_THRESHOLD:
                height_px, _ = self.estimate_height_px(liquid_mask_current)
                if height_px is not None:
                    height_px_ema = self.apply_height_ema(height_px)
                    volume_ml = self.height_px_to_volume_ml(height_px_ema)
                    volume_ml_ema = self.apply_liquid_ml_ema(volume_ml)
                    self.current_height_px_ema = height_px_ema

                    vol_msg = Float32()
                    vol_msg.data = float(volume_ml_ema)
                    self.volume_pub.publish(vol_msg)

                    if self.is_pouring_active and self.target_total_volume_ml > 0:
                        # Forward Snapping Trigger Logic
                        # Triggers when: (Current Volume - Start Volume) >= (Target Increment * Ratio)
                        current_increment = volume_ml_ema - self.start_volume_ml
                        trigger_threshold_ml = self.target_volume_ml * self.SNAP_RATIO
                        
                        if current_increment >= trigger_threshold_ml and not self.snap_triggered:
                            self.trigger_pub.publish(Empty())
                            self.snap_triggered = True
                            self.get_logger().info(f"--- FORWARD SNAP TRIGGERED (Ratio: {self.SNAP_RATIO}): {volume_ml_ema:.1f}ml (Goal: {self.target_total_volume_ml:.1f}, Start: {self.start_volume_ml:.1f}) ---")
        else:
            self.liquid_stability_count = 0

        # Visualization
        if self.last_cup_bbox is not None:
            cx1, _, _, cy2 = self.last_cup_bbox
            current_vol = self.estimated_ml_ema if self.estimated_ml_ema is not None else 0.0
            cv2.putText(overlay, f"CUR: {current_vol:.1f}ml", (cx1, cy2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if self.target_line_y is not None and self.fixed_bottle_bottom_y is not None and self.last_cup_bbox is not None:
            H, W = overlay.shape[:2]
            if 0 < self.target_line_y < H:
                cx1, _, cx2, _ = self.last_cup_bbox
                center_x = (cx1 + cx2) // 2
                x_start, x_end = max(0, center_x - int((cx2-cx1)*0.75)), min(W, center_x + int((cx2-cx1)*0.75))
                cv2.line(overlay, (x_start, self.target_line_y), (x_end, self.target_line_y), (0, 255, 0), 1)
                cv2.putText(overlay, f"FILL TO: {self.target_total_volume_ml:.1f}ml", (x_start, self.target_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Status Display
        status_msg = "POURING" if self.is_pouring_active else ("READY" if self.locked_total_cup_px else "SEARCHING")
        status_color = (0, 255, 0) if self.is_pouring_active else ((0, 255, 255) if self.locked_total_cup_px else (0, 0, 255))
        cv2.putText(overlay, f"STATUS: {status_msg}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow("RGB with Depth (YOLO-Seg)", overlay)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DepthReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok(): rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
