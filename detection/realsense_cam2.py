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
        self.current_waterline_y = None # Current highest pixel of liquid

        # 고정 bottom_y
        self.fixed_bottle_bottom_y = None

        # EMA filter
        self.ema_alpha = 0.2
        self.estimated_ml_ema = None

        # --- Automatic Snap Trigger Logic ---
        self.flow_started_pub = self.create_publisher(Empty, '/dsr01/robotender/flow_started', 10)
        self.volume_pub = self.create_publisher(Float32, '/dsr01/robotender/liquid_volume', 10)
        
        # HANDSHAKE: Service to prepare pouring
        self.prepare_srv = self.create_service(Trigger, '/dsr01/robotender/prepare_pouring', self.prepare_pouring_callback)
        # HANDSHAKE: Topic to end pouring
        self.status_sub = self.create_subscription(String, '/dsr01/robotender/pouring_status', self.pouring_status_callback, 10)

        self.snap_triggered = False    # To ensure we only trigger once per pour
        self.flow_started_sent = False # To ensure we only send flow_started once
        self.flow_stability_count = 0  # Counter for consecutive detections
        self.FLOW_STABILITY_THRESHOLD = 1 # 1 frame for immediate responsiveness

        self.low_volume_count = 0      # Used to reset the trigger flag when cup is empty
        
        # --- Stability Logic ---
        self.liquid_stability_count = 0 
        self.STABILITY_THRESHOLD = 5   # number of frames for stabilzation
        
        # --- Tare Safety Logic ---
        self.tare_stability_count = 0 
        self.no_cup_count = 0 
        self.TARE_STABILITY_THRESHOLD = 90 
        self.NO_CUP_THRESHOLD = 45 
        
        # Store last known cup bbox for visualization
        self.last_cup_bbox = None

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

    def prepare_pouring_callback(self, request, response):
        """Service handler: Robot is at CHEERS and wants to start pouring."""
        self.get_logger().info("PREPARE POURING Service Called. Waiting for first liquid detection...")
        
        # 1. Wait for stable cup detection if not locked
        wait_start = time.time()
        while self.locked_total_cup_px is None and (time.time() - wait_start < 2.0):
            time.sleep(0.1)
            
        if self.locked_total_cup_px is None:
            self.get_logger().error("Prepare Failed: Cup not detected/stable.")
            response.success = False
            response.message = "Cup not found."
            return response

        # 2. Lock scale
        self.bottom_y_locked = True
        self.snap_triggered = False
        self.flow_started_sent = False 
        self.flow_stability_count = 0 
        self.liquid_stability_count = 0 
        self.is_pouring_active = True 
        
        self.get_logger().info("Prepare Successful. Ready to detect flow.")
        response.success = True
        response.message = "Ready."
        return response

    def pouring_status_callback(self, msg):
        """Topic handler: Robot says it is finished."""
        if msg.data == 'done':
            self.get_logger().info("Pouring DONE signal received. Resetting state.")
            self.is_pouring_active = False
            self.bottom_y_locked = False
            self.flow_stability_count = 0
            self.liquid_stability_count = 0

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        if self.color_image is None: return
        img_vis = self.color_image.copy()

        results = model.predict(source=img_vis, conf=0.4, iou=0.5, retina_masks=True, verbose=False)
        overlay = img_vis.copy()
        bottle_mask_current = None
        liquid_mask_combined = np.zeros(img_vis.shape[:2], dtype=np.uint8)

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
                else:
                    # Combine all liquid detections
                    liquid_mask_combined = cv2.bitwise_or(liquid_mask_combined, mask_bin)

                color = CLASS_COLORS.get(cls_id, (255,255,255))
                MASK_ALPHA = 0.2
                colored = np.zeros_like(overlay, dtype=np.uint8)
                colored[:, :] = color
                overlay = np.where(mask_bin[:, :, None].astype(bool), ((1 - MASK_ALPHA) * overlay + MASK_ALPHA * colored).astype(np.uint8), overlay)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, f"{class_name}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Calibration Logic
        if bottle_mask_current is not None and self.last_cup_bbox is not None:
            self.no_cup_count = 0 
            if not self.bottom_y_locked:
                self.tare_stability_count += 1
                if self.tare_stability_count >= self.TARE_STABILITY_THRESHOLD:
                    cx1, cy1, cx2, cy2 = self.last_cup_bbox
                    self.fixed_bottle_bottom_y, self.locked_total_cup_px = int(cy2), float(cy2 - cy1)
        else:
            self.no_cup_count += 1
            if self.no_cup_count >= self.NO_CUP_THRESHOLD:
                self.fixed_bottle_bottom_y, self.locked_total_cup_px, self.bottom_y_locked = None, None, False
                self.is_pouring_active = False
        
        # Liquid Logic
        has_liquid = np.any(liquid_mask_combined > 0)
        if self.fixed_bottle_bottom_y is not None:
            if has_liquid:
                height_px, waterline_y = self.estimate_height_px(liquid_mask_combined)
                self.current_waterline_y = waterline_y 

            # --- Simple First Liquid Detection Trigger ---
            if self.is_pouring_active and not self.flow_started_sent:
                if has_liquid:
                    self.flow_stability_count += 1
                    if self.flow_stability_count >= self.FLOW_STABILITY_THRESHOLD:
                        self.flow_started_pub.publish(Empty())
                        self.flow_started_sent = True
                        self.get_logger().info("--- FLOW STARTED! Liquid detected by YOLO ---")
                else:
                    self.flow_stability_count = 0 

            # Report volume
            if has_liquid:
                self.liquid_stability_count += 1
                if self.liquid_stability_count >= self.STABILITY_THRESHOLD:
                    if height_px is not None:
                        height_px_ema = self.apply_height_ema(height_px)
                        volume_ml = self.height_px_to_volume_ml(height_px_ema)
                        self.estimated_ml_ema = self.apply_liquid_ml_ema(volume_ml)
                        vol_msg = Float32()
                        vol_msg.data = float(self.estimated_ml_ema)
                        self.volume_pub.publish(vol_msg)
        else:
            self.current_waterline_y = None

        # Visualization
        if self.last_cup_bbox is not None:
            cx1, cy1, cx2, cy2 = self.last_cup_bbox
            current_vol = self.estimated_ml_ema if self.estimated_ml_ema is not None else 0.0
            cv2.putText(overlay, f"CUR: {current_vol:.1f}ml", (cx1, cy2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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
