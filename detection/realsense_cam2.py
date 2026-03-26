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
        self.FLOW_STABILITY_THRESHOLD = 1  # Keep at 1 for immediate responsiveness
        self.PADDING_RATIO = 0.05          # 5% height padding between liquid and detection zone

        self.baseline_height_px = 0.0

        self.baseline_liquid_mask = None
        self.available_mask = None
        self.last_height_px = 0.0
        
        # --- Noise Filtering Logic ---
        self.MIN_LIQUID_RATIO = 0.05      # Ignore liquid detections smaller than 2% of cup mask (reflections)
        self.liquid_present_frames = 0    # Stable detection count
        self.liquid_absent_frames = 0     # Consistent absence count
        self.STABILITY_CONSENSUS = 10     # Need 10 frames of stable liquid to trust it for baseline
        self.ABSENCE_RESET_THRESHOLD = 5  # Frames of no liquid to force reset EMAs to zero

        # --- Stability Logic ---
        self.liquid_stability_count = 0 
        self.STABILITY_THRESHOLD = 10   # number of frames for stabilzation
        
        # --- Tare Safety Logic ---
        self.tare_stability_count = 0 
        self.no_cup_count = 0 
        self.TARE_STABILITY_THRESHOLD = 45
        self.NO_CUP_THRESHOLD = 15
        
        # Store last known masks for handshake
        self.last_cup_mask = None
        self.last_liquid_mask = None
        self.last_cup_bbox = None

        # 구독자 설정 (QoS: Keep Last 1 to avoid lag)
        qos_profile = rclpy.qos.QoSProfile(depth=1)
        self.create_subscription(Image, '/camera/camera_2/color/image_raw', self.color_callback, qos_profile)
        self.create_subscription(Image, '/camera/camera_2/aligned_depth_to_color/image_raw', self.depth_callback, qos_profile)

        # 윈도우 초기화
        cv2.namedWindow("RGB with Depth (YOLO-Seg)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RGB with Depth (YOLO-Seg)", 850, 500)

        # 타이머 기반 처리 (15 FPS 정도로 안정화)
        self.timer = self.create_timer(1.0/15.0, self.timer_callback)

    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

    def apply_height_ema(self, value: float):
        if value is None:
            return self.height_px_ema
        if self.height_px_ema is None:
            self.height_px_ema = value
        else:
            self.height_px_ema = (self.height_ema_alpha * value + (1.0 - self.height_ema_alpha) * self.height_px_ema)
        return self.height_px_ema

    def estimate_height_px(self, liquid_mask: np.ndarray):
        # Only consider liquid that is INSIDE the cup mask to avoid bottle/stream noise
        if self.last_cup_mask is not None:
            cup_liquid_mask = cv2.bitwise_and(liquid_mask, self.last_cup_mask)
        else:
            cup_liquid_mask = liquid_mask

        ys_l, _ = np.where(cup_liquid_mask > 0)
        if len(ys_l) == 0: return None, None
        if self.fixed_bottle_bottom_y is None: return None, None

        waterline_y = int(np.min(ys_l))
        height_px = self.fixed_bottle_bottom_y - waterline_y
        
        # Noise floor: If height is less than 5px, it's likely sensor jitter at the bottom
        if height_px < 5: return 0.0, waterline_y
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
        self.get_logger().info("PREPARE POURING Service Called. Defining available area with padding...")
        
        # 1. Wait for stable cup detection if not locked
        wait_start = time.time()
        while self.locked_total_cup_px is None and (time.time() - wait_start < 2.0):
            time.sleep(0.1)
            
        if self.locked_total_cup_px is None or self.last_cup_mask is None:
            self.get_logger().error("Prepare Failed: Cup not detected/stable.")
            response.success = False
            response.message = "Cup not found."
            return response

        # 2. Lock scale & Baseline Area
        self.bottom_y_locked = True
        self.snap_triggered = False
        self.flow_started_sent = False 
        self.flow_stability_count = 0 
        self.liquid_stability_count = 0 
        self.is_pouring_active = True 
        
        # CONSENSUS: Only set non-zero baseline if liquid has been stable for STABILITY_CONSENSUS frames
        is_liquid_stable = self.liquid_present_frames >= self.STABILITY_CONSENSUS
        
        if is_liquid_stable and self.height_px_ema is not None:
            self.baseline_height_px = self.height_px_ema
            self.get_logger().info(f"Baseline set from stable liquid: {self.baseline_height_px:.1f}px")
        else:
            self.baseline_height_px = 0.0
            self.height_px_ema = 0.0 
            self.estimated_ml_ema = 0.0
            if self.liquid_present_frames > 0:
                self.get_logger().warn(f"Liquid detected but not stable ({self.liquid_present_frames}/{self.STABILITY_CONSENSUS}). Forcing empty baseline.")
            else:
                self.get_logger().info("Empty baseline set.")
            
        # Define Available Area: Cup Mask MINUS Current Liquid Mask
        has_liquid = self.last_liquid_mask is not None and np.any(self.last_liquid_mask > 0)
        if has_liquid and is_liquid_stable:
            self.baseline_liquid_mask = self.last_liquid_mask.copy()
            # 1. Start with Cup MINUS Liquid
            available = cv2.bitwise_and(self.last_cup_mask, cv2.bitwise_not(self.baseline_liquid_mask))
            
            # 2. Add Padding (Clear area just above the current surface to avoid ripple noise)
            padding_px = int(self.PADDING_RATIO * self.locked_total_cup_px)
            if padding_px > 0:
                ys_l, _ = np.where(self.baseline_liquid_mask > 0)
                if len(ys_l) > 0:
                    waterline_y = int(np.min(ys_l))
                    # Clear pixels in a strip of padding_px height above the waterline
                    available[max(0, waterline_y - padding_px):waterline_y, :] = 0
            
            self.available_mask = available
        else:
            # Force full cup as available if liquid wasn't stable
            self.baseline_liquid_mask = np.zeros_like(self.last_cup_mask)
            self.available_mask = self.last_cup_mask.copy()
            
        self.get_logger().info(f"Prepare Successful. Baseline: {self.baseline_height_px:.1f}px. Available area defined.")
        response.success = True
        response.message = f"Ready. Baseline {self.baseline_height_px:.1f}px"
        return response

    def pouring_status_callback(self, msg):
        """Topic handler: Robot says it is finished."""
        if msg.data == 'done':
            self.get_logger().info("Pouring DONE signal received. Resetting state.")
            self.is_pouring_active = False
            self.bottom_y_locked = False
            self.flow_stability_count = 0
            self.liquid_stability_count = 0
            self.baseline_height_px = 0.0
            self.available_mask = None
            # Reset EMAs so next pour starts fresh
            self.height_px_ema = None
            self.estimated_ml_ema = None
            self.liquid_present_frames = 0
            self.liquid_absent_frames = 10 # Start as absent

    def timer_callback(self):
        if self.color_image is None or self.depth_image is None:
            return

        img_vis = self.color_image.copy()
        depth_img = self.depth_image.copy() # Local copy to avoid thread collision

        results = model.predict(source=img_vis, conf=0.5, iou=0.5, retina_masks=True, verbose=False)
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
                        self.last_cup_mask = bottle_mask_current # Update for handshake
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

        self.last_liquid_mask = liquid_mask_combined # Update for handshake

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
                self.last_cup_bbox = None # Explicitly clear last known bbox when calibration is lost
        
        # Liquid Ratio Filter
        has_liquid = False
        if self.last_cup_mask is not None:
            liquid_area = int(np.count_nonzero(liquid_mask_combined))
            cup_area = int(np.count_nonzero(self.last_cup_mask))
            if cup_area > 0:
                liquid_ratio = liquid_area / cup_area
                has_liquid = liquid_ratio > self.MIN_LIQUID_RATIO
        
        # Stability State Update
        if has_liquid:
            self.liquid_present_frames += 1
            self.liquid_absent_frames = 0
        else:
            self.liquid_present_frames = 0
            self.liquid_absent_frames += 1
            # Aggressive Reset: If liquid is absent for long enough, clear EMA history
            if self.liquid_absent_frames >= self.ABSENCE_RESET_THRESHOLD:
                self.height_px_ema = 0.0
                self.estimated_ml_ema = 0.0

        height_px = None
        if self.fixed_bottle_bottom_y is not None:
            if has_liquid:
                height_px, waterline_y = self.estimate_height_px(liquid_mask_combined)
                self.current_waterline_y = waterline_y 

            # --- Spatial Flow Detection Strategy (Strict Available Area) ---
            if self.is_pouring_active and not self.flow_started_sent:
                if self.available_mask is not None:
                    # Only consider liquid that appears in the previously empty 'available' area
                    new_liquid_in_available_area = cv2.bitwise_and(liquid_mask_combined, self.available_mask)
                    new_liquid_px_count = int(np.count_nonzero(new_liquid_in_available_area))
                    
                    # Threshold for detection
                    if new_liquid_px_count > 100:
                        self.flow_stability_count += 1
                        if self.flow_stability_count >= self.FLOW_STABILITY_THRESHOLD:
                            self.flow_started_pub.publish(Empty())
                            self.flow_started_sent = True
                            self.get_logger().info(f"--- FLOW STARTED! New liquid detected in available area ({new_liquid_px_count} px) ---")
                    else:
                        self.flow_stability_count = 0
                else:
                    self.get_logger().warn("Pouring active but available_mask is None. Handshake might have failed.")

            # Report volume
            if has_liquid and height_px is not None:
                self.liquid_stability_count += 1
                if self.liquid_stability_count >= self.STABILITY_THRESHOLD:
                    height_px_ema = self.apply_height_ema(height_px)
                    volume_ml = self.height_px_to_volume_ml(height_px_ema)
                    self.estimated_ml_ema = self.apply_liquid_ml_ema(volume_ml)
                    vol_msg = Float32()
                    vol_msg.data = float(self.estimated_ml_ema)
                    self.volume_pub.publish(vol_msg)
            else:
                self.liquid_stability_count = 0
                self.estimated_ml_ema = 0.0  # Reset volume if no liquid detected
        else:
            self.current_waterline_y = None
            self.estimated_ml_ema = None  # Reset volume if no cup/scale

        # Visualization for available area
        if self.available_mask is not None and self.is_pouring_active:
            # Draw available area as a subtle green overlay
            colored_available = np.zeros_like(overlay)
            colored_available[self.available_mask > 0] = (0, 100, 0)
            overlay = cv2.addWeighted(overlay, 1.0, colored_available, 0.3, 0)

        # Visualization for volume
        if bottle_mask_current is not None and self.last_cup_bbox is not None:
            cx1, cy1, cx2, cy2 = self.last_cup_bbox
            if has_liquid and self.estimated_ml_ema is not None:
                vol_text = f"CUR: {self.estimated_ml_ema:.1f}ml"
            else:
                vol_text = "CUR: 0.0ml"
            cv2.putText(overlay, vol_text, (cx1, cy2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Immediate feedback: If cup is not detected in current frame, show N/A
            cv2.putText(overlay, "CUR: N/A", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
