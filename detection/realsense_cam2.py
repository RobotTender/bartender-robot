import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, Float32

import cv2
import random
import numpy as np
from ultralytics import YOLO


# ✅ YOLO Segmentation 모델로 교체하세요 (예: yolo11n-seg.pt 또는 커스텀 best.pt)
# model = YOLO("yolo11n-seg.pt")
model = YOLO("/home/fastcampus/bartender-robot/detection/weights/cam_2.pt")  # Updated path to match workspace

# 클래스별 색 (BGR)
CLASS_COLORS = {
    0: (0, 0, 255),   # 빨강
    1: (255, 0, 0),   # 파랑
    # 2: (0, 255, 0)    # 초록
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

        # 소주병처럼 비원통형 bottle이면 lookup table 권장 (Original Version Reference)
        self.known_heights_px = np.array([0, 26.3, 50, 67.4, 84, 100, 115.7, 131.9, 140.1, 158, 174], dtype=np.float32)
        self.known_volumes_ml = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500], dtype=np.float32)

        # --- Ratio-Based Volume Lookup Table ---
        # Format: [Ratio (h_px / Locked_Total_px), Volume (ml)]
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
        # -----------------------------------------------------------

        # height EMA
        self.height_ema_alpha = 0.2
        self.height_px_ema = None
        self.current_height_px_ema = None

        # 고정 bottom_y
        self.fixed_bottle_bottom_y = None
        self.bottom_y_locked = False

        # bottle EMA filter
        self.bottle_ema_alpha = 0.1
        self.bottle_area_ema = None

        # EMA filter
        self.ema_alpha = 0.2
        self.estimated_ml_ema = None
        self.last_total_cup_px = 1.0 # Initialize to 1 to avoid div by zero

        # --- Automatic Snap Trigger Logic ---
        self.trigger_pub = self.create_publisher(Empty, '/dsr01/robotender_snap/trigger', 10)
        self.volume_pub = self.create_publisher(Float32, '/dsr01/robotender/liquid_volume', 10)
        
        # New: Subscriber for dynamic target volume from Pour node
        self.target_ml_sub = self.create_subscription(Float32, '/detection/cup_target_volume', self.target_volume_callback, 10)
        
        self.target_volume_ml = 100.0  # Default incremental target (ml)
        self.start_volume_ml = 0.0     # Volume at the start of the pour
        self.target_total_volume_ml = 0.0 # start_volume + target_volume (Absolute Goal)
        self.target_line_y = None      # Y-coordinate for the target visualization line
        
        self.snap_triggered = False    # To ensure we only trigger once per pour
        self.low_volume_count = 0      # Used to reset the trigger flag when cup is empty
        
        # --- Tare Safety Logic ---
        self.tare_stability_count = 0  # To ensure we only tare on a truly empty cup
        self.no_cup_count = 0          # To reset state when cup is removed
        self.TARE_STABILITY_THRESHOLD = 120 # 4 seconds at 30fps
        self.NO_CUP_THRESHOLD = 45         # 1.5 seconds at 30fps
        
        # Store last known cup bbox for line visualization
        self.last_cup_bbox = None
        # -----------------------------------

        # 구독자 설정
        self.create_subscription(Image, '/camera/camera_2/color/image_raw', self.color_callback, 10)
        # aligned_depth_to_color/image_raw 토픽 사용 권장
        self.create_subscription(Image, '/camera/camera_2/aligned_depth_to_color/image_raw', self.depth_callback, 10)

    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def apply_height_ema(self, value: float):
        if value is None:
            return self.height_px_ema

        if self.height_px_ema is None:
            self.height_px_ema = value
        else:
            self.height_px_ema = (
                self.height_ema_alpha * value
                + (1.0 - self.height_ema_alpha) * self.height_px_ema
            )
        return self.height_px_ema

    def estimate_height_px(self, liquid_mask: np.ndarray):
        ys_l, xs_l = np.where(liquid_mask > 0)

        if len(ys_l) == 0:
            return None, None

        if self.fixed_bottle_bottom_y is None:
            return None, None

        waterline_y = int(np.min(ys_l))
        height_px = self.fixed_bottle_bottom_y - waterline_y

        if height_px < 0:
            # Waterline below bottom line (measurement error or noise)
            return 0.0, waterline_y

        return float(height_px), waterline_y

    def height_px_to_volume_ml(self, height_px: float):
        """Calculates volume using only the Empirical Ratio Lookup Table."""
        if self.locked_total_cup_px is None or self.locked_total_cup_px <= 0 or height_px <= 0:
            return 0.0
            
        # 1. Calculate the current fill ratio based on the LOCKED reference
        current_ratio = height_px / self.locked_total_cup_px
        
        # 2. Linear Interpolation from measured points
        ratios = self.ratio_to_ml_table[:, 0]
        volumes = self.ratio_to_ml_table[:, 1]
        vol_ml = np.interp(current_ratio, ratios, volumes)
        
        return float(vol_ml)

    def apply_bottle_area_ema(self, value: float):
        if value is None:
            return self.bottle_area_ema

        if self.bottle_area_ema is None:
            self.bottle_area_ema = value
        else:
            self.bottle_area_ema = (
                self.bottle_ema_alpha * value
                + (1.0 - self.bottle_ema_alpha) * self.bottle_area_ema
            )

        return self.bottle_area_ema

    def apply_liquid_ml_ema(self, value: float):
        if value is None:
            return self.estimated_ml_ema

        if self.estimated_ml_ema is None:
            self.estimated_ml_ema = value
        else:
            self.estimated_ml_ema = (
                self.ema_alpha * value
                + (1.0 - self.ema_alpha) * self.estimated_ml_ema
            )

        return self.estimated_ml_ema

    def target_volume_callback(self, msg):
        """Callback to receive dynamic target volume (ml to add) from the Pour node."""
        val = float(msg.data)
        
        if val <= 0.0:
            self.get_logger().info("Target Volume Cleared.")
            self.target_volume_ml = 0.0
            self.target_total_volume_ml = 0.0
            self.target_line_y = None
            return

        self.target_volume_ml = val
        self.snap_triggered = False  # Reset for the new pour
        
        # Snapshot current volume to determine the absolute goal
        if self.estimated_ml_ema is not None:
            self.start_volume_ml = self.estimated_ml_ema
        else:
            self.start_volume_ml = 0.0
            
        self.target_total_volume_ml = self.start_volume_ml + self.target_volume_ml
        
        # Calculate the Y coordinate for the target line
        if self.locked_total_cup_px is not None and self.fixed_bottle_bottom_y is not None:
            # 1. Reverse lookup: ml -> Ratio
            ratios = self.ratio_to_ml_table[:, 0]
            volumes = self.ratio_to_ml_table[:, 1]
            target_ratio = np.interp(self.target_total_volume_ml, volumes, ratios)
            
            # 2. Ratio -> height_px
            target_h_px = target_ratio * self.locked_total_cup_px
            
            # 3. height_px -> waterline_y (global pixels)
            self.target_line_y = int(self.fixed_bottle_bottom_y - target_h_px)
            
        self.get_logger().info(f"New Target Received: +{self.target_volume_ml}ml (Absolute Goal: {self.target_total_volume_ml:.1f}ml)")

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

        if self.color_image is None:
            return

        img_vis = self.color_image.copy()

        # ✅ YOLO-seg 추론
        results = model.predict(
            source=img_vis,
            conf=0.4,
            iou=0.5,
            retina_masks=True,
            verbose=False
        )

        overlay = img_vis.copy()

        bottle_mask_current = None
        liquid_mask_current = None
        liquid_bbox_current = None

        for result in results:
            boxes = result.boxes
            masks = result.masks

            if boxes is None or len(boxes) == 0:
                continue

            mdata = masks.data
            try:
                mdata = mdata.detach().cpu().numpy()
            except Exception:
                mdata = np.asarray(mdata)

            H, W = img_vis.shape[:2]

            for i, b in enumerate(boxes):
                conf = float(b.conf[0].cpu().numpy())
                cls_id = int(b.cls[0].cpu().numpy())
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)

                mi = mdata[i]
                if mi.shape != (H, W):
                    mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)

                mask_bin = (mi > 0.5).astype(np.uint8)
                mask_area = int(np.count_nonzero(mask_bin))
                class_name = model.names.get(cls_id, cls_id)

                if class_name == self.cup_class_name:
                    if bottle_mask_current is None or mask_area > int(np.count_nonzero(bottle_mask_current)):
                        bottle_mask_current = mask_bin.copy()
                        # Update last known cup bbox for line visualization
                        self.last_cup_bbox = (x1, y1, x2, y2)

                if class_name != self.cup_class_name:
                    if liquid_mask_current is None or mask_area > int(np.count_nonzero(liquid_mask_current)):
                        liquid_mask_current = mask_bin.copy()
                        liquid_bbox_current = (x1, y1, x2, y2)

                color = CLASS_COLORS.get(cls_id, (255,255,255))
                MASK_ALPHA = 0.2
                colored = np.zeros_like(overlay, dtype=np.uint8)
                colored[:, :] = color
                overlay = np.where(
                    mask_bin[:, :, None].astype(bool),
                    ((1 - MASK_ALPHA) * overlay + MASK_ALPHA * colored).astype(np.uint8),
                    overlay
                )

                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # -----------------------------------
        # HYBRID LOGIC: Auto-Tare & Reference Locking
        # -----------------------------------
        if bottle_mask_current is not None:
            self.no_cup_count = 0 
            if liquid_mask_current is None:
                self.tare_stability_count += 1
                if self.tare_stability_count >= self.TARE_STABILITY_THRESHOLD:
                    ys_b, xs_b = np.where(bottle_mask_current > 0)
                    if len(ys_b) > 0:
                        self.fixed_bottle_bottom_y = int(np.max(ys_b))
                        self.locked_total_cup_px = float(np.max(ys_b) - np.min(ys_b))
                        self.last_total_cup_px = self.locked_total_cup_px
                        self.bottom_y_locked = False 
            else:
                self.tare_stability_count = 0
        else:
            self.no_cup_count += 1
            self.tare_stability_count = 0
            if self.no_cup_count >= self.NO_CUP_THRESHOLD:
                self.fixed_bottle_bottom_y = None
                self.locked_total_cup_px = None
                self.bottom_y_locked = False
                self.snap_triggered = False
                self.height_px_ema = None
                self.estimated_ml_ema = None
                self.target_line_y = None # Reset visualization line
        
        # -----------------------------------
        # HYBRID LOGIC: Liquid Volume Estimation
        # -----------------------------------
        if liquid_mask_current is not None and self.fixed_bottle_bottom_y is not None:
            self.bottom_y_locked = True
            height_px, waterline_y = self.estimate_height_px(liquid_mask_current)

            if height_px is not None:
                height_px_ema = self.apply_height_ema(height_px)
                volume_ml = self.height_px_to_volume_ml(height_px_ema)
                volume_ml_ema = self.apply_liquid_ml_ema(volume_ml)
                self.current_height_px_ema = height_px_ema

                vol_msg = Float32()
                vol_msg.data = float(volume_ml_ema)
                self.volume_pub.publish(vol_msg)

                # Auto-Trigger Snap based on absolute goal
                if self.target_total_volume_ml > 0 and volume_ml_ema >= self.target_total_volume_ml and not self.snap_triggered:
                    self.trigger_pub.publish(Empty())
                    self.snap_triggered = True
                    self.get_logger().info(f"--- AUTO SNAP TRIGGERED: {volume_ml_ema:.1f}ml (Goal: {self.target_total_volume_ml:.1f}) ---")
                
                if volume_ml_ema > 5.0:
                    self.low_volume_count = 0

                if liquid_bbox_current is not None:
                    lx1, ly1, lx2, ly2 = liquid_bbox_current
                    target_text = f"GOAL: {self.target_total_volume_ml:.1f}ml"
                    if self.snap_triggered:
                        target_text += " [OK]"
                    cv2.putText(overlay, target_text, (lx1, max(30, ly1 - 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    # The following line will be replaced by a persistent display below
                    # cv2.putText(overlay, f"CUR: {volume_ml_ema:.1f}ml", (lx1, ly2 + 25),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # --- Persistent Volume Display (Always show near cup) ---
        if self.last_cup_bbox is not None:
            cx1, cy1, cx2, cy2 = self.last_cup_bbox
            current_vol = self.estimated_ml_ema if self.estimated_ml_ema is not None else 0.0
            # If liquid is actually detected, use red. Otherwise, use white for empty.
            vol_color = (0, 0, 255) if liquid_mask_current is not None else (255, 255, 255)
            
            cv2.putText(overlay, f"CUR: {current_vol:.1f}ml", (cx1, cy2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, vol_color, 2)

        # Draw Target Horizontal Line (Centered and Shorter)
        if self.target_line_y is not None and self.fixed_bottle_bottom_y is not None and self.last_cup_bbox is not None:
            H, W = overlay.shape[:2]
            if 0 < self.target_line_y < H:
                # Use the cup bbox to calculate center and width
                cx1, cy1, cx2, cy2 = self.last_cup_bbox
                cup_width = cx2 - cx1
                line_width = int(cup_width * 1.5)
                center_x = (cx1 + cx2) // 2

                x_start = max(0, center_x - (line_width // 2))
                x_end = min(W, center_x + (line_width // 2))

                # Draw a thinner line (thickness=1)
                cv2.line(overlay, (x_start, self.target_line_y), (x_end, self.target_line_y), (0, 255, 0), 1)
                cv2.putText(overlay, f"FILL TO: {self.target_total_volume_ml:.1f}ml", (x_start, self.target_line_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # --- Visual Status Overlay ---
        if self.locked_total_cup_px is not None:
            status_color = (0, 255, 0) if self.bottom_y_locked else (0, 255, 255)
            status_msg = "POURING (LOCKED)" if self.bottom_y_locked else "READY (SCALE LOCKED)"
            cv2.putText(overlay, f"STATUS: {status_msg}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        else:
            cv2.putText(overlay, "STATUS: SEARCHING/CALIBRATING...", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Reset trigger when liquid is gone
        if liquid_mask_current is None:
            self.low_volume_count += 1
            if self.low_volume_count > 45:
                self.snap_triggered = False
                self.bottom_y_locked = False

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
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
