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
        self.target_volume_ml = 180.0  # Target volume to stop pouring (ml)
        self.snap_triggered = False    # To ensure we only trigger once per pour
        self.low_volume_count = 0      # Used to reset the trigger flag when cup is empty
        
        # --- Tare Safety Logic ---
        self.tare_stability_count = 0  # To ensure we only tare on a truly empty cup
        self.TARE_STABILITY_THRESHOLD = 60 # ~2 seconds at 30fps
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

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

        if self.color_image is None:
            return

        # depth(m)
        depth_m = (self.depth_image.astype(np.float32) * self.depth_scale)

        img_vis = self.color_image.copy()

        # ✅ YOLO-seg 추론
        results = model.predict(
            source=img_vis,
            conf=0.4,  # Lowered slightly for better liquid detection
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
        if bottle_mask_current is not None and liquid_mask_current is None:
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
        
        # -----------------------------------
        # HYBRID LOGIC: Liquid Volume Estimation (Persistent)
        # -----------------------------------
        if liquid_mask_current is not None and self.fixed_bottle_bottom_y is not None:
            # Even if bottle_mask_current is None, we use the locked bottom reference
            self.bottom_y_locked = True

            height_px, waterline_y = self.estimate_height_px(liquid_mask_current)

            if height_px is not None:
                height_px_ema = self.apply_height_ema(height_px)
                volume_ml = self.height_px_to_volume_ml(height_px_ema)
                self.current_height_px_ema = height_px_ema

                vol_msg = Float32()
                vol_msg.data = float(volume_ml)
                self.volume_pub.publish(vol_msg)

                # Auto-Trigger Snap
                if volume_ml >= self.target_volume_ml and not self.snap_triggered:
                    self.trigger_pub.publish(Empty())
                    self.snap_triggered = True
                    self.get_logger().info(f"--- AUTO SNAP TRIGGERED: {volume_ml:.1f}ml ---")
                
                if volume_ml > 5.0:
                    self.low_volume_count = 0

                if liquid_bbox_current is not None:
                    lx1, ly1, lx2, ly2 = liquid_bbox_current
                    target_text = f"TARGET: {self.target_volume_ml}ml"
                    if self.snap_triggered:
                        target_text += " [OK]"
                    cv2.putText(overlay, target_text, (lx1, max(30, ly1 - 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(overlay, f"VOL: {volume_ml:.1f}ml", (lx1, ly2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # --- Visual Status Overlay ---
        if self.locked_total_cup_px is not None:
            status_color = (0, 255, 0) if self.bottom_y_locked else (0, 255, 255)
            if self.bottom_y_locked:
                status_msg = "POURING (LOCKED)"
            else:
                status_msg = "READY (SCALE LOCKED)"
            
            cv2.putText(overlay, f"STATUS: {status_msg}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(overlay, f"ZERO: {self.fixed_bottle_bottom_y}px | SCALE: {self.locked_total_cup_px:.1f}px", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            if bottle_mask_current is not None:
                status_msg = f"CALIBRATING... ({self.tare_stability_count/30:.1f}s)"
                cv2.putText(overlay, f"STATUS: {status_msg}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(overlay, "STATUS: SEARCHING FOR CUP...", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Reset trigger when liquid is gone
        if liquid_mask_current is None:
            self.low_volume_count += 1
            if self.low_volume_count > 45: # ~1.5 seconds
                if self.snap_triggered:
                    self.get_logger().info("Ready for next pour (Trigger reset).")
                self.snap_triggered = False
                self.bottom_y_locked = False # Allow re-tare if cup is empty

        cv2.imshow("RGB with Depth (YOLO-Seg)", overlay)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if self.current_height_px_ema is not None and self.locked_total_cup_px is not None:
                ratio = self.current_height_px_ema / self.locked_total_cup_px
                print("\n" + "="*40)
                print(f"NEW CALIBRATION POINT (Locked Scale: {self.locked_total_cup_px:.1f}px)")
                print(f"Ratio: {ratio:.4f}")
                print(f"Entry: [{ratio:.4f}, <ENTER_ML_HERE>]")
                print("="*40 + "\n")
            else:
                print("[CALIB] Error: height_px or locked_total_cup_px not available. Ensure cup is calibrated first.")
        
        if key == ord('t'):
            self.fixed_bottle_bottom_y = None
            self.bottom_y_locked = False
            self.height_px_ema = None
            print("[TARE] Manual Reset Triggered. Place empty cup to re-calibrate.")


def main(args=None):
    rclpy.init(args=args)
    node = DepthReader()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
