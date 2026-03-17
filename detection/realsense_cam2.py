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

        # 소주병처럼 비원통형 bottle이면 lookup table 권장
        self.known_heights_px = np.array([0, 26.3, 50, 67.4, 84, 100, 115.7, 131.9, 140.1, 158, 174], dtype=np.float32)
        self.known_volumes_ml = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500], dtype=np.float32)

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

        # --- Automatic Snap Trigger Logic ---
        self.trigger_pub = self.create_publisher(Empty, '/dsr01/robotender_snap/trigger', 10)
        self.volume_pub = self.create_publisher(Float32, '/dsr01/robotender/liquid_volume', 10)
        self.target_volume_ml = 180.0  # Target volume to stop pouring (ml)
        self.snap_triggered = False    # To ensure we only trigger once per pour
        self.low_volume_count = 0      # Used to reset the trigger flag when cup is empty
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

    def estimate_height_px(self, bottle_mask: np.ndarray, liquid_mask: np.ndarray):
        ys_l, xs_l = np.where(liquid_mask > 0)

        if len(ys_l) == 0:
            return None, None, None

        if self.fixed_bottle_bottom_y is None:
            return None, None, None

        waterline_y = int(np.min(ys_l))
        height_px = self.fixed_bottle_bottom_y - waterline_y

        if height_px < 0:
            return None, self.fixed_bottle_bottom_y, waterline_y

        return height_px, self.fixed_bottle_bottom_y, waterline_y

    def height_px_to_volume_ml(self, height_px: float):
        return float(np.interp(height_px, self.known_heights_px, self.known_volumes_ml))

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
        # retina_masks=True 권장: 마스크를 원본 해상도에 더 잘 맞춰줌
        results = model.predict(
            source=img_vis,
            conf=0.5,  # default = 0.5
            iou=0.5,
            retina_masks=True,
            verbose=False
        )

        # 시각화용 오버레이
        overlay = img_vis.copy()

        # 현재 프레임에서 사용할 대표 mask 저장
        bottle_mask_current = None
        liquid_mask_current = None

        # 표시용 bbox도 같이 저장해두면 나중에 라벨 위치 잡기 좋음
        bottle_bbox_current = None
        liquid_bbox_current = None

        for result in results:
            boxes = result.boxes
            masks = result.masks  # ✅ segmentation mask

            if boxes is None or len(boxes) == 0:
                continue

            # ✅ masks.data: (N, Hm, Wm) 텐서(0~1)
            mdata = masks.data
            try:
                mdata = mdata.detach().cpu().numpy()
            except Exception:
                mdata = np.asarray(mdata)

            H, W = img_vis.shape[:2]

            # boxes와 masks의 개수는 보통 동일(N)
            for i, b in enumerate(boxes):
                conf = float(b.conf[0].cpu().numpy())
                cls_id = int(b.cls[0].cpu().numpy())

                # bbox (표시용)
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)

                # 마스크를 원본 해상도로 맞춤
                mi = mdata[i]
                if mi.shape != (H, W):
                    mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)

                mask_bin = (mi > 0.5).astype(np.uint8)  # 0/1

                # ✅ mask 넓이(pixel 수)
                mask_area = int(np.count_nonzero(mask_bin))

                class_name = model.names.get(cls_id, cls_id)

                # bottle / soju 대표 mask 저장
                # 여러 개 잡힐 수 있으므로 일단 "가장 큰 mask"를 대표로 사용
                if class_name == self.cup_class_name:
                    if bottle_mask_current is None or mask_area > int(np.count_nonzero(bottle_mask_current)):
                        bottle_mask_current = mask_bin.copy()
                        bottle_bbox_current = (x1, y1, x2, y2)

                if class_name != self.cup_class_name:
                    if liquid_mask_current is None or mask_area > int(np.count_nonzero(liquid_mask_current)):
                        liquid_mask_current = mask_bin.copy()
                        liquid_bbox_current = (x1, y1, x2, y2)

                # ✅ 마스크 내부 depth median 계산 (0이나 NaN 제거)
                dm = depth_m[mask_bin.astype(bool)]
                dm = dm[np.isfinite(dm) & (dm > 0)]
                obj_depth = float(np.median(dm)) if dm.size else float('nan')

                # 색상은 클래스별로 다르게 하고 싶다면 여기서 바꾸세요
                color = CLASS_COLORS.get(cls_id, (255,255,255))

                # 마스크 overlay
                # (반투명)
                MASK_ALPHA = 0.2
                colored = np.zeros_like(overlay, dtype=np.uint8)
                colored[:, :] = color
                overlay = np.where(
                    mask_bin[:, :, None].astype(bool),
                    ((1 - MASK_ALPHA) * overlay + MASK_ALPHA * colored).astype(np.uint8),
                    overlay
                )

                # bbox + 라벨
                label = f"{model.names.get(cls_id, cls_id)} {conf:.2f}"
                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # 로그
                # if class_name != self.cup_class_name and estimated_ml is not None and estimated_ml_ema is not None:
                #     print(
                #         f"{class_name}: area={mask_area}px, "
                #         f"raw={estimated_ml:.1f}ml, ema={estimated_ml_ema:.1f}ml, conf={conf:.2f}"
                #     )
                # elif class_name == self.cup_class_name and self.bottle_area_ema is not None:
                #     print(
                #         f"{class_name}: area={mask_area}px, "
                #         f"bottle_ema={self.bottle_area_ema:.1f}px, conf={conf:.2f}"
                #     )
                # else:
                #     print(f"{class_name}: area={mask_area}px, conf={conf:.2f}")
        
        # -----------------------------------
        # bottle만 보이고 liquid가 없을 때, bottom_y 1회 저장
        # -----------------------------------
        if bottle_mask_current is not None and liquid_mask_current is None and self.fixed_bottle_bottom_y is None:
            ys_b, xs_b = np.where(bottle_mask_current > 0)
            if len(ys_b) > 0:
                self.fixed_bottle_bottom_y = int(np.max(ys_b))
                self.bottom_y_locked = True
                print(f"[CALIB] fixed bottom_y saved = {self.fixed_bottle_bottom_y}")

        # -----------------------------------
        # bottle + soju 기반 수면선 높이 계산
        # -----------------------------------
        if bottle_mask_current is not None and liquid_mask_current is not None:
            height_px, bottle_bottom_y, waterline_y = self.estimate_height_px(
                bottle_mask_current,
                liquid_mask_current
            )

            if height_px is not None:
                height_px_ema = self.apply_height_ema(height_px)
                volume_ml = self.height_px_to_volume_ml(height_px_ema)

                self.current_height_px_ema = height_px_ema

                # ✅ Publish Volume for Monitoring
                vol_msg = Float32()
                vol_msg.data = float(volume_ml)
                self.volume_pub.publish(vol_msg)

                # ✅ Auto-Trigger Snap Logic
                if volume_ml >= self.target_volume_ml and not self.snap_triggered:
                    self.trigger_pub.publish(Empty())
                    self.snap_triggered = True
                    self.get_logger().info(f"--- AUTO SNAP TRIGGERED: {volume_ml:.1f}ml ---")
                
                # Reset low volume counter if liquid is detected
                if volume_ml > 5.0:
                    self.low_volume_count = 0

                # # 수면선과 bottle 바닥선 시각화
                # h, w = overlay.shape[:2]
                # cv2.line(overlay, (0, waterline_y), (w - 1, waterline_y), (0, 255, 255), 2)
                # cv2.line(overlay, (0, bottle_bottom_y), (w - 1, bottle_bottom_y), (255, 255, 0), 2)

                # # ✅ 캘리브레이션용: 좌상단 고정 표시
                # cv2.putText(
                #     overlay,
                #     f"height_px(raw) = {height_px:.1f}",
                #     (30, 40),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1.0,
                #     (0, 255, 255),
                #     2
                # )

                # cv2.putText(
                #     overlay,
                #     f"height_px(ema) = {height_px_ema:.1f}",
                #     (30, 80),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1.0,
                #     (0, 255, 255),
                #     2
                # )

                # cv2.putText(
                #     overlay,
                #     f"bottom_y = {bottle_bottom_y}",
                #     (30, 120),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.8,
                #     (255, 255, 0),
                #     2
                # )

                # cv2.putText(
                #     overlay,
                #     f"waterline_y = {waterline_y}",
                #     (30, 155),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.8,
                #     (0, 255, 255),
                #     2
                # )

                # if self.fixed_bottle_bottom_y is not None:
                #     cv2.putText(
                #         overlay,
                #         f"fixed_bottom_y = {self.fixed_bottle_bottom_y}",
                #         (30, 190),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         0.8,
                #         (255, 255, 0),
                #         2
                #     )
                # else:
                #     cv2.putText(
                #         overlay,
                #         "fixed_bottom_y = not set",
                #         (30, 190),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         0.8,
                #         (0, 0, 255),
                #         2
                #     )

                # liquid bbox 근처에 결과 표시
                if liquid_bbox_current is not None:
                    lx1, ly1, lx2, ly2 = liquid_bbox_current
                    text1 = f"height={height_px:.1f}px / ema={height_px_ema:.1f}px"
                    text2 = f"volume={volume_ml:.1f}ml"
                    
                    # Target status display
                    target_text = f"TARGET: {self.target_volume_ml}ml"
                    if self.snap_triggered:
                        target_text += " [OK]"
                    cv2.putText(overlay, target_text, (lx1, max(30, ly1 - 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # cv2.putText(overlay, text1, (lx1, min(h - 40, ly2 + 20)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(overlay, text2, (lx2 - 10, max(0, ly2 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # print(
                #     f"[HEIGHT] raw={height_px:.1f}px, "
                #     f"ema={height_px_ema:.1f}px, "
                #     f"bottom_y={bottle_bottom_y}, "
                #     f"waterline_y={waterline_y}, "
                #     f"volume={volume_ml:.1f}ml"
                # )
        
        # Reset Logic: If no liquid is detected or volume is very low, count frames to reset trigger for next pour
        if liquid_mask_current is None:
            self.low_volume_count += 1
            if self.low_volume_count > 30: # Wait ~1 second at 30fps
                if self.snap_triggered:
                    self.get_logger().info("Ready for next pour (Trigger reset).")
                self.snap_triggered = False

        cv2.namedWindow("RGB with Depth (YOLO-Seg)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RGB with Depth (YOLO-Seg)", 900, 520)
        cv2.imshow("RGB with Depth (YOLO-Seg)", overlay)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if self.current_height_px_ema is not None:
                print(f"[CALIB] height_px = {self.current_height_px_ema:.2f}")
            else:
                print("[CALIB] height_px not available")


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
