# 2번 카메라 (시리얼 넘버 = 313522301601)
# 따라야 할 컵이 있는지 확인?

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import cv2
import numpy as np
from ultralytics import YOLO


# Load drink volume YOLO model
model = YOLO("/home/been/camera_ws/training/runs/segment/train2/weights/best.pt")  # TODO: 경로 수정

# 클래스별 색 (BGR)
CLASS_COLORS = {
    0: (0, 0, 255),   # 빨강
    1: (255, 0, 0),   # 파랑
    2: (0, 255, 0)    # 초록
}


class DepthReader(Node):
    def __init__(self):
        super().__init__('depth_reader')
        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None

        self.depth_scale = 0.001  # 16UC1(mm) -> m 스케일 (환경에 맞게)

        self.bottle_class_name = "bottle"

        # 소주병처럼 비원통형 bottle이면 lookup table 권장
        # 400ml 이상은 segmentation 모델이 측정 불가 (학습을 하지 않음)
        self.known_heights_px = np.array([0, 71, 105, 140, 180, 206, 253, 290, 338, 400, 450], dtype=np.float32)
        self.known_volumes_ml = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500], dtype=np.float32)

        # height EMA
        self.height_ema_alpha = 0.2
        self.height_px_ema = None
        self.current_height_px_ema = None

        # bottom_y
        self.fixed_bottle_bottom_y = None
        self.bottom_y_locked = False

        # 구독자 설정
        self.create_subscription(Image, '/camera/camera_2/color/image_raw', self.color_callback, 10)
        # aligned_depth_to_color/image_raw 토픽 사용 권장
        self.create_subscription(Image, '/camera/camera_2/aligned_depth_to_color/image_raw', self.depth_callback, 10)

    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

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
    
    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

        if self.color_image is None:
            return

        # depth(m)
        depth_m = (self.depth_image.astype(np.float32) * self.depth_scale)

        # 원본 코드처럼 flip 유지(필요 없으면 제거)
        depth_m = cv2.flip(depth_m, -1)
        img_vis = cv2.flip(self.color_image.copy(), -1)

        # ✅ YOLO-seg 추론
        # retina_masks=True 권장: 마스크를 원본 해상도에 더 잘 맞춰줌
        results = model.predict(
            source=img_vis,
            conf=0.25,  # default = 0.5
            iou=0.5,
            retina_masks=True,
            verbose=False
        )

        # 시각화용 오버레이
        overlay = img_vis.copy()

        # 현재 frame에서 사용할 대표 mask
        bottle_mask_current = None
        liquid_mask_current = None

        # 현재 frame에서 사용할 대표 bbox
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

                # bottle / liquid mask
                if class_name == self.bottle_class_name:
                    if bottle_mask_current is None or mask_area > int(np.count_nonzero(bottle_mask_current)):
                        bottle_mask_current = mask_bin.copy()
                        bottle_bbox_current = (x1, y1, x2, y2)

                if class_name != self.bottle_class_name:
                    if liquid_mask_current is None or mask_area > int(np.count_nonzero(liquid_mask_current)):
                        liquid_mask_current = mask_bin.copy()
                        liquid_bbox_current = (x1, y1, x2, y2)

                # ✅ 마스크 내부 depth median 계산 (0이나 NaN 제거)
                dm = depth_m[mask_bin.astype(bool)]
                dm = dm[np.isfinite(dm) & (dm > 0)]
                obj_depth = float(np.median(dm)) if dm.size else float('nan')

                # 색상은 클래스별로 다르게 하고 싶다면 여기서 바꾸세요
                # color = (252, 119, 30)
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
                # label = f"{model.names.get(cls_id, cls_id)} {conf:.2f} / {obj_depth:.2f}m"
                label = f"{model.names.get(cls_id, cls_id)} {conf:.2f}"
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # 로그
                # print(f"{model.names.get(cls_id, cls_id)}: {obj_depth:.2f}m (conf={conf:.2f})")
                # print(f"{model.names.get(cls_id, cls_id)}: (conf={conf:.2f})")

        # -----------------------------------
        # bottle만 보이고 liquid가 없을 때, bottom_y 1회 저장
        # -----------------------------------
        if bottle_mask_current is not None and liquid_mask_current is not None and self.fixed_bottle_bottom_y is None:
            ys_b, xs_b = np.where(bottle_mask_current > 0)
            if len(ys_b) > 0:
                self.fixed_bottle_bottom_y = int(np.max(ys_b))
                self.bottom_y_locked = True

        # -----------------------------------
        # bottle + liquid 기반 수면선 높이 계산
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

                # liquid volume
                if liquid_bbox_current is not None:
                    lx1, ly1, lx2, ly2 = liquid_bbox_current
                    text_vol = f"volume={volume_ml:.1f}ml"

                    cv2.putText(overlay, text_vol, (lx2 - 10, max(0, ly1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.namedWindow("RGB with Depth (YOLO-Seg)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RGB with Depth (YOLO-Seg)", 900, 520)
        cv2.imshow("RGB with Depth (YOLO-Seg)", overlay)
        cv2.waitKey(1)


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
