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

        # 구독자 설정
        self.create_subscription(Image, '/camera/camera_2/color/image_raw', self.color_callback, 10)
        # aligned_depth_to_color/image_raw 토픽 사용 권장
        self.create_subscription(Image, '/camera/camera_2/aligned_depth_to_color/image_raw', self.depth_callback, 10)

    def color_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

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

        for result in results:
            boxes = result.boxes
            masks = result.masks  # ✅ segmentation mask

            if boxes is None or len(boxes) == 0:
                continue

            # # masks가 없으면(모델/설정 문제) bbox만이라도 그리기
            # if masks is None:
            #     for b in boxes:
            #         x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            #         conf = float(b.conf[0].cpu().numpy())
            #         cls_id = int(b.cls[0].cpu().numpy())

            #         # bbox depth (fallback)
            #         roi = depth_m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            #         roi = roi[np.isfinite(roi) & (roi > 0)]
            #         obj_depth = float(np.median(roi)) if roi.size else float('nan')

            #         label = f"{model.names.get(cls_id, cls_id)} {conf:.2f} / {obj_depth:.2f}m"
            #         cv2.rectangle(overlay, (x1, y1), (x2, y2), (252, 119, 30), 2)
            #         cv2.putText(overlay, label, (x1, max(0, y1 - 10)),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (252, 119, 30), 2)
            #     continue

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
                print(f"{model.names.get(cls_id, cls_id)}: (conf={conf:.2f})")

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
