# 1번 카메라 (시리얼 넘버 = 311322302867)
# 주류 정보를 받아 해당 주류를 reach -> grip
# 주류 정보 = json
"""
json 예시
{
    "주문": "소맥",
    "레시피": {
        "소주": 20,
        "맥주": 100
    }
}
"""

import rclpy
from rclpy.node import Node
from pathlib import Path

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

import cv2
import numpy as np
from ultralytics import YOLO


# Load drink YOLO model
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "cam_1.pt"
model = YOLO(str(WEIGHTS_PATH))


class DepthReader(Node):
  def __init__(self):
    super().__init__('depth_reader')
    self.bridge = CvBridge()
    self.depth_image = None
    self.color_image = None

    self.depth_scale = 0.001

    # Set subscriber
    # camera_1 = drink cam / camera_2 = cup cam
    self.create_subscription(Image, '/camera/camera_1/color/image_raw', self.color_callback, 10)
    
    # aligned_depth_to_color/image_raw 토픽을 사용해야 제대로 된 depth image를 사용할 수 있음
    self.create_subscription(Image, '/camera/camera_1/aligned_depth_to_color/image_raw', self.depth_callback, 10)

  def color_callback(self, msg):
    self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

  def depth_callback(self, msg):
    self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")  # 깊이값은 uint16

    if self.color_image is not None:

      depth_image = self.depth_image * self.depth_scale
      depth_image = cv2.flip(depth_image, -1)

      # 시각화 (깊이 텍스트 출력)
      img_vis = self.color_image.copy()
      
      # Detect objects using YOLO
      img_vis = cv2.flip(img_vis, -1)
      results = model(img_vis)

      # Process the results
      for result in results:
        boxes = result.boxes
        for box in boxes:
          x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
          confidence = box.conf[0].cpu().numpy()
          class_id = box.cls[0].cpu().numpy()

          if confidence < 0.5:
            continue  # Skip detections with low confidence

          # Calculate the distance to the object
          object_depth = np.median(depth_image[y1:y2, x1:x2])
          label = f"{model.names[int(class_id)]}/{object_depth:.2f}m"

          # Draw a rectangle around the object
          cv2.rectangle(img_vis, (x1, y1), (x2, y2), (252, 119, 30), 2)

          # Draw the bounding box
          cv2.putText(img_vis, label, (x1, y1-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (252, 119, 30), 2)
          
          print(f"{model.names[int(class_id)]}: {object_depth:.2f}m")
      
      cv2.namedWindow("RGB with Depth", cv2.WINDOW_NORMAL)
      cv2.resizeWindow("RGB with Depth", 850, 500)

      cv2.imshow("RGB with Depth", img_vis)
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
