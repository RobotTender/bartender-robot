import cv2
import time
import json
import numpy as np
from ultralytics import YOLO
from pathlib import Path

import rclpy
import pyrealsense2 as rs
from rclpy.node import Node

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from dsr_msgs2.srv import SetRobotMode, MoveJoint, MoveLine, GetCurrentPosx, GetRobotState

import DR_init
from .defines import CHEERS_POSE


ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 30, 30
ORDER_TOPIC = "/bartender/order_detail"
ACTION_TOPIC = "/bartender/action_request"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

object_dict = {"0": 'juice', "1": 'beer', "2": 'soju'}
object_dict_reverse = {'juice': "0", 'beer': "1", 'soju': "2"}

# 클래스별 색 (BGR)
CLASS_COLORS = {
    0: (0, 0, 255),   # 빨강
    1: (255, 0, 0),   # 파랑
    2: (0, 255, 0)    # 초록
}

# Load YOLO model
MODEL_PATH = Path(__file__).resolve().parents[4] / "detection" / "weights" / "cam_1.pt"
model = YOLO(str(MODEL_PATH))


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robot_controller_node")

        self.bridge = CvBridge()

        self.get_logger().info("ROS2 구독자 설정 시작")

        self.intrinsics = None
        self.latest_cv_color = None
        self.latest_cv_depth = None
        self.latest_cv_depth_mm = None
        self.latest_cv_vis = None

        self.state = 'IDLE'  # IDLE, RUNNING, STOPPED
        self.task_received = False
        self.items = None

        self.depth_scale = 0.001

        self.R = np.array([
            [-0.788489317968,  -0.614148198918, -0.0332653756482],
            [-0.0868309265704,  0.0576098432706,  0.994555929121],
            [-0.608888319515,   0.787085189623,  -0.0987518031922],
        ], dtype=np.float64)

        self.t = np.array([521.115058698, 170.946228749, 834.749571453], dtype=np.float64)

        self.color_sub = message_filters.Subscriber(
            self, Image, '/camera/camera_1/color/image_raw'
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera_1/aligned_depth_to_color/image_raw'
        )
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, '/camera/camera_1/aligned_depth_to_color/camera_info'
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)
        self.order_sub = self.create_subscription(String, ORDER_TOPIC, self.order_callback, 10)
        # self.action_pub = self.create_publisher(String, ACTION_TOPIC, 10) # Deprecated topic

        # New Service Clients
        self.gripper_open_cli = self.create_client(Trigger, f'/{ROBOT_ID}/gripper/open')
        self.gripper_close_cli = self.create_client(Trigger, f'/{ROBOT_ID}/gripper/close')
        self.pour_start_cli = self.create_client(Trigger, f'/{ROBOT_ID}/pour/start')

        self.get_logger().info("컬러/뎁스/카메라정보 토픽 구독 대기 중...")
        self.get_logger().info(f"주문 토픽 구독 대기 중... {ORDER_TOPIC}")
        self.get_logger().info("후속 액션 서비스 준비... /dsr01/pour/start")
        self.get_logger().info("화면이 나오지 않으면 Launch 명령어를 확인하세요.")

        self.get_logger().info("RealSense ROS2 구독자와 로봇 컨트롤러가 초기화되었습니다.")

    def order_callback(self, msg: String):
        try:
            items = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"주문 JSON 파싱 실패: {exc}")
            return

        if not isinstance(items, dict) or "recipe" not in items:
            self.get_logger().error(f"주문 형식이 올바르지 않습니다: {msg.data}")
            return

        self.items = items
        self.task_received = True
        self.get_logger().info(f"주문 수신: {self.items}")
        self._try_start_task()

    def _try_start_task(self):
        if self.state == "RUNNING":
            self.get_logger().info("State is already RUNNING, skipping.")
            return
        if not self.task_received or self.items is None:
            self.get_logger().info("No task received or items is None.")
            return

        if self.latest_cv_color is None:
            self.get_logger().info("latest_cv_color is None, waiting for camera...")
        if self.latest_cv_depth_mm is None:
            self.get_logger().info("latest_cv_depth_mm is None, waiting for camera...")
        if self.intrinsics is None:
            self.get_logger().info("intrinsics is None, waiting for camera...")

        if self.latest_cv_color is None or self.latest_cv_depth_mm is None or self.intrinsics is None:
            self.get_logger().info("카메라 입력 준비 전이라 주문을 보류합니다.")
            return

        import threading
        self.get_logger().info("Starting process_grip thread...")
        self.state = "RUNNING"
        threading.Thread(target=self.process_grip, args=(self.items,), daemon=True).start()
    def camera_to_robot(self, point_cam):
        return point_cam @ self.R + self.t

    def synced_callback(self, color_msg, depth_msg, info_msg):
        try:
            self.latest_cv_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.latest_cv_depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge 변환 오류: {e}")
            return

        if self.intrinsics is None:
            self.intrinsics = rs.intrinsics()
            self.intrinsics.width = info_msg.width
            self.intrinsics.height = info_msg.height
            self.intrinsics.ppx = info_msg.k[2]
            self.intrinsics.ppy = info_msg.k[5]
            self.intrinsics.fx = info_msg.k[0]
            self.intrinsics.fy = info_msg.k[4]
            if info_msg.distortion_model == 'plumb_bob' or info_msg.distortion_model == 'rational_polynomial':
                self.intrinsics.model = rs.distortion.brown_conrady
            else:
                self.intrinsics.model = rs.distortion.none
            self.intrinsics.coeffs = list(info_msg.d)
            self.get_logger().info("카메라 내장 파라미터(Intrinsics) 수신 완료.")


    def _gripper_open(self):
        self.get_logger().info("Service Call: Gripper Open")
        if not self.gripper_open_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Gripper open service not available!")
            return False
        future = self.gripper_open_cli.call_async(Trigger.Request())
        while not future.done(): time.sleep(0.01)
        return future.result().success

    def _gripper_close(self, force=800):
        self.get_logger().info(f"Service Call: Gripper Close (Force fixed to 800 in ActionNode)")
        if not self.gripper_close_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Gripper close service not available!")
            return False
        future = self.gripper_close_cli.call_async(Trigger.Request())
        while not future.done(): time.sleep(0.01)
        return future.result().success

    def _pour_start(self):
        self.get_logger().info("Service Call: Pouring Start")
        if not self.pour_start_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Pour service not available!")
            return False
        future = self.pour_start_cli.call_async(Trigger.Request())
        while not future.done(): time.sleep(0.01)
        return future.result().success

    def _movej(self, pos, vel=30.0, acc=30.0, mode=0, radius=0.0, blend_type=0, sync_type=0):
        from dsr_msgs2.srv import MoveJoint
        cli = self.create_client(MoveJoint, '/dsr01/motion/move_joint')
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for MoveJoint service...")
        req = MoveJoint.Request()
        req.pos = [float(x) for x in pos]; req.vel = float(vel); req.acc = float(acc)
        req.mode = int(mode); req.radius = float(radius); req.blend_type = int(blend_type); req.sync_type = int(sync_type)
        future = cli.call_async(req)
        while not future.done(): time.sleep(0.01)
        return future.result()

    def _movel(self, pos, vel=[30.0, 30.0], acc=[30.0, 30.0], mode=0, radius=0.0, blend_type=0, sync_type=0):
        from dsr_msgs2.srv import MoveLine
        cli = self.create_client(MoveLine, '/dsr01/motion/move_line')
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for MoveLine service...")
        req = MoveLine.Request()
        req.pos = [float(x) for x in pos]; req.vel = [float(x) for x in vel]; req.acc = [float(x) for x in acc]
        req.mode = int(mode); req.radius = float(radius); req.blend_type = int(blend_type); req.sync_type = int(sync_type)
        future = cli.call_async(req)
        while not future.done(): time.sleep(0.01)
        return future.result()

    def process_grip(self, items):
        self.get_logger().info(f"Received item command: {self.items}")
        class_id = object_dict_reverse[[x for x in items["recipe"].keys()][0]]
        volume = [y for y in items["recipe"].values()][0]

        try:
            self.get_logger().info("Starting grip process (Robot should be in AUTONOMOUS mode)...")
            time.sleep(0.1)

            if self.latest_cv_depth_mm is None or self.intrinsics is None:
                self.get_logger().warn("아직 뎁스 프레임 또는 카메라 정보가 수신되지 않았습니다.")
                return

            self.get_logger().info("초기 자세")
            home_posj = [69.5, -43.0, 102.0, 101.0, -72.0, -213.0]
            self._movej(home_posj, vel=VELOCITY, acc=ACC)
            self._gripper_open() # Replaced self.gripper.move(0)
            time.sleep(2)

            img_raw = self.latest_cv_color.copy()
            depth_raw = self.latest_cv_depth_mm.copy()
            img = cv2.flip(img_raw, -1)
            self.latest_cv_vis = cv2.flip(img_raw.copy(), -1)
            h, w = img.shape[:2]
            object_loc_dict = {'0': [], '1': [], '2': []}

            results = model(img)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype('int')
                    confidence = box.conf[0].cpu().numpy()
                    cls_id = str(int(box.cls[0].cpu().numpy()))
                    if confidence < 0.5: continue
                    u_flip, v_flip = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    u_raw, v_raw = w - 1 - u_flip, h - 1 - v_flip
                    roi_half = 5
                    x_min, x_max = max(0, u_raw - roi_half), min(w, u_raw + roi_half + 1)
                    y_min, y_max = max(0, v_raw - roi_half), min(h, v_raw + roi_half + 1)
                    depth_roi = depth_raw[y_min:y_max, x_min:x_max]
                    valid_depth = depth_roi[depth_roi > 0]
                    if valid_depth.size == 0: continue
                    object_depth = float(np.median(valid_depth))
                    label = f"{object_dict[cls_id]} / {object_depth*0.001:.2f}m"
                    color = CLASS_COLORS.get(int(cls_id), (255, 255, 255))
                    object_loc_dict[cls_id] = {"u_flip": u_flip, "v_flip": v_flip, "u_raw": u_raw, "v_raw": v_raw, "depth_mm": object_depth}
                    cv2.rectangle(self.latest_cv_vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(self.latest_cv_vis, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            target = object_loc_dict[class_id]
            depth_mm = target["depth_mm"]
            if depth_mm == 0:
                print(f"해당 {model.names[class_id]}의 깊이를 측정할 수 없습니다 (값: 0)")
                return

            depth_m = float(depth_mm) / 1000.0
            u_raw, v_raw = target["u_raw"], target["v_raw"]
            u_vis, v_vis = target["u_flip"], target["v_flip"]
            cx, cy = self.intrinsics.ppx, self.intrinsics.ppy
            fx, fy = self.intrinsics.fx, self.intrinsics.fy
            x_cam, y_cam, z_cam = (u_raw - cx) * depth_m / fx, (v_raw - cy) * depth_m / fy, depth_m
            p_cam = np.array([x_cam * 1000, y_cam * 1000, z_cam * 1000], dtype=np.float64)
            p_robot = self.camera_to_robot(p_cam)

            print("--- 변환된 최종 3D 좌표 ---")
            print(f"flip 픽셀 좌표: (u_vis={u_vis}, v_vis={v_vis})")
            print(f"raw 픽셀 좌표 : (u_raw={u_raw}, v_raw={v_raw}), Depth: {depth_m*1000:.1f} mm")
            self.get_logger().info(f"로봇 목표 좌표: X={p_robot[0]:.1f}, Y={p_robot[1]:.1f}, Z={p_robot[2]:.1f}\n")

            self.move_robot_and_control_gripper(p_robot[0], p_robot[1], p_robot[2])
            
            # Use service instead of topic
            self.get_logger().info("--- 액션 요청 (Pour Service Call) ---")
            success = self._pour_start()
            if success:
                self.get_logger().info("따르기 액션 성공적으로 완료")
            else:
                self.get_logger().error("따르기 액션 실패")
            print("=" * 50)

        except Exception as e:
            import traceback
            self.get_logger().error(f"Detection Error!!\n{traceback.format_exc()}")
            return
        finally:
            self.task_received = False
            self.items = None
            self.state = "IDLE"

    def move_robot_and_control_gripper(self, x, y, z):
        import time
        try:
            self.get_logger().info("잡기 전 자세")
            P_ready = [28.0, -35.0, 100.0, 77.0, 63.0, -154.0]
            self._movej(P_ready, VELOCITY, ACC)

            from dsr_msgs2.srv import GetCurrentPose
            pose_cli = self.create_client(GetCurrentPose, '/dsr01/system/get_current_pose')
            while not pose_cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Waiting for GetCurrentPose service...")
            pose_req = GetCurrentPose.Request(space_type=1)
            pose_future = pose_cli.call_async(pose_req)
            while not pose_future.done(): time.sleep(0.01)
            current_pos = list(pose_future.result().pos)

            target_pos_list_1 = [x - 20, y - 50, z, current_pos[3], current_pos[4], current_pos[5]]
            target_pos_list_2 = [x - 20, y + 50, z - 20, current_pos[3], current_pos[4], current_pos[5]]
            
            self.get_logger().info("목표 지점으로 접근합니다")
            self._movel(target_pos_list_1, vel=[40.0, 40.0], acc=[ACC, ACC])
            self.get_logger().info(f"목표 지점으로 이동합니다: {target_pos_list_2}")
            self._movel(target_pos_list_2, vel=[40.0, 40.0], acc=[ACC, ACC])

            time.sleep(0.5)
            # gripper 주류 잡기 - Replaced self.gripper.move(550) with service call
            self._gripper_close(800) 
            self.get_logger().info("그리퍼가 닫힐 때까지 대기합니다 (3초)...")
            time.sleep(3.0)

            P_mid = [61.0, -20.0, 97.0, 96.0, -63.0, -195.0]
            self._movej(P_mid, VELOCITY, ACC)
            time.sleep(1.0)

            self.get_logger().info("따르기 위한 자세를 취합니다.")
            last_posj = [45.0, 0.0, 135.0, 90.0, -90.0, -135.0]
            self._movej(last_posj, VELOCITY, ACC)

        except Exception as e:
            self.get_logger().error(f"로봇 이동 및 그리퍼 제어 중 오류 발생: {e}")


def main(args=None):
    rclpy.init(args=args)
    robot_controller = None
    try:
        robot_controller = RobotControllerNode()
        rclpy.spin(robot_controller)
    except Exception as e:
        print(f"노드 실행 중 오류: {e}")
    finally:
        print("프로그램을 종료합니다")
        if robot_controller is not None:
            robot_controller.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok(): rclpy.shutdown()
        print("종료 완료")

if __name__ == "__main__":
    main()
