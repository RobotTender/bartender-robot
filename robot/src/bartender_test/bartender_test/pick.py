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

import DR_init
# TODO: import 경로 문제
from .gripper_controller import GripperController
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

        # row-vector form transform matrix
        self.R = np.array([
            [-0.87922318205,  -0.472272400698,  0.0626528186579],
            [ 0.00958792731834, 0.113942620585, 0.993441065622],
            [-0.476313623358,  0.874057125566, -0.0956528799897]
        ], dtype=np.float64)

        self.t = np.array([520.763531569, 171.101863274, 838.568720692], dtype=np.float64)

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
        self.action_pub = self.create_publisher(String, ACTION_TOPIC, 10)

        self.get_logger().info("컬러/뎁스/카메라정보 토픽 구독 대기 중...")
        self.get_logger().info(f"주문 토픽 구독 대기 중... {ORDER_TOPIC}")
        self.get_logger().info(f"후속 액션 토픽 발행 준비... {ACTION_TOPIC}")
        self.get_logger().info("화면이 나오지 않으면 Launch 명령어를 확인하세요.")

        self.get_logger().info("RealSense ROS2 구독자와 로봇 컨트롤러가 초기화되었습니다.")

        # Initialize robot mode and gripper in a background thread to avoid deadlock during spin
        import threading
        threading.Thread(target=self._init_robot, daemon=True).start()

    def _init_robot(self):
        from dsr_msgs2.srv import SetRobotMode
        import time
        try:
            # Wait for services to be ready
            time.sleep(2.0)
            
            self.get_logger().info("Setting robot to AUTONOMOUS mode...")
            # Use direct service call to avoid deadlock with DSR_ROBOT2's internal spin
            cli = self.create_client(SetRobotMode, '/dsr01/system/set_robot_mode')
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Waiting for /dsr01/system/set_robot_mode service...")
            
            req = SetRobotMode.Request()
            req.robot_mode = 1 # ROBOT_MODE_AUTONOMOUS
            future = cli.call_async(req)
            
            # Wait for result without spin_until_future_complete
            while not future.done():
                time.sleep(0.1)
            result = future.result()
            self.get_logger().info(f"Set Robot Mode Result: {result.success if result else 'Failed'}")
            time.sleep(1.0)

            # Use direct MoveJoint service call to 'wake up' the task manager before DRL execution
            from dsr_msgs2.srv import MoveJoint
            movej_cli = self.create_client(MoveJoint, '/dsr01/motion/move_joint')
            while not movej_cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Waiting for /dsr01/motion/move_joint service...")

            self.get_logger().info("Moving to CHEERS_POSE (Prime)...")
            req = MoveJoint.Request()
            req.pos = [float(x) for x in CHEERS_POSE]
            req.vel = float(VELOCITY)
            req.acc = float(ACC)
            req.time = 0.0
            req.mode = 0 # ABS
            req.radius = 0.0
            req.blend_type = 0 # DUPLICATE
            req.sync_type = 0 # SYNC

            future = movej_cli.call_async(req)
            while not future.done():
                time.sleep(0.1)
            self.get_logger().info("Move CHEERS_POSE finished.")

            # Wait for robot state transition to settle
            time.sleep(1.0)
            
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            self.get_logger().info("그리퍼를 활성화합니다...")
            time.sleep(0.5)
            self.gripper_is_open = True
            self.gripper.move(0)
            self.get_logger().info("그리퍼 활성화 완료.")
        except Exception as e:
            import traceback
            self.get_logger().error(f"Robot Initialization Error: {e}\n{traceback.format_exc()}")

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
            return
        if not self.task_received or self.items is None:
            return
        if self.latest_cv_color is None or self.latest_cv_depth_mm is None or self.intrinsics is None:
            self.get_logger().info("카메라 입력 준비 전이라 주문을 보류합니다.")
            return
        
        import threading
        self.state = "RUNNING"
        threading.Thread(target=self.process_grip, args=(self.items,), daemon=True).start()

    def camera_to_robot(self, point_cam):
        """
        point_cam: numpy array shape (3,)
        row-vector form:
            p_robot = p_cam @ R + t
        """
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

        self._try_start_task()

    def terminate_gripper(self):
        if self.gripper:
            self.gripper.terminate()

    def _movej(self, pos, vel=30.0, acc=30.0, mode=0, radius=0.0, blend_type=0, sync_type=0):
        from dsr_msgs2.srv import MoveJoint
        cli = self.create_client(MoveJoint, '/dsr01/motion/move_joint')
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for MoveJoint service...")
        req = MoveJoint.Request()
        req.pos = [float(x) for x in pos]
        req.vel = float(vel)
        req.acc = float(acc)
        req.mode = int(mode)
        req.radius = float(radius)
        req.blend_type = int(blend_type)
        req.sync_type = int(sync_type)
        future = cli.call_async(req)
        import time
        while not future.done():
            time.sleep(0.01)
        return future.result()

    def _movel(self, pos, vel=[30.0, 30.0], acc=[30.0, 30.0], mode=0, radius=0.0, blend_type=0, sync_type=0):
        from dsr_msgs2.srv import MoveLine
        cli = self.create_client(MoveLine, '/dsr01/motion/move_line')
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for MoveLine service...")
        req = MoveLine.Request()
        req.pos = [float(x) for x in pos]
        req.vel = [float(x) for x in vel]
        req.acc = [float(x) for x in acc]
        req.mode = int(mode)
        req.radius = float(radius)
        req.blend_type = int(blend_type)
        req.sync_type = int(sync_type)
        future = cli.call_async(req)
        import time
        while not future.done():
            time.sleep(0.01)
        return future.result()

    def process_grip(self, items):
        # 이미 작업 중이면 무시 (중복 방지)
        # (self.state check is already done in _try_start_task)

        self.get_logger().info(f"Received item command: {self.items}")
        
        # TODO: 해당 노드 안에서 반복을 할지, 해당 노드를 여러번 불러올지 선택해야 함
        # --------------------
        # 주종에 따른 반복문 처리
        # --------------------
        # class_ids = [x for x in items["recipe"].keys()]
        # volumes = [y for y in items["recipe"].values()]

        # for i, (x, y) in enumerate(class_ids, volumes):

        class_id = object_dict_reverse[[x for x in items["recipe"].keys()][0]]
        
        # TODO: 따라야 하는 술의 양
        volume = [y for y in items["recipe"].values()][0]
        
        from DR_common2 import posx, posj

        try:
            # Robot mode is already set to AUTONOMOUS in _init_robot
            self.get_logger().info("Starting grip process (Robot should be in AUTONOMOUS mode)...")
            time.sleep(0.1)

            if self.latest_cv_depth_mm is None or self.intrinsics is None:
                self.get_logger().warn("아직 뎁스 프레임 또는 카메라 정보가 수신되지 않았습니다.")
                return

            self.get_logger().info("초기 자세")
            home_posj = [69.5, -43.0, 102.0, 101.0, -72.0, -213.0]
            self._movej(home_posj, vel=VELOCITY, acc=ACC)
            self.gripper.move(0)
            time.sleep(2)

            img_raw = self.latest_cv_color.copy()
            depth_raw = self.latest_cv_depth_mm.copy()

            # 화면 표시/YOLO용으로만 flip
            img = cv2.flip(img_raw, -1)
            self.latest_cv_vis = cv2.flip(img_raw.copy(), -1)

            h, w = img.shape[:2]

            object_loc_dict = {'0': [], '1': [], '2': []}

            # Detect objects using YOLO
            results = model(img)

            # Process the result
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype('int')
                    confidence = box.conf[0].cpu().numpy()
                    cls_id = str(int(box.cls[0].cpu().numpy()))

                    if confidence < 0.5:
                        continue

                    # flip된 화면 기준 bbox 중심
                    u_flip = int((x1 + x2) / 2)
                    v_flip = int((y1 + y2) / 2)

                    # flip된 화면 좌표 -> 원본 센서 좌표
                    u_raw = w - 1 - u_flip
                    v_raw = h - 1 - v_flip

                    # 원본 depth에서 작은 ROI로 median 계산
                    roi_half = 5
                    roi_half = 5
                    x_min = max(0, u_raw - roi_half)
                    x_max = min(w, u_raw + roi_half + 1)
                    y_min = max(0, v_raw - roi_half)
                    y_max = min(h, v_raw + roi_half + 1)

                    depth_roi = depth_raw[y_min:y_max, x_min:x_max]
                    valid_depth = depth_roi[depth_roi > 0]

                    if valid_depth.size == 0:
                        continue

                    object_depth = float(np.median(valid_depth))  # mm

                    label = f"{object_dict[cls_id]} / {object_depth*0.001:.2f}m"
                    color = CLASS_COLORS.get(int(cls_id), (255, 255, 255))

                    # flip 화면 좌표와 raw 좌표를 둘 다 저장
                    object_loc_dict[str(int(cls_id))] = {
                        "u_flip": u_flip, "v_flip": v_flip,
                        "u_raw": u_raw, "v_raw": v_raw,
                        "depth_mm": object_depth,
                    }

                    # Draw a rectangle around the object
                    cv2.rectangle(self.latest_cv_vis, (x1, y1), (x2, y2), color, 2)

                    # Draw the bounding box
                    cv2.putText(self.latest_cv_vis, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            target = object_loc_dict[class_id]
            depth_mm = target["depth_mm"]

            if depth_mm == 0:
                print(f"해당 {model.names[class_id]}의 깊이를 측정할 수 없습니다 (값: 0)")
                return

            depth_m = float(depth_mm) / 1000.0

            # deproject는 반드시 raw 좌표 사용
            u_raw = target["u_raw"]
            v_raw = target["v_raw"]

            # 시각화는 flip 좌표 사용
            u_vis = target["u_flip"]
            v_vis = target["v_flip"]

            # 카메라 calibration 시 좌표계를 새로 만들었기 때문에
            # deproject 대신 아래 코드를 사용
            cx = self.intrinsics.ppx
            cy = self.intrinsics.ppy
            fx = self.intrinsics.fx
            fy = self.intrinsics.fy

            # 중심 원점 기준 정규화
            x_cam = (u_raw - cx) * depth_m / fx
            y_cam = (v_raw - cy) * depth_m / fy
            z_cam = depth_m

            # RealSense raw camera coordinates
            x_mm = x_cam * 1000
            y_mm = y_cam * 1000
            z_mm = z_cam * 1000

            p_cam = np.array([x_mm, y_mm, z_mm], dtype=np.float64)
            p_robot = self.camera_to_robot(p_cam)

            print("--- 변환된 최종 3D 좌표 ---")
            print(f"flip 픽셀 좌표: (u_vis={u_vis}, v_vis={v_vis})")
            print(f"raw 픽셀 좌표 : (u_raw={u_raw}, v_raw={v_raw}), Depth: {depth_m*1000:.1f} mm")
            print(f"로봇 목표 좌표: X={p_robot[0]:.1f}, Y={p_robot[1]:.1f}, Z={p_robot[2]:.1f}\n")

            if self.latest_cv_vis is not None:
                vis = self.latest_cv_vis.copy()

                # 좌표 계산에 사용한 중심점 표시
                cv2.circle(vis, (u_vis, v_vis), 6, (0, 255, 255), -1)

                # 픽셀 좌표 표시
                cv2.putText(vis, f"flip center: ({u_vis}, {v_vis})", (u_vis + 10, v_vis + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(vis, f"raw center: ({u_raw}, {v_raw})", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 로봇 좌표 표시
                cv2.putText(vis, f"Robot XYZ: ({p_robot[0]:.1f}, {p_robot[1]:.1f}, {p_robot[2]:.1f})", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            self.move_robot_and_control_gripper(p_robot[0], p_robot[1], p_robot[2])
            action_msg = String()
            action_msg.data = json.dumps(
                {
                    "action": "pour",
                    "count": 1,
                    "drinks": items.get("drinks", ""),
                    "recipe": items.get("recipe", {}),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            self.action_pub.publish(action_msg)
            self.get_logger().info(f"후속 액션 요청 발행: {action_msg.data}")
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
            # TODO: Debugging을 위해서 logger를 많이 사용함
            # 통합 과정에서 일부 생략해도 됨
            self.get_logger().info("잡기 전 자세")
            P_ready = [28.0, -35.0, 100.0, 77.0, 63.0, -154.0]
            self._movej(P_ready, VELOCITY, ACC)

            # Get current pose via service
            from dsr_msgs2.srv import GetCurrentPose
            pose_cli = self.create_client(GetCurrentPose, '/dsr01/system/get_current_pose')
            while not pose_cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Waiting for GetCurrentPose service...")
            pose_req = GetCurrentPose.Request()
            pose_req.space_type = 1 # TASK
            pose_future = pose_cli.call_async(pose_req)
            while not pose_future.done():
                time.sleep(0.01)
            current_pos = list(pose_future.result().pos)

            self.get_logger().info(f"current_pos : {current_pos}")
            target_pos_list_1 = [x - 20, y - 50, z, current_pos[3], current_pos[4], current_pos[5]]
            target_pos_list_2 = [x - 20, y + 50, z -20, current_pos[3], current_pos[4], current_pos[5]]
            self.get_logger().info(f"target_pos_list_1 : {target_pos_list_1}")
            self.get_logger().info(f"target_pos_list_2 : {target_pos_list_2}")            
            
            self.get_logger().info("목표 지점으로 접근합니다")
            self._movel(target_pos_list_1, vel=[40.0, 40.0], acc=[ACC, ACC])

            self.get_logger().info(f"목표 지점으로 이동합니다: {target_pos_list_2}")
            self._movel(target_pos_list_2, vel=[40.0, 40.0], acc=[ACC, ACC])

            # gripper 주류 잡기
            self.gripper.move(550)
            time.sleep(1)

            # 중간 자세
            P_mid = [61.0, -20.0, 97.0, 96.0, -63.0, -195.0]
            self._movej(P_mid, VELOCITY, ACC)
            time.sleep(1)

            # 따르기 전 자세
            self.get_logger().info("따르기 위한 자세를 취합니다.")
            last_posj = [45.0, 0.0, 135.0, 90.0, -90.0, -135.0]
            self._movej(last_posj, VELOCITY, ACC)

        except Exception as e:
            self.get_logger().error(f"로봇 이동 및 그리퍼 제어 중 오류 발생: {e}")


def main(args=None):
    rclpy.init(args=args)

    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    try:
        from DSR_ROBOT2 import get_current_posx, movel, wait, movej
        from DR_common2 import posx, posj
    except ImportError as e:
        print(f"DSR_ROBOT2 라이브러리를 import할 수 없습니다: {e}")
        rclpy.shutdown()
        return

    robot_controller = None
    try:
        robot_controller = RobotControllerNode()

        rclpy.spin(robot_controller)
    
    except Exception as e:
        print(f"노드 실행 중 오류: {e}")
    
    finally:
        print("프로그램을 종료합니다")
        if robot_controller is not None:
            robot_controller.terminate_gripper()
            robot_controller.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()
        print("종료 완료")


if __name__ == "__main__":
    main()
