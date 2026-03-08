import os
import sys
import glob
import json
import csv
import shlex
import signal
import subprocess
import threading
import time
import traceback
import warnings
from datetime import datetime

from PyQt5.QtCore import QTimer, QObject, pyqtSignal, Qt, QThread, QEvent
from PyQt5.QtGui import QFont, QImage, QPixmap, QBrush, QColor, QDoubleValidator, QPalette, QTransform
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QInputDialog,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QDialogButtonBox,
    QButtonGroup,
    QLabel,
    QProgressBar,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFrame,
    QPushButton,
    QLineEdit,
    QCheckBox,
    QFileDialog,
    QSizePolicy,
)
from PyQt5 import uic

try:
    from rcl_interfaces.msg import Log as RosLogMsg
except Exception:
    RosLogMsg = None
try:
    from sensor_msgs.msg import Image as RosImageMsg
except Exception:
    RosImageMsg = None
try:
    from sensor_msgs.msg import CameraInfo as RosCameraInfoMsg
except Exception:
    RosCameraInfoMsg = None
try:
    from geometry_msgs.msg import PointStamped as RosPointStamped
except Exception:
    RosPointStamped = None
try:
    from std_msgs.msg import String as RosStringMsg
except Exception:
    RosStringMsg = None
try:
    from rclpy.qos import (
        qos_profile_sensor_data,
        QoSProfile,
        QoSHistoryPolicy,
        QoSReliabilityPolicy,
    )
except Exception:
    qos_profile_sensor_data = None
    QoSProfile = None
    QoSHistoryPolicy = None
    QoSReliabilityPolicy = None
try:
    from rclpy.callback_groups import ReentrantCallbackGroup
except Exception:
    ReentrantCallbackGroup = None
import numpy as np

# Project layout root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ROS2 logs to stdout so UI terminal can capture
os.environ.setdefault("RCUTILS_LOGGING_USE_STDOUT", "1")
os.environ.setdefault("RCUTILS_LOGGING_BUFFERED_STREAM", "1")
os.environ.setdefault("RCUTILS_LOGGING_SEVERITY", "WARN")

warnings.filterwarnings(
    "ignore",
    message=r"A NumPy version >=1\.17\.3 and <1\.25\.0 is required for this version of SciPy.*",
    category=UserWarning,
)

from task_backend_node import RobotBackend, ROBOT_ID, HOME_POSJ

form = uic.loadUiType(os.path.join(PROJECT_ROOT, "assets", "frontend", "developer_frontend.ui"))[0]
UI_FONT_FAMILY = "Noto Sans CJK KR"
UI_FONT_SIZE = 9
UI_TERMINAL_FONT_SIZE = max(6, int(os.environ.get("UI_TERMINAL_FONT_SIZE", "10")))
UI_TABLE_FONT_SIZE = max(12, int(os.environ.get("UI_TABLE_FONT_SIZE", "12")))
UI_TABLE_ROW_HEIGHT = max(26, int(os.environ.get("UI_TABLE_ROW_HEIGHT", "30")))
UI_PANEL_TABLE_ROW_HEIGHT = max(22, int(UI_TABLE_ROW_HEIGHT * 0.875))
STARTUP_USE_REAL_GRIPPER = os.environ.get("STARTUP_USE_REAL_GRIPPER", "1") == "1"

YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "")
YOLO_CAMERA_INDEX = int(os.environ.get("YOLO_CAMERA_INDEX", "0"))
YOLO_ENABLE_TRACK = os.environ.get("YOLO_ENABLE_TRACK", "0") == "1"
YOLO_INFER_EVERY_N = max(1, int(os.environ.get("YOLO_INFER_EVERY_N", "2")))
UI_ENABLE_VISION = os.environ.get("UI_ENABLE_VISION", "1") == "1"
YOLO_EXTERNAL_NODE = os.environ.get("YOLO_EXTERNAL_NODE", "1") == "1"
YOLO_ALLOW_INTERNAL_WEBCAM = os.environ.get("YOLO_ALLOW_INTERNAL_WEBCAM", "0") == "1"
YOLO_EXTERNAL_TOPIC = os.environ.get("YOLO_EXTERNAL_TOPIC", "/yolo/annotated_image")
YOLO_EXTERNAL_TOPIC_2 = os.environ.get("YOLO_EXTERNAL_TOPIC_2", "/yolo2/annotated_image")
CALIB_VISION_TOPIC_PRIMARY = os.environ.get("CALIB_VISION_TOPIC", "/camera/camera/color/image_raw")
CALIB_VISION_TOPIC_FALLBACK = os.environ.get("CALIB_VISION_TOPIC_FALLBACK", "/camera/color/image_raw")
CALIB_CAMERA_INFO_TOPIC_PRIMARY = os.environ.get("CALIB_CAMERA_INFO_TOPIC", "/camera/camera/color/camera_info")
CALIB_CAMERA_INFO_TOPIC_FALLBACK = os.environ.get("CALIB_CAMERA_INFO_TOPIC_FALLBACK", "/camera/color/camera_info")
CALIB_OUTPUT_META_TOPIC_1 = os.environ.get("CALIB_OUTPUT_META_TOPIC_1", "/vision1/calibration/meta")
CALIB_OUTPUT_META_TOPIC_2 = os.environ.get("CALIB_OUTPUT_META_TOPIC_2", "/vision2/calibration/meta")
YOLO_AUTO_LAUNCH_NODE = os.environ.get("YOLO_AUTO_LAUNCH_NODE", "0") == "1"
YOLO_AUTO_LAUNCH_ALWAYS = os.environ.get("YOLO_AUTO_LAUNCH_ALWAYS", "0") == "1"
YOLO_AUTO_LAUNCH_CMD = os.environ.get("YOLO_AUTO_LAUNCH_CMD", "").strip()
CALIB_HELPER_AUTO_LAUNCH = os.environ.get("CALIB_HELPER_AUTO_LAUNCH", "0") == "1"
CALIB_HELPER_CMD = os.environ.get("CALIB_HELPER_CMD", "").strip()
YOLO_EXTERNAL_DEPTH_TOPIC = os.environ.get(
    "YOLO_EXTERNAL_DEPTH_TOPIC",
    os.environ.get("YOLO_DEPTH_TOPIC", "/camera/camera/aligned_depth_to_color/image_raw"),
)

POSITION_STALE_SEC = float(os.environ.get("UI_POSITION_STALE_SEC", "2.0"))
STATE_FLASH_SEC = float(os.environ.get("UI_STATE_FLASH_SEC", "1.2"))
POSITION_FLASH_SEC = float(os.environ.get("UI_POSITION_FLASH_SEC", "0.9"))
STATE_COLOR_NORMAL = "#14863B"
STATE_COLOR_WARNING = "#EF6C00"
STATE_COLOR_ERROR = "#C62828"
TOP_STATUS_BAR_HEIGHT = 40
TOP_STATUS_GAP = 10
VISION_MOVE_Z_MARGIN_MM = 200.0
CALIB_SEQUENCE_VELOCITY = float(os.environ.get("CALIB_SEQUENCE_VELOCITY", "20"))
CALIB_SEQUENCE_ACC = float(os.environ.get("CALIB_SEQUENCE_ACC", "20"))
JOINT_INPUT_LIMITS_DEG = (
    (-360.0, 360.0),  # J1
    (-360.0, 360.0),  # J2
    (-155.0, 155.0),  # J3
    (-360.0, 360.0),  # J4
    (-360.0, 360.0),  # J5
    (-360.0, 360.0),  # J6
)
LOG_AREA_SHIFT_Y = 4
UI_USE_DESIGN_GEOMETRY = True
MODE_SWITCH_GRACE_SEC = float(os.environ.get("MODE_SWITCH_GRACE_SEC", "4.0"))
PARAM_DIR = os.path.join(PROJECT_ROOT, "config")
PARAM_FILE = os.path.join(PARAM_DIR, "parameter.csv")
CALIB_DIR = os.path.join(PARAM_DIR, "calibration")
CALIB_ROBOT_DIR = CALIB_DIR
CALIB_ROTMAT_DIR = CALIB_DIR
LEGACY_CALIB_DIR = os.path.join(PROJECT_ROOT, "calibration")
LEGACY_CALIB_ROBOT_DIR = os.path.join(LEGACY_CALIB_DIR, "Data_robot")
LEGACY_CALIB_ROTMAT_DIR = os.path.join(LEGACY_CALIB_DIR, "Data_RotMat")
LEGACY_CALIB_ACTIVE_PATH_FILE = os.path.join(LEGACY_CALIB_DIR, "active_calibration_path.txt")
CALIB_DEFAULT_PATTERN_COLS = 7
CALIB_DEFAULT_PATTERN_ROWS = 9
CALIB_DETECTION_HOLD_SEC = float(os.environ.get("CALIB_DETECTION_HOLD_SEC", "1.2"))
CALIB_DETECT_INTERVAL_SEC = float(os.environ.get("CALIB_DETECT_INTERVAL_SEC", "0.18"))
CALIB_PROCESS_HZ = float(os.environ.get("CALIB_PROCESS_HZ", "10.0"))
DEFAULT_VISION1_SERIAL = os.environ.get("DEFAULT_VISION1_SERIAL", "313522301601")
DEFAULT_VISION2_SERIAL = os.environ.get("DEFAULT_VISION2_SERIAL", "311322302867")
CALIB_SEQUENCE_ROW_DEFS = [
    ("home", "홈위치"),
    ("wait1", "대기위치1"),
    ("wait2", "대기위치2"),
    ("p1", "데이터1"),
    ("p2", "데이터2"),
    ("p3", "데이터3"),
    ("p4", "데이터4"),
    ("p5", "데이터5"),
    ("p6", "데이터6"),
    ("p7", "데이터7"),
    ("p8", "데이터8"),
    ("p9", "데이터9"),
    ("p10", "데이터10"),
    ("return_home", "복귀위치"),
    ("end_home", "홈위치"),
]

# Vision(U,V,Zmm) -> Robot(X,Y,Zmm) calibration pairs
VISION_CALIB_UVZ_MM = np.array(
    [
        [470.0, 283.0, 657.0],  # P1
        [363.0, 286.0, 656.0],  # P2
        [468.0, 137.0, 647.0],  # P3
        [361.0, 139.0, 648.0],  # P4
        [419.0, 209.0, 652.0],  # P5
    ],
    dtype=np.float64,
)
ROBOT_CALIB_XYZ_MM = np.array(
    [
        [603.11, 53.40, 175.51],    # P1
        [601.41, -126.30, 175.56],  # P2
        [355.93, 56.11, 173.28],    # P3
        [354.56, -125.54, 172.57],  # P4
        [474.10, -30.35, 173.89],   # P5
    ],
    dtype=np.float64,
)

ROBOT_STATE_KR_MAP = {
    "STATE_INITIALIZING": "초기화",
    "STATE_STANDBY": "대기",
    "STATE_MOVING": "이동 중",
    "STATE_SAFE_OFF": "세이프 오프",
    "STATE_TEACHING": "티칭",
    "STATE_SAFE_STOP": "세이프 스톱",
    "STATE_EMERGENCY_STOP": "비상 정지",
    "STATE_HOMMING": "홈 동작",
    "STATE_RECOVERY": "복구",
    "STATE_SAFE_STOP2": "세이프 스톱2",
    "STATE_SAFE_OFF2": "세이프 오프2",
    "STATE_RESERVED1": "예약 상태1",
    "STATE_RESERVED2": "예약 상태2",
    "STATE_RESERVED3": "예약 상태3",
    "STATE_RESERVED4": "예약 상태4",
    "STATE_NOT_READY": "준비 안됨",
    "STATE_LAST": "마지막 상태",
}

ROBOT_STATE_NORMAL_CODES = {1, 2}
ROBOT_STATE_WARNING_CODES = {0, 4, 5, 8, 9, 10, 12, 13, 14}
ROBOT_STATE_ERROR_CODES = {3, 6, 7, 11, 15, 16}


def _default_model_path():
    model_dir = os.path.join(PROJECT_ROOT, "assets", "models")
    for name in ("bartender_yolo.pt", "snack_detector.pt", "best.pt"):
        candidate = os.path.join(model_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(model_dir, "best.pt")


if not YOLO_MODEL_PATH:
    YOLO_MODEL_PATH = _default_model_path()


def _project_relative_path(path_value):
    if not path_value:
        return ""
    abs_path = os.path.abspath(str(path_value))
    try:
        rel_path = os.path.relpath(abs_path, PROJECT_ROOT)
    except ValueError:
        return abs_path
    if rel_path.startswith(".."):
        return abs_path
    return rel_path.replace(os.sep, "/")


def _resolve_repo_file(path_value, fallback_dirs=None):
    raw = str(path_value or "").strip()
    if not raw:
        return ""

    normalized = raw.replace("\\", "/")
    candidates = []
    if os.path.isabs(raw):
        candidates.append(os.path.abspath(raw))
    else:
        candidates.append(os.path.abspath(os.path.join(PROJECT_ROOT, raw)))

    basename = os.path.basename(normalized)
    if basename:
        for base_dir in fallback_dirs or ():
            candidates.append(os.path.join(base_dir, basename))

    for marker in ("config/calibration/", "param/calibration/", "calibration/"):
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1]
            suffix = suffix.replace("Data_RotMat/", "").replace("Data_robot/", "")
            suffix = suffix.lstrip("/")
            if suffix:
                candidates.append(os.path.join(CALIB_DIR, suffix))
                candidates.append(os.path.join(CALIB_DIR, os.path.basename(suffix)))

    seen = set()
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            return candidate
    return ""


class EmittingStream(QObject):
    text_written = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._buffer = ""

    def write(self, text):
        if not text:
            return
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.text_written.emit(line + "\n")

    def flush(self):
        if self._buffer:
            self.text_written.emit(self._buffer)
            self._buffer = ""


class BackendStartupWorker(QObject):
    progress = pyqtSignal(int, str, float)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str, str)

    def __init__(self, use_real_gripper):
        super().__init__()
        self.use_real_gripper = use_real_gripper

    def run(self):
        started = time.monotonic()
        backend = RobotBackend(use_real_gripper=self.use_real_gripper)

        def progress_callback(percent, message):
            elapsed = time.monotonic() - started
            self.progress.emit(percent, message, elapsed)

        try:
            backend.start(progress_callback=progress_callback)
            elapsed = time.monotonic() - started
            self.progress.emit(100, "초기화 완료", elapsed)
            self.finished.emit(backend)
        except Exception as e:
            self.failed.emit(str(e), traceback.format_exc())


class BackendResetWorker(QObject):
    finished = pyqtSignal(bool, str)
    failed = pyqtSignal(str, str)

    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def run(self):
        try:
            ok, msg = self.backend.reset_robot_state()
            self.finished.emit(bool(ok), str(msg))
        except Exception as e:
            self.failed.emit(str(e), traceback.format_exc())


class App(QMainWindow, form):
    ros_log_received = pyqtSignal(str)
    ros_image_received = pyqtSignal(QImage)
    calibration_ui_refresh_requested = pyqtSignal()

    def __init__(self, backend=None, auto_start_backend=True):
        super().__init__()
        self.setupUi(self)
        self._status_row_ready = False

        self.backend = backend
        self._auto_start_backend = bool(auto_start_backend)
        self._backend_thread = None
        self._backend_worker = None
        self._reset_thread = None
        self._reset_worker = None
        self._mode_scan_at = 0.0
        self._mode_text_cached = "판단 중"
        self._table_value_cache = {}
        self._table_flash_until = {}
        self._pos_value_cache = {}
        self._pos_flash_until = {}
        self._current_tool_label = None
        self._current_tool_text_cache = ""
        self._last_robot_state_text = ""
        self._robot_state_flash_until = 0.0
        self._source_scan_at = 0.0
        self._source_scan_interval_sec = 1.0
        self._realsense_connected = False
        self._yolo_topic_connected = False
        self._top_status_widgets = {}
        self._top_status_toggles = {}
        self._top_status_enabled = {"vision": True, "vision2": True, "robot": True}
        self._top_status_panel = None
        self._top_status_mid_line = None
        self._top_status_state_cache = {}
        self._robot_controls_enabled_cache = None

        self._reserve_top_status_space()
        self._setup_top_status_row()

        self.terminal.setReadOnly(True)
        self.terminal.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.terminal.document().setMaximumBlockCount(600)
        self.terminal.setFont(QFont(UI_FONT_FAMILY, UI_TERMINAL_FONT_SIZE))
        self.terminal.setStyleSheet(
            f"font-family: '{UI_FONT_FAMILY}'; font-size: {UI_TERMINAL_FONT_SIZE}pt;"
            " selection-background-color: #1f6feb;"
            " selection-color: #ffffff;"
        )
        self._setup_log_controls()

        # Control-button wiring reads these states, so initialize before _setup_control_buttons().
        self._vision_click_dialog_enabled = False
        self._vision_move_z_margin_mm = float(VISION_MOVE_Z_MARGIN_MM)
        self._calibration_mode_enabled = False
        self._calibration_mode_enabled_1 = False
        self._calibration_mode_enabled_2 = False
        self._calib_pattern_cols = CALIB_DEFAULT_PATTERN_COLS
        self._calib_pattern_rows = CALIB_DEFAULT_PATTERN_ROWS
        self._calib_last_points_uvz_mm = None
        self._calib_last_points_uvz_mm_1 = None
        self._calib_last_points_uvz_mm_2 = None
        self._calib_last_points_at = None
        self._calib_last_points_at_1 = None
        self._calib_last_points_at_2 = None
        self._calib_last_detect_run_at_1 = 0.0
        self._calib_last_detect_run_at_2 = 0.0
        self._calib_last_grid_pts_1 = None
        self._calib_last_grid_pts_2 = None
        self._calib_last_grid_status_1 = None
        self._calib_last_grid_status_2 = None
        self._calib_matrix_path = None
        self._calib_matrix_path_1 = None
        self._calib_matrix_path_2 = None
        self._calib_last_reason = "대기"
        self._calib_last_reason_1 = "대기"
        self._calib_last_reason_2 = "대기"
        self._calib_full_ready_1 = False
        self._calib_full_ready_2 = False
        self._calib_full_reason_1 = "대기"
        self._calib_full_reason_2 = "대기"
        self._calib_proc_1 = None
        self._calib_proc_2 = None
        self._calib_proc_cmd_1 = None
        self._calib_proc_cmd_2 = None
        self._calib_proc_started_by_ui_1 = False
        self._calib_proc_started_by_ui_2 = False
        self._calib_status_blink_on = False
        self._calib_grid_camera_xyz_1 = None
        self._calib_grid_camera_xyz_2 = None
        self._calibration_sequence_running = False

        self._setup_control_buttons()
        self._set_robot_controls_enabled(False)

        # logging state must exist before stdout/stderr redirection starts
        self._log_buffer = []
        self._last_log_line = ""
        self._last_log_line_at = 0.0
        self._stdout = EmittingStream()
        self._stderr = EmittingStream()
        self._stdout.text_written.connect(self.append_log)
        self._stderr.text_written.connect(self.append_log)
        self.ros_log_received.connect(self.append_log)
        self.ros_image_received.connect(self._update_yolo_view)
        self.calibration_ui_refresh_requested.connect(self._update_calibration_mode_ui)
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        sys.excepthook = self._handle_exception
        if hasattr(threading, "excepthook"):
            threading.excepthook = self._handle_thread_exception

        self._log_timer = QTimer(self)
        self._log_timer.timeout.connect(self._flush_log_buffer)
        self._log_timer.start(100)
        self.append_log("[시작] 메인창 실행 완료\n")
        self.append_log("[시작] 백엔드 초기화 대기 중...\n")

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._refresh_robot_status)
        self._status_timer.start(250)
        self._vision_status_timer = QTimer(self)
        self._vision_status_timer.timeout.connect(self._refresh_vision_status)
        self._vision_status_timer.start(300)
        self._top_status_anim_timer = QTimer(self)
        self._top_status_anim_timer.timeout.connect(self._tick_top_status_animation)
        self._top_status_anim_timer.start(30)
        self._calib_status_blink_timer = QTimer(self)
        self._calib_status_blink_timer.timeout.connect(self._tick_calibration_status_blink)
        self._calib_status_blink_timer.start(500)

        self._pos_timer = QTimer(self)
        self._pos_timer.timeout.connect(self._refresh_positions)
        self._pos_timer.start(200)
        self._vision_render_timer_1 = QTimer(self)
        self._vision_render_timer_1.timeout.connect(self._drain_pending_vision_frame_1)
        self._vision_render_timer_1.start(16)
        self._vision_render_timer_2 = QTimer(self)
        self._vision_render_timer_2.timeout.connect(self._drain_pending_vision_frame_2)
        self._vision_render_timer_2.start(16)

        self._yolo_thread = None
        self._yolo_worker = None
        self._rosout_sub = None
        self._vision_sub = None
        self._vision_sub_2 = None
        self._calib_meta_sub_1 = None
        self._calib_meta_sub_2 = None
        self._vision_depth_sub = None
        self._vision_depth_sub_2 = None
        self._vision_cb_group_1 = ReentrantCallbackGroup() if ReentrantCallbackGroup is not None else None
        self._vision_cb_group_2 = ReentrantCallbackGroup() if ReentrantCallbackGroup is not None else None
        self._vision_point_sub = None
        self._vision_preview_sub = None
        self._vision_info_sub = None
        self._vision_info_sub_2 = None
        self._vision_image_topic_in_use = None
        self._vision_image_topic_in_use_2 = None
        self._calib_meta_topic_in_use_1 = None
        self._calib_meta_topic_in_use_2 = None
        self._vision_rebind_last_try_at = 0.0
        self._vision_bridge_fail_last_at = 0.0
        self._vision_bridge_fail_last_msg = ""
        self._external_vision_proc = None
        self._external_vision_cmd = None
        self._external_vision_started_by_ui = False
        self._last_yolo_image_size = None
        self._last_yolo_image_size_2 = None
        self._vision_depth_image = None
        self._vision_depth_image_2 = None
        self._vision_depth_encoding = None
        self._vision_depth_encoding_2 = None
        self._vision_depth_shape = None
        self._vision_depth_shape_2 = None
        self._camera_intrinsics = None  # legacy alias for panel 1
        self._camera_intrinsics_1 = None  # (fx, fy, cx, cy)
        self._camera_intrinsics_2 = None  # (fx, fy, cx, cy)
        self._last_mouse_xy = None
        self._last_mouse_xy_2 = None
        self._last_clicked_vision_xyz_mm = None
        self._last_clicked_robot_xyz_mm = None
        self._last_clicked_robot_target_xyz_mm = None
        self._last_clicked_source_vision = 1
        self._yolo_zoom = 1.0
        self._yolo_zoom_min = 1.0
        self._yolo_zoom_max = 5.0
        # 시작 시 좌표계 원점(좌상단) 축이 보이도록 초기 포커스를 좌상단으로 둔다.
        self._yolo_pan_x = -1000000.0
        self._yolo_pan_y = -1000000.0
        self._yolo_panning = False
        self._yolo_pan_last_pos = None
        self._yolo_zoom_2 = 1.0
        self._yolo_pan_x_2 = -1000000.0
        self._yolo_pan_y_2 = -1000000.0
        self._yolo_panning_2 = False
        self._yolo_pan_last_pos_2 = None
        self._last_yolo_qimage = None
        self._last_yolo_qimage_2 = None
        self._pending_yolo_qimage = None
        self._pending_yolo_qimage_2 = None
        self._pending_vision_token_1 = 0
        self._pending_vision_token_2 = 0
        self._vision_frame_pending = False
        self._vision_frame_pending_2 = False
        self._vision_frame_lock_1 = threading.Lock()
        self._vision_frame_lock_2 = threading.Lock()
        self._yolo_view_map = None
        self._yolo_view_map_2 = None
        self._last_robot_posx = None
        self._last_positions_seen_at = None
        self._last_vision_frame_at = None
        self._last_vision_frame_at_2 = None
        self._last_camera_frame_at_1 = None
        self._last_camera_frame_at_2 = None
        self._last_vision_point_at = None
        self._vision_prev_update_at = None
        self._vision_prev_update_at_2 = None
        self._vision_render_prev_at = None
        self._vision_render_prev_at_2 = None
        self._robot_prev_state_seen_at = None
        self._vision_cycle_ms = None
        self._vision_cycle_ms_2 = None
        self._vision_decode_ms = None
        self._vision_decode_ms_2 = None
        self._vision_render_delay_ms = None
        self._vision_render_delay_ms_2 = None
        self._vision_render_interval_ms = None
        self._vision_render_interval_ms_2 = None
        self._pending_vision_enqueued_at_1 = None
        self._pending_vision_enqueued_at_2 = None
        self._calib_proc_input_ms_1 = None
        self._calib_proc_input_ms_2 = None
        self._calib_proc_ms_1 = None
        self._calib_proc_ms_2 = None
        self._calib_proc_publish_ms_1 = None
        self._calib_proc_publish_ms_2 = None
        self._calib_proc_wait_ms_1 = None
        self._calib_proc_wait_ms_2 = None
        self._robot_cycle_ms = None
        self._vision_state_text = "끊김"
        self._vision_state_text_2 = "끊김"
        self._vision_to_robot_affine = None
        self._vision_to_robot_rmse = None
        self._vision_to_robot_affine_1 = None
        self._vision_to_robot_affine_2 = None
        self._vision_to_robot_rmse_1 = None
        self._vision_to_robot_rmse_2 = None
        self._vision_assigned_serial_1 = DEFAULT_VISION1_SERIAL
        self._vision_assigned_serial_2 = DEFAULT_VISION2_SERIAL
        self._runtime_camera_serial_1 = DEFAULT_VISION1_SERIAL
        self._runtime_camera_serial_2 = DEFAULT_VISION2_SERIAL
        self._available_camera_serials = []
        self._vision_serial_label = None
        self._vision_serial_label_2 = None
        self._vision_serial_change_button = None
        self._vision_serial_change_button_2 = None
        self._vision_rotate_left_button = None
        self._vision_rotate_zero_button = None
        self._vision_rotate_right_button = None
        self._vision_rotation_label = None
        self._vision_rotate_left_button_2 = None
        self._vision_rotate_zero_button_2 = None
        self._vision_rotate_right_button_2 = None
        self._vision_rotation_label_2 = None
        self._vision_rotation_deg_1 = 0
        self._vision_rotation_deg_2 = 0
        self._vision_stream_token_1 = 0
        self._vision_stream_token_2 = 0
        self._load_vision_serial_settings()
        self._set_vision_panel_controls_enabled(1, bool(self._top_status_enabled.get("vision", True)))
        self._set_vision_panel_controls_enabled(2, bool(self._top_status_enabled.get("vision2", True)))
        self._sync_vision_render_timers()
        self._save_vision_serial_settings()
        self._mode_switch_grace_until = 0.0
        self._vision_drop_frames_until_1 = 0.0
        self._vision_drop_frames_until_2 = 0.0
        self._last_robot_comm_connected = False
        self._build_vision_to_robot_affine()
        self._try_load_calibration_matrix_on_startup()
        self._update_calibration_mode_ui()
        self._svc_retry_log_last_at = 0.0
        self._vision_retry_notice_logged = False

        self._vision_state_label = getattr(self, "vision_state_label", None)
        if self._vision_state_label is not None:
            self._vision_state_label.setText("끊김")
        self._vision_coord_label = getattr(self, "vision_coord_label", None)
        if self._vision_coord_label is not None:
            self._vision_coord_label.setText("")
            self._vision_coord_label.hide()
        self._vision_coord_label_2 = getattr(self, "vision_coord_label_2", None)
        if self._vision_coord_label_2 is not None:
            self._vision_coord_label_2.setText("")
            self._vision_coord_label_2.hide()

        self._robot_state_label = getattr(self, "robot_state_label", None)
        if self._robot_state_label is not None:
            self._robot_state_label.setText("초기화 중")

        self._setup_robot_state_table()
        self._setup_position_tables()
        self._hide_robot_control_title_label()
        self._setup_cycle_time_labels()
        self._update_cycle_time_labels()

        self.yolo_view.setMouseTracking(True)
        self.yolo_view.installEventFilter(self)
        self.yolo_view_2 = getattr(self, "yolo_view_2", None)
        if self.yolo_view_2 is not None:
            self.yolo_view_2.setMouseTracking(True)
            self.yolo_view_2.installEventFilter(self)

        self.statusBar().hide()
        if self._auto_start_backend:
            self._start_backend_async()

    def _reserve_top_status_space(self):
        if self._status_row_ready:
            return
        self._layout_main_frames()
        self._status_row_ready = True

    def _layout_main_frames(self):
        if UI_USE_DESIGN_GEOMETRY:
            return
        frame_l = getattr(self, "frame", None)
        frame_m = getattr(self, "frame_4", None)
        frame_r = getattr(self, "frame_2", None)
        frame_log = getattr(self, "frame_3", None)
        term = getattr(self, "terminal", None)
        if frame_l is None or frame_r is None or frame_log is None:
            return

        target_y = TOP_STATUS_BAR_HEIGHT + TOP_STATUS_GAP
        dx_l = frame_l.x()
        dx_m = frame_m.x() if frame_m is not None else None
        dx_r = frame_r.x()
        w_l = frame_l.width()
        w_m = frame_m.width() if frame_m is not None else None
        w_r = frame_r.width()

        total_h = max(640, self.centralwidget.height())
        upper_h = int(total_h * 0.58)
        upper_h = max(470, upper_h)
        upper_h = min(upper_h, total_h - 240)
        h_lr = upper_h
        frame_l.setGeometry(dx_l, target_y, w_l, h_lr)
        if frame_m is not None and dx_m is not None and w_m is not None:
            frame_m.setGeometry(dx_m, target_y, w_m, h_lr)
        frame_r.setGeometry(dx_r, target_y, w_r, h_lr)
        self._layout_vision_widgets()

        log_y = target_y + h_lr + TOP_STATUS_GAP + LOG_AREA_SHIFT_Y
        log_h = max(260, self.centralwidget.height() - log_y - 8)
        frame_log.setGeometry(frame_log.x(), log_y, frame_log.width(), log_h)
        if hasattr(self, "_log_clear_button") and self._log_clear_button is not None:
            btn_w = 96
            btn_h = 24
            self._log_clear_button.setGeometry(max(10, frame_log.width() - btn_w - 12), 8, btn_w, btn_h)
        if term is not None:
            term.setGeometry(10, 40, frame_log.width() - 30, max(120, log_h - 50))

    def _layout_vision_widgets(self):
        frame_l = getattr(self, "frame", None)
        view = getattr(self, "yolo_view", None)
        if frame_l is None or view is None:
            return
        top_y = 72
        side = 10
        bottom = 14
        w = max(220, frame_l.width() - (side * 2))
        calib_box = getattr(self, "calibration_group_box", None)
        if calib_box is not None:
            h = max(180, calib_box.y() - top_y - 8)
        else:
            h = max(180, frame_l.height() - top_y - bottom)
        view.setGeometry(side, top_y, w, h)
        frame_m = getattr(self, "frame_4", None)
        view2 = getattr(self, "yolo_view_2", None)
        if frame_m is not None and view2 is not None:
            w2 = max(220, frame_m.width() - (side * 2))
            calib_box2 = getattr(self, "calibration_group_box_2", None)
            if calib_box2 is not None:
                h2 = max(180, calib_box2.y() - top_y - 8)
            else:
                h2 = max(180, frame_m.height() - top_y - bottom)
            view2.setGeometry(side, top_y, w2, h2)

    def _setup_log_controls(self):
        frame_log = getattr(self, "frame_3", None)
        if frame_log is None:
            self._log_clear_button = None
            return
        self._log_clear_button = getattr(self, "log_clear_button", None)
        if self._log_clear_button is None:
            for btn in frame_log.findChildren(QPushButton):
                if "로그" in btn.text() and "클리어" in btn.text():
                    self._log_clear_button = btn
                    break
        if self._log_clear_button is not None:
            self._log_clear_button.clicked.connect(self._clear_log_terminal)

    def _clear_log_terminal(self):
        self._log_buffer.clear()
        self.terminal.clear()

    def append_log(self, text):
        try:
            msg = str(text)
        except Exception:
            return
        if not msg:
            return
        if not hasattr(self, "_log_buffer") or self._log_buffer is None:
            return
        self._log_buffer.append(msg.replace("\r\n", "\n").replace("\r", "\n"))

    def _flush_log_buffer(self):
        if not hasattr(self, "_log_buffer") or not self._log_buffer:
            return
        chunk = "".join(self._log_buffer)
        self._log_buffer.clear()
        term = getattr(self, "terminal", None)
        if term is None:
            return
        try:
            from PyQt5.QtGui import QTextCursor

            cursor = term.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(chunk)
            term.setTextCursor(cursor)
        except Exception:
            try:
                if hasattr(term, "insertPlainText"):
                    term.insertPlainText(chunk)
                else:
                    existing = term.toPlainText() if hasattr(term, "toPlainText") else ""
                    term.setPlainText(existing + chunk)
            except Exception:
                return
        try:
            sb = term.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())
        except Exception:
            pass

    def _fmt_ui_float(self, value, digits=2):
        try:
            v = float(value)
        except Exception:
            return "-"
        if not np.isfinite(v):
            return "-"
        return f"{v:.{int(digits)}f}"

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        try:
            text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        except Exception:
            text = f"{exc_type}: {exc_value}"
        try:
            sys.__stderr__.write(text)
            sys.__stderr__.flush()
        except Exception:
            pass
        try:
            self.append_log(f"{text}\n")
        except Exception:
            pass

    def _handle_thread_exception(self, args):
        if args is None:
            return
        self._handle_exception(
            getattr(args, "exc_type", Exception),
            getattr(args, "exc_value", Exception("thread exception")),
            getattr(args, "exc_traceback", None),
        )

    def _setup_top_status_row(self):
        self._top_status_panel = getattr(self, "top_status_panel", None)
        self._top_status_mid_line = getattr(self, "top_status_mid_line", None)
        self._top_status_widgets = {}
        if self._top_status_panel is not None:
            self._top_status_panel.setFrameShape(QFrame.NoFrame)
            self._top_status_panel.setStyleSheet(
                "QFrame#top_status_panel { border: none; background: #f8fafc; }"
            )
        if self._top_status_mid_line is not None:
            self._top_status_mid_line.hide()

        defs = [
            ("vision", "비전1", "top_status_vision_box", "top_status_vision_dot", "top_status_vision_text"),
            ("vision2", "비전2", "top_status_vision_box_2", "top_status_vision_dot_2", "top_status_vision_text_2"),
            ("robot", "로봇", "top_status_robot_box", "top_status_robot_dot", "top_status_robot_text"),
        ]
        for key, title, box_name, dot_name, text_name in defs:
            box = getattr(self, box_name, None)
            dot = getattr(self, dot_name, None)
            text = getattr(self, text_name, None)
            # New UI may remove *box and keep only dot/text.
            if dot is None or text is None:
                continue
            if box is not None:
                box.setFrameShape(QFrame.NoFrame)
                box.setStyleSheet("background: #ffffff; border: 1px solid #cfd8dc; border-radius: 3px;")
            dot.setAlignment(Qt.AlignCenter)
            dot.setText("●")
            text.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self._top_status_widgets[key] = (title, box, dot, text)
        ui_toggles = {
            "vision": getattr(self, "top_status_vision_toggle", None),
            "vision2": getattr(self, "top_status_vision2_toggle", None),
            "robot": getattr(self, "top_status_robot_toggle", None),
        }
        for key, toggle in ui_toggles.items():
            if toggle is None:
                continue
            self._top_status_toggles[key] = toggle
            toggle.blockSignals(True)
            toggle.setChecked(bool(self._top_status_enabled.get(key, True)))
            toggle.setText("ON" if bool(self._top_status_enabled.get(key, True)) else "OFF")
            toggle.blockSignals(False)
            try:
                toggle.toggled.disconnect()
            except Exception:
                pass
            toggle.toggled.connect(lambda checked, src=key: self._on_top_status_toggle_changed(src, checked))

        self._reposition_top_status_row()
        self._set_top_status("vision", "확인중", "warning")
        self._set_top_status("vision2", "확인중", "warning")
        self._set_top_status("robot", "확인중", "warning")

    def _on_top_status_toggle_changed(self, key: str, checked: bool):
        k = str(key)
        prev_on = bool(self._top_status_enabled.get(k, True))
        on = bool(checked)
        self._top_status_enabled[k] = on
        toggle = self._top_status_toggles.get(k)
        if toggle is not None:
            toggle.setText("ON" if on else "OFF")
        if k == "vision":
            self._vision_state_text = "비활성화" if (not on) else "확인중"
            if not on:
                if bool(getattr(self, "_calibration_mode_enabled_1", False)):
                    sw = getattr(self, "_calibration_mode_switch", None)
                    if sw is not None:
                        sw.setChecked(False)
                    else:
                        self._calibration_mode_enabled_1 = False
                        self._calibration_mode_enabled = bool(self._calibration_mode_enabled_2)
                        self._update_calibration_mode_ui()
                        self._save_vision_serial_settings()
                self._vision_cycle_ms = None
                if YOLO_EXTERNAL_NODE:
                    self._teardown_external_vision_bridge_panel(1)
                self._stop_calibration_process(1)
                self._clear_vision_view_data(panel_index=1)
                if YOLO_EXTERNAL_NODE and (not bool(self._top_status_enabled.get("vision2", True))):
                    self._stop_external_vision_process()
            else:
                self._vision_cycle_ms = None
            self._set_vision_panel_controls_enabled(1, on)
            self._sync_vision_render_timers()
        elif k == "vision2":
            self._vision_state_text_2 = "비활성화" if (not on) else "확인중"
            if not on:
                if bool(getattr(self, "_calibration_mode_enabled_2", False)):
                    sw = getattr(self, "_calibration_mode_switch_2", None)
                    if sw is not None:
                        sw.setChecked(False)
                    else:
                        self._calibration_mode_enabled_2 = False
                        self._calibration_mode_enabled = bool(self._calibration_mode_enabled_1)
                        self._update_calibration_mode_ui()
                        self._save_vision_serial_settings()
                self._vision_cycle_ms_2 = None
                if YOLO_EXTERNAL_NODE:
                    self._teardown_external_vision_bridge_panel(2)
                self._stop_calibration_process(2)
                self._clear_vision_view_data(panel_index=2)
                if YOLO_EXTERNAL_NODE and (not bool(self._top_status_enabled.get("vision", True))):
                    self._stop_external_vision_process()
            else:
                self._vision_cycle_ms_2 = None
            self._set_vision_panel_controls_enabled(2, on)
            self._sync_vision_render_timers()
        elif k == "robot" and (not on):
            self._robot_cycle_ms = None
            if self.backend is not None and hasattr(self.backend, "stop_robot_state_subscriptions"):
                try:
                    ok, msg = self.backend.stop_robot_state_subscriptions()
                    self.append_log(f"[로봇] {msg}\n")
                except Exception as e:
                    self.append_log(f"[로봇] 구독 종료 실패: {e}\n")
            self._clear_robot_data_view()
        elif k == "robot" and on and (not prev_on):
            if self.backend is not None and hasattr(self.backend, "start_robot_state_subscriptions"):
                try:
                    ok, msg = self.backend.start_robot_state_subscriptions()
                    self.append_log(f"[로봇] {msg}\n")
                except Exception as e:
                    self.append_log(f"[로봇] 구독 재시작 실패: {e}\n")
            self._robot_cycle_ms = None
            self._robot_prev_state_seen_at = None
            self._last_positions_seen_at = None
            self._set_robot_controls_enabled(self.backend is not None and self.backend.is_ready())
        self.append_log(f"[상태토글] {k} {'활성화' if on else '비활성화'}\n")
        if on and (not prev_on) and k in ("vision", "vision2"):
            self._restart_vision_after_toggle_enable(k)
        self._save_vision_serial_settings()
        self._refresh_robot_status()
        self._refresh_vision_status()

    def _restart_vision_after_toggle_enable(self, key: str):
        if not UI_ENABLE_VISION:
            return
        now = time.monotonic()
        self._mode_switch_grace_until = now + MODE_SWITCH_GRACE_SEC
        if key == "vision":
            self._vision_state_text = "시작 중"
            self._last_vision_frame_at = now
        else:
            self._vision_state_text_2 = "시작 중"
            self._last_vision_frame_at_2 = now
        self.append_log(f"[비전] {key} 활성화: 비전 연결 재초기화 시작\n")
        if self.backend is None:
            return
        if YOLO_EXTERNAL_NODE:
            try:
                calib_on_any = bool(self._calibration_mode_enabled_1 or self._calibration_mode_enabled_2)
                if YOLO_AUTO_LAUNCH_NODE and (not calib_on_any):
                    self._ensure_external_vision_process()
                self._sync_calibration_processes()
                panel = 1 if key == "vision" else 2
                self._setup_external_vision_bridge_panel(panel)
            except Exception as e:
                self.append_log(f"[비전] 재초기화 실패: {e}\n")
        else:
            # 내부 모드에서도 공용 카메라 재시작(양쪽 영향)은 하지 않는다.
            pass

    def _set_vision_panel_controls_enabled(self, panel_index: int, enabled: bool):
        panel = 2 if int(panel_index) == 2 else 1
        on = bool(enabled)
        if panel == 2:
            targets = [
                getattr(self, "yolo_view_2", None),
                getattr(self, "_vision_rotate_left_button_2", None),
                getattr(self, "_vision_rotate_zero_button_2", None),
                getattr(self, "_vision_rotate_right_button_2", None),
                getattr(self, "_vision_serial_change_button_2", None),
                getattr(self, "_calibration_mode_switch_2", None),
                getattr(self, "_calibration_transform_button_2", None),
                getattr(self, "_calibration_load_button_2", None),
                getattr(self, "calibration_group_box_2", None),
            ]
        else:
            targets = [
                getattr(self, "yolo_view", None),
                getattr(self, "_vision_rotate_left_button", None),
                getattr(self, "_vision_rotate_zero_button", None),
                getattr(self, "_vision_rotate_right_button", None),
                getattr(self, "_vision_serial_change_button", None),
                getattr(self, "_calibration_mode_switch", None),
                getattr(self, "_calibration_transform_button", None),
                getattr(self, "_calibration_load_button", None),
                getattr(self, "calibration_group_box", None),
            ]
        for w in targets:
            if w is not None:
                w.setEnabled(on)

    def _sync_vision_render_timers(self):
        if hasattr(self, "_vision_render_timer_1") and self._vision_render_timer_1 is not None:
            want_1 = bool(self._top_status_enabled.get("vision", True))
            if want_1 and (not self._vision_render_timer_1.isActive()):
                self._vision_render_timer_1.start(16)
            elif (not want_1) and self._vision_render_timer_1.isActive():
                self._vision_render_timer_1.stop()
        if hasattr(self, "_vision_render_timer_2") and self._vision_render_timer_2 is not None:
            want_2 = bool(self._top_status_enabled.get("vision2", True))
            if want_2 and (not self._vision_render_timer_2.isActive()):
                self._vision_render_timer_2.start(16)
            elif (not want_2) and self._vision_render_timer_2.isActive():
                self._vision_render_timer_2.stop()

    def _clear_vision_view_data(self, panel_index: int):
        panel = 2 if int(panel_index) == 2 else 1
        if panel == 2:
            with self._vision_frame_lock_2:
                self._pending_yolo_qimage_2 = None
                self._pending_vision_token_2 = 0
                self._vision_frame_pending_2 = False
            self._last_yolo_qimage_2 = None
            self._vision_prev_update_at_2 = None
            self._vision_cycle_ms_2 = None
            self._vision_decode_ms_2 = None
            self._vision_render_delay_ms_2 = None
            self._vision_render_interval_ms_2 = None
            self._pending_vision_enqueued_at_2 = None
            self._vision_render_prev_at_2 = None
            self._vision_depth_image_2 = None
            self._vision_depth_shape_2 = None
            self._vision_depth_encoding_2 = None
            self._camera_intrinsics_2 = None
        else:
            with self._vision_frame_lock_1:
                self._pending_yolo_qimage = None
                self._pending_vision_token_1 = 0
                self._vision_frame_pending = False
            self._last_yolo_qimage = None
            self._vision_prev_update_at = None
            self._vision_cycle_ms = None
            self._vision_decode_ms = None
            self._vision_render_delay_ms = None
            self._vision_render_interval_ms = None
            self._pending_vision_enqueued_at_1 = None
            self._vision_render_prev_at = None
            self._vision_depth_image = None
            self._vision_depth_shape = None
            self._vision_depth_encoding = None
            self._camera_intrinsics_1 = None
            self._camera_intrinsics = None
        if panel == 2:
            self._last_vision_frame_at_2 = None
            view = getattr(self, "yolo_view_2", None)
        else:
            self._last_vision_frame_at = None
            view = getattr(self, "yolo_view", None)
        if view is not None:
            view.clear()
            view.setPixmap(QPixmap())
            view.setAlignment(Qt.AlignCenter)
            view.setStyleSheet("background-color: #000000; color: #f3f3f3;")
            view.setText("이미지 데이터 없음")

    def _clear_vision_depth_cache(self, panel_index: int):
        panel = 2 if int(panel_index) == 2 else 1
        if panel == 2:
            self._vision_depth_image_2 = None
            self._vision_depth_shape_2 = None
            self._vision_depth_encoding_2 = None
        else:
            self._vision_depth_image = None
            self._vision_depth_shape = None
            self._vision_depth_encoding = None

    def _clear_calibration_panel_state(self, panel_index: int):
        panel = 2 if int(panel_index) == 2 else 1
        if panel == 2:
            self._calib_last_points_uvz_mm_2 = None
            self._calib_last_points_at_2 = None
            self._calib_last_detect_run_at_2 = 0.0
            self._calib_last_grid_pts_2 = None
            self._calib_last_grid_status_2 = None
            self._calib_grid_camera_xyz_2 = None
            self._calib_last_reason_2 = "대기"
            self._calib_full_ready_2 = False
            self._calib_full_reason_2 = "대기"
        else:
            self._calib_last_points_uvz_mm_1 = None
            self._calib_last_points_at_1 = None
            self._calib_last_detect_run_at_1 = 0.0
            self._calib_last_grid_pts_1 = None
            self._calib_last_grid_status_1 = None
            self._calib_grid_camera_xyz_1 = None
            self._calib_last_reason_1 = "대기"
            self._calib_full_ready_1 = False
            self._calib_full_reason_1 = "대기"
            self._calib_last_points_uvz_mm = None
            self._calib_last_points_at = None
            self._calib_last_reason = "대기"
        self.calibration_ui_refresh_requested.emit()

    def _advance_vision_stream_token(self, panel_index: int):
        panel = 2 if int(panel_index) == 2 else 1
        if panel == 2:
            self._vision_stream_token_2 += 1
            return self._vision_stream_token_2
        self._vision_stream_token_1 += 1
        return self._vision_stream_token_1

    def _camera_status_for_panel(self, panel_index: int, now: float, enabled: bool):
        panel = 2 if int(panel_index) == 2 else 1
        if not enabled:
            return "비활성화"
        depth_sub = getattr(self, "_vision_depth_sub_2", None) if panel == 2 else getattr(self, "_vision_depth_sub", None)
        last_camera_frame_at = (
            getattr(self, "_last_camera_frame_at_2", None) if panel == 2 else getattr(self, "_last_camera_frame_at_1", None)
        )
        if depth_sub is None:
            return "전환중" if now < float(getattr(self, "_mode_switch_grace_until", 0.0)) else "끊김"
        if last_camera_frame_at is None:
            return "확인중" if now < float(getattr(self, "_mode_switch_grace_until", 0.0)) else "끊김"
        if (now - float(last_camera_frame_at)) > POSITION_STALE_SEC:
            return "지연"
        return "정상 수신 중"

    def _clear_robot_data_view(self):
        self._set_signal_state_label(self._robot_state_label, "비활성화", "warning", False)
        for i in range(6):
            self._set_table_value(i, "-", "", flash=False)
        self._clear_position_views()

    def _clear_position_views(self):
        for table in (getattr(self, "_joint_table", None), getattr(self, "_cart_table", None)):
            if table is None:
                continue
            for row in range(table.rowCount()):
                item = table.item(row, 1)
                if item is None:
                    item = QTableWidgetItem("-")
                    table.setItem(row, 1, item)
                item.setText("-")
                item.setForeground(QBrush(self.palette().text().color()))
                font = item.font()
                font.setBold(False)
                item.setFont(font)
        self._pos_value_cache.clear()
        self._pos_flash_until.clear()
        self._last_positions_seen_at = None
        self._set_current_tool_text("-")

    def _reposition_top_status_row(self):
        if not self._top_status_widgets:
            return
        # If panel/box widgets were removed in .ui, preserve Designer geometry.
        if self._top_status_panel is None:
            return
        frame_l = getattr(self, "frame", None)
        frame_r = getattr(self, "frame_2", None)
        use_frame_alignment = frame_l is not None and frame_r is not None

        left = 10
        y = 4
        if use_frame_alignment:
            panel_left = min(frame_l.x(), frame_r.x())
            panel_right = max(frame_l.x() + frame_l.width(), frame_r.x() + frame_r.width())
            left = panel_left
            total_w = max(120, panel_right - panel_left)
        else:
            total_w = max(400, self.centralwidget.width() - 20)
        panel_h = max(28, TOP_STATUS_BAR_HEIGHT - 6)
        if self._top_status_panel is not None:
            self._top_status_panel.setGeometry(left, y, total_w, panel_h)
        if use_frame_alignment and "vision" in self._top_status_widgets and "robot" in self._top_status_widgets:
            v = self._top_status_widgets.get("vision")
            v2 = self._top_status_widgets.get("vision2")
            r = self._top_status_widgets.get("robot")
            if v is None or r is None:
                return
            v_title, v_box, v_dot, v_text = v
            r_title, r_box, r_dot, r_text = r
            if v_box is None or r_box is None:
                return

            v_x = frame_l.x() - left
            frame_m = getattr(self, "frame_4", None)
            m_x = frame_m.x() - left if frame_m is not None else None
            m_w = frame_m.width() if frame_m is not None else None
            r_x = frame_r.x() - left
            v_w = frame_l.width()
            r_w = frame_r.width()

            v_box.setGeometry(v_x, 1, max(90, v_w), panel_h - 2)
            v_dot.setGeometry(6, 1, 16, v_box.height() - 2)
            v_text.setGeometry(24, 1, max(64, v_box.width() - 28), v_box.height() - 2)

            if v2 is not None:
                _t2, v2_box, v2_dot, v2_text = v2
                if v2_box is not None and m_x is not None and m_w is not None:
                    v2_box.setGeometry(m_x, 1, max(90, m_w), panel_h - 2)
                    v2_dot.setGeometry(6, 1, 16, v2_box.height() - 2)
                    v2_text.setGeometry(24, 1, max(64, v2_box.width() - 28), v2_box.height() - 2)

            r_box.setGeometry(r_x, 1, max(90, r_w), panel_h - 2)
            r_dot.setGeometry(6, 1, 16, r_box.height() - 2)
            r_text.setGeometry(24, 1, max(64, r_box.width() - 28), r_box.height() - 2)
            return

        count = max(1, len(self._top_status_widgets))
        side_margin = 4
        box_gap = 8
        usable_w = max(120, total_w - (2 * side_margin) - ((count - 1) * box_gap))
        slot_w = int(usable_w / count)
        x = side_margin
        for idx, key in enumerate(self._top_status_widgets.keys()):
            title, box, dot, text = self._top_status_widgets[key]
            if box is None:
                continue
            if idx == (count - 1):
                box_w = max(90, total_w - side_margin - x)
            else:
                box_w = max(90, slot_w)
            box.setGeometry(x, 1, box_w, panel_h - 2)
            dot.setGeometry(6, 1, 16, box.height() - 2)
            text.setGeometry(24, 1, max(64, box.width() - 28), box.height() - 2)
            x += box_w + box_gap

    def _set_top_status(self, key: str, state_text: str, severity: str):
        self._top_status_state_cache[str(key)] = (str(state_text), str(severity))
        item = self._top_status_widgets.get(key)
        if item is None:
            return
        title, box, dot, text = item
        # 1-second fade cycle.
        phase = time.monotonic() % 1.0  # 0..1 per second
        # 0 -> 1 -> 0 within 1 second
        fade = 0.5 * (1.0 + np.sin((2.0 * np.pi * phase) - (np.pi / 2.0)))
        state_text_norm = str(state_text or "").strip()
        should_blink = (str(key) in {"vision", "vision2"}) or (state_text_norm in {
            "비활성화",
            "끊김",
            "대기",
            "초기화",
            "초기화중",
            "지연",
            "중지",
            "시작 중",
            "확인중",
            "연결",
            "정상 수신 중",
        })
        def _hex_to_rgb(h: str):
            h = h.lstrip("#")
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

        def _rgb_to_hex(rgb):
            r, g, b = rgb
            return f"#{int(max(0, min(255, r))):02x}{int(max(0, min(255, g))):02x}{int(max(0, min(255, b))):02x}"

        def _mix(c1: str, c2: str, t: float):
            r1, g1, b1 = _hex_to_rgb(c1)
            r2, g2, b2 = _hex_to_rgb(c2)
            return _rgb_to_hex((r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t))

        if severity == "normal":
            bright, dim = STATE_COLOR_NORMAL, "#ffffff"
            bg_bright, bg_dim = "#eef9f1", "#fbfcfb"
        elif severity == "error":
            bright, dim = STATE_COLOR_ERROR, "#ffffff"
            bg_bright, bg_dim = "#fdeeee", "#fffafb"
        else:
            bright, dim = STATE_COLOR_WARNING, "#ffffff"
            bg_bright, bg_dim = "#fff5e8", "#fffdf9"

        if should_blink:
            color = _mix(dim, bright, fade)
            bg = _mix(bg_dim, bg_bright, fade)
            alpha = int(max(0, min(255, round(255.0 * fade))))
        else:
            color = bright
            bg = bg_bright
            alpha = 255
        r, g, b = _hex_to_rgb(color)
        dot.setStyleSheet(f"color: rgba({r}, {g}, {b}, {alpha}); font-size: 20pt;")
        text.setStyleSheet("color: #202020; font-weight: 700; font-size: 14pt;")
        if box is not None:
            box.setStyleSheet(f"background: {bg}; border: 1px solid #cfd8dc; border-radius: 3px;")
        text.setText(f"{title}: {state_text}")

    def _tick_top_status_animation(self):
        if not self._top_status_state_cache:
            return
        for key, payload in list(self._top_status_state_cache.items()):
            if not isinstance(payload, tuple) or len(payload) != 2:
                continue
            state_text, severity = payload
            self._set_top_status(key, state_text, severity)

    def _scan_source_links(self):
        now = time.monotonic()
        if (now - self._source_scan_at) < self._source_scan_interval_sec:
            return
        self._source_scan_at = now
        node = getattr(self.backend, "robot_controller", None) if self.backend is not None else None
        if node is None:
            self._realsense_connected = False
            self._yolo_topic_connected = False
            return
        try:
            topic_map = dict(node.get_topic_names_and_types())
        except Exception:
            return

        rs_color = any(t in topic_map for t in ("/camera/camera/color/image_raw", "/camera/color/image_raw"))
        rs_depth = any(
            t in topic_map for t in ("/camera/camera/aligned_depth_to_color/image_raw", "/camera/aligned_depth_to_color/image_raw")
        )
        rs_info = any(t in topic_map for t in ("/camera/camera/color/camera_info", "/camera/color/camera_info"))
        self._realsense_connected = rs_color and rs_depth and rs_info

        types = topic_map.get(YOLO_EXTERNAL_TOPIC, [])
        self._yolo_topic_connected = "sensor_msgs/msg/Image" in types

    def _set_robot_controls_enabled(self, enabled: bool):
        enabled = bool(enabled)
        calib_seq_running = bool(getattr(self, "_calibration_sequence_running", False))
        calib_on = bool(
            getattr(self, "_calibration_mode_enabled_1", False)
            or getattr(self, "_calibration_mode_enabled_2", False)
        )
        calib_ready_1 = bool(self._is_calibration_points_ready(panel_index=1))
        calib_ready_2 = bool(self._is_calibration_points_ready(panel_index=2))
        calib_ready = bool(calib_ready_1 or calib_ready_2)
        mode_value = None
        mode_seen_at = None
        if self.backend is not None and hasattr(self.backend, "get_robot_mode_snapshot"):
            try:
                mode_value, mode_seen_at = self.backend.get_robot_mode_snapshot()
            except Exception:
                mode_value, mode_seen_at = None, None
        mode_key = None
        if mode_value is not None and mode_seen_at is not None:
            try:
                mode_key = int(mode_value)
            except Exception:
                mode_key = None
        state_key = (enabled, calib_on, calib_ready, mode_key, calib_seq_running)
        if self._robot_controls_enabled_cache is not None and self._robot_controls_enabled_cache == state_key:
            return
        self._robot_controls_enabled_cache = state_key
        # 자동 캘리브레이션 시퀀스 진행 중에는 로봇 조작 버튼을 잠근다.
        normal_enabled = enabled and (not calib_seq_running)
        gripper_enabled = normal_enabled and (mode_key == 1)
        tool_change_enabled = normal_enabled and (mode_key == 0)

        self.pushButton.setEnabled(normal_enabled)
        if hasattr(self, "_print_pos_button") and self._print_pos_button is not None:
            self._print_pos_button.setEnabled(normal_enabled)
        if hasattr(self, "_reset_button") and self._reset_button is not None:
            self._reset_button.setEnabled(normal_enabled)
        if hasattr(self, "_home_button") and self._home_button is not None:
            self._home_button.setEnabled(normal_enabled)
        if hasattr(self, "_home_save_button") and self._home_save_button is not None:
            self._home_save_button.setEnabled(normal_enabled)
        if hasattr(self, "_robot_mode_button") and self._robot_mode_button is not None:
            self._robot_mode_button.setEnabled(normal_enabled)
        if hasattr(self, "_tool_change_button") and self._tool_change_button is not None:
            self._tool_change_button.setEnabled(tool_change_enabled)
        if hasattr(self, "_gripper_move_button") and self._gripper_move_button is not None:
            self._gripper_move_button.setEnabled(gripper_enabled)
        if hasattr(self, "_gripper_stroke_input") and self._gripper_stroke_input is not None:
            self._gripper_stroke_input.setEnabled(gripper_enabled)
        if hasattr(self, "_vision_move_button") and self._vision_move_button is not None:
            self._vision_move_button.setEnabled(normal_enabled)
        if hasattr(self, "_vision_z_margin_input") and self._vision_z_margin_input is not None:
            self._vision_z_margin_input.setEnabled(normal_enabled)
        if hasattr(self, "_vision_dialog_toggle_button") and self._vision_dialog_toggle_button is not None:
            self._vision_dialog_toggle_button.setEnabled(normal_enabled)
        if hasattr(self, "_vision_dialog_toggle_switch") and self._vision_dialog_toggle_switch is not None:
            # 클릭 확인 스위치는 백엔드 준비 전에도 사용자가 미리 바꿀 수 있게 항상 활성화한다.
            self._vision_dialog_toggle_switch.setEnabled(True)
        if hasattr(self, "_calibration_mode_switch") and self._calibration_mode_switch is not None:
            self._calibration_mode_switch.setEnabled(True)
        if hasattr(self, "_calibration_mode_switch_2") and self._calibration_mode_switch_2 is not None:
            self._calibration_mode_switch_2.setEnabled(True)
        if hasattr(self, "_calibration_transform_button") and self._calibration_transform_button is not None:
            self._calibration_transform_button.setEnabled(
                bool((not calib_seq_running) and self._calibration_mode_enabled_1 and calib_ready_1)
            )
        if hasattr(self, "_calibration_transform_button_2") and self._calibration_transform_button_2 is not None:
            self._calibration_transform_button_2.setEnabled(
                bool((not calib_seq_running) and self._calibration_mode_enabled_2 and calib_ready_2)
            )
        if hasattr(self, "_calibration_load_button") and self._calibration_load_button is not None:
            self._calibration_load_button.setEnabled(bool(self._calibration_mode_enabled_1))
        if hasattr(self, "_calibration_load_button_2") and self._calibration_load_button_2 is not None:
            self._calibration_load_button_2.setEnabled(bool(self._calibration_mode_enabled_2))
        self._set_vision_panel_controls_enabled(1, bool(self._top_status_enabled.get("vision", True)))
        self._set_vision_panel_controls_enabled(2, bool(self._top_status_enabled.get("vision2", True)))

    def _start_backend_async(self):
        if self.backend is not None and hasattr(self.backend, "is_ready") and self.backend.is_ready():
            self._on_backend_ready(self.backend)
            return
        if self._backend_thread is not None:
            return

        self.append_log("[초기화] 백엔드 시작 중...\n")
        self._backend_thread = QThread(self)
        self._backend_worker = BackendStartupWorker(use_real_gripper=STARTUP_USE_REAL_GRIPPER)
        self._backend_worker.moveToThread(self._backend_thread)

        self._backend_thread.started.connect(self._backend_worker.run)
        self._backend_worker.progress.connect(self._on_backend_progress)
        self._backend_worker.finished.connect(self._on_backend_ready)
        self._backend_worker.failed.connect(self._on_backend_failed)

        self._backend_worker.finished.connect(self._backend_thread.quit)
        self._backend_worker.failed.connect(lambda *_: self._backend_thread.quit())
        self._backend_thread.finished.connect(self._backend_thread.deleteLater)
        self._backend_thread.finished.connect(self._backend_worker.deleteLater)
        self._backend_thread.finished.connect(self._clear_backend_worker_refs)

        self._backend_thread.start()

    def _clear_backend_worker_refs(self):
        self._backend_thread = None
        self._backend_worker = None

    def _clear_reset_worker_refs(self):
        self._reset_thread = None
        self._reset_worker = None

    def _start_reset_async(self):
        if self.backend is None:
            self.append_log("[리셋] 백엔드 초기화 중입니다.\n")
            return
        if self._reset_thread is not None:
            self.append_log("[리셋] 이미 진행 중입니다.\n")
            return
        if not hasattr(self.backend, "reset_robot_state"):
            self.append_log("[리셋] 백엔드 리셋 기능을 지원하지 않습니다.\n")
            return

        self._set_robot_controls_enabled(False)
        self.append_log("[리셋] 요청 전송, 응답 대기 중...\n")

        self._reset_thread = QThread(self)
        self._reset_worker = BackendResetWorker(self.backend)
        self._reset_worker.moveToThread(self._reset_thread)

        self._reset_thread.started.connect(self._reset_worker.run)
        self._reset_worker.finished.connect(self._on_reset_finished)
        self._reset_worker.failed.connect(self._on_reset_failed)

        self._reset_worker.finished.connect(lambda *_: self._reset_thread.quit())
        self._reset_worker.failed.connect(lambda *_: self._reset_thread.quit())
        self._reset_thread.finished.connect(self._reset_thread.deleteLater)
        self._reset_thread.finished.connect(self._reset_worker.deleteLater)
        self._reset_thread.finished.connect(self._clear_reset_worker_refs)
        self._reset_thread.start()

    def _on_reset_finished(self, ok, msg):
        self.append_log(f"[리셋] {msg}\n")
        self._set_robot_controls_enabled(self.backend is not None and self.backend.is_ready())

    def _on_reset_failed(self, err, tb):
        self.append_log(f"[리셋] 실패: {err}\n")
        self.append_log(tb + "\n")
        self._set_robot_controls_enabled(self.backend is not None and self.backend.is_ready())

    def _on_backend_progress(self, percent, message, elapsed_sec):
        self.append_log(f"[초기화] {int(percent)}% {message} ({elapsed_sec:.1f}s)\n")

    def _on_backend_ready(self, backend):
        self.backend = backend
        self.append_log("[초기화] 백엔드 초기화 완료\n")
        self._refresh_current_tool_label_once(log_fail=False)
        if not bool(self._top_status_enabled.get("robot", True)):
            if hasattr(self.backend, "stop_robot_state_subscriptions"):
                try:
                    ok, msg = self.backend.stop_robot_state_subscriptions()
                    self.append_log(f"[로봇] {msg}\n")
                except Exception as e:
                    self.append_log(f"[로봇] 구독 종료 실패: {e}\n")
        self._set_robot_controls_enabled(bool(self._top_status_enabled.get("robot", True)))
        self._setup_rosout_bridge()
        self._start_yolo_camera()

    def _apply_backend_failed_state(self, err, tb, show_popup=True):
        self.append_log(f"[초기화] 실패: {err}\n")
        if tb:
            self.append_log(f"{tb}\n")
        self._set_robot_controls_enabled(False)
        self._vision_state_text = "오류"
        self._vision_state_text_2 = "오류"
        if hasattr(self, "yolo_view") and self.yolo_view is not None:
            self.yolo_view.setText("초기화 실패")
        if hasattr(self, "yolo_view_2") and self.yolo_view_2 is not None:
            self.yolo_view_2.setText("초기화 실패")
        if show_popup:
            QMessageBox.warning(self, "초기화 실패", f"백엔드 시작 실패\n\n{err}")

    def _on_backend_failed(self, err, tb):
        self._apply_backend_failed_state(err, tb, show_popup=True)

    def _refresh_status(self):
        # Backward-compat wrapper
        self._refresh_robot_status()
        self._refresh_vision_status()

    def _refresh_robot_status(self):
        if not bool(self._top_status_enabled.get("robot", True)):
            self._robot_cycle_ms = None
            self._set_robot_controls_enabled(False)
            self._clear_robot_data_view()
            self._set_top_status("robot", "비활성화", "warning")
            self._update_cycle_time_labels()
            return
        if self.backend is None:
            self._set_robot_controls_enabled(False)
            self._set_signal_state_label(self._robot_state_label, "초기화 중", "warning", False)
            self._set_table_value(0, "초기화", "warning", flash=False)
            self._set_table_value(1, "-", "", flash=False)
            self._set_table_value(2, "-", "", flash=False)
            self._set_table_value(3, "-", "", flash=False)
            self._set_table_value(4, "-", "", flash=False)
            self._set_table_value(5, "-", "", flash=False)
            self._set_top_status("robot", "초기화중", "warning")
            self._update_cycle_time_labels()
            return

        mode_text = self._detect_robot_mode_text()
        robot_mode_value = None
        robot_mode_seen_at = None
        if hasattr(self.backend, "get_robot_mode_snapshot"):
            robot_mode_value, robot_mode_seen_at = self.backend.get_robot_mode_snapshot()
        control_mode_value = None
        control_mode_seen_at = None
        if hasattr(self.backend, "get_control_mode_snapshot"):
            control_mode_value, control_mode_seen_at = self.backend.get_control_mode_snapshot()
        state_code = None
        state_name = ""
        state_seen_at = None
        if hasattr(self.backend, "get_robot_state_snapshot"):
            state_code, state_name, state_seen_at = self.backend.get_robot_state_snapshot()

        now = time.monotonic()
        if state_seen_at is not None:
            if self._robot_prev_state_seen_at is not None:
                dt = state_seen_at - self._robot_prev_state_seen_at
                if dt > 0.0:
                    self._robot_cycle_ms = dt * 1000.0
            self._robot_prev_state_seen_at = state_seen_at
        stale = state_seen_at is None or (now - state_seen_at) > POSITION_STALE_SEC
        # _last_error는 과거 예외 이력이 누적될 수 있으므로, UI 상태등급은 현재 상태코드 기준으로 판정한다.
        has_error = (state_code is not None) and (int(state_code) in ROBOT_STATE_ERROR_CODES)
        comm_connected = (not stale) and (state_code is not None)
        if now < float(getattr(self, "_mode_switch_grace_until", 0.0)):
            # 전환 직후 짧은 공백은 연결 끊김으로 보지 않는다.
            if self.backend.is_ready() and ((state_code is None) or stale):
                comm_connected = bool(getattr(self, "_last_robot_comm_connected", True))
                stale = False
        self._last_robot_comm_connected = bool(comm_connected)
        controls_enabled = self.backend.is_ready() and comm_connected
        if self._reset_thread is not None and self._reset_thread.isRunning():
            controls_enabled = False
        self._set_robot_controls_enabled(controls_enabled)

        if state_code is not None:
            state_name = (state_name or f"STATE_{state_code}").strip()
            state_text = ROBOT_STATE_KR_MAP.get(state_name, state_name)
            state_label_text = f"{state_text} ({state_name})"
            if int(state_code) == 1:
                # STANDBY(대기)는 에러 이력과 무관하게 정상색(초록)으로 유지한다.
                robot_severity = "normal"
            elif has_error or int(state_code) in ROBOT_STATE_ERROR_CODES:
                robot_severity = "error"
            elif stale or int(state_code) in ROBOT_STATE_WARNING_CODES:
                robot_severity = "warning"
            elif int(state_code) in ROBOT_STATE_NORMAL_CODES:
                robot_severity = "normal"
            else:
                robot_severity = "warning"
        else:
            if not self.backend.is_ready():
                state_text = "초기화 중"
                robot_severity = "warning"
            elif has_error:
                state_text = "오류"
                robot_severity = "error"
            elif self.backend.is_busy():
                # 명령 처리 중이더라도 실제 로봇 상태 미수신이면 "동작"으로 단정하지 않는다.
                state_text = "요청중"
                robot_severity = "normal"
            else:
                state_text = "대기"
                robot_severity = "warning"
            state_label_text = state_text

        if state_text != self._last_robot_state_text:
            self._last_robot_state_text = state_text
            self._robot_state_flash_until = now + STATE_FLASH_SEC
        flash_state = now < self._robot_state_flash_until

        self._set_signal_state_label(self._robot_state_label, state_label_text, robot_severity, flash_state)
        self._update_robot_state_table(
            state_code=state_code,
            state_name=state_name,
            stale=stale,
            mode_text=mode_text,
            robot_mode_value=robot_mode_value,
            robot_mode_seen_at=robot_mode_seen_at,
            control_mode_value=control_mode_value,
            control_mode_seen_at=control_mode_seen_at,
            app_ready=self.backend.is_ready(),
            app_busy=self.backend.is_busy(),
            has_error=has_error,
        )
        robot_link_text = "정상 수신 중" if comm_connected else "끊김"
        robot_link_sev = "normal" if comm_connected else "error"
        self._set_top_status("robot", robot_link_text, robot_link_sev)
        self._update_cycle_time_labels()

    def _refresh_vision_status(self):
        now = time.monotonic()
        vision1_on = bool(self._top_status_enabled.get("vision", True))
        vision2_on = bool(self._top_status_enabled.get("vision2", True))
        retry_needed = False
        if not vision1_on:
            self._vision_cycle_ms = None
        if not vision2_on:
            self._vision_cycle_ms_2 = None
        if self.backend is None:
            self._set_signal_state_label(self._vision_state_label, self._vision_state_text, "warning", False)
            self._set_top_status("vision", "비활성화" if (not vision1_on) else "끊김", "warning" if (not vision1_on) else "error")
            self._set_top_status("vision2", "비활성화" if (not vision2_on) else "끊김", "warning" if (not vision2_on) else "error")
            if not vision1_on:
                self._clear_vision_view_data(panel_index=1)
            if not vision2_on:
                self._clear_vision_view_data(panel_index=2)
            return

        if not UI_ENABLE_VISION:
            vision_text = "비활성화"
            vision2_text = "비활성화"
        else:
            if not vision1_on:
                vision_text = "비활성화"
                self._clear_vision_view_data(panel_index=1)
            else:
                vision_text = self._camera_status_for_panel(1, now, vision1_on)

            if not vision2_on:
                vision2_text = "비활성화"
                self._clear_vision_view_data(panel_index=2)
            else:
                vision2_text = self._camera_status_for_panel(2, now, vision2_on)

        if now < float(getattr(self, "_mode_switch_grace_until", 0.0)):
            if vision_text in ("지연", "끊김", "오류", "실패"):
                vision_text = "전환중"
            if vision2_text in ("지연", "끊김", "오류", "실패"):
                vision2_text = "전환중"

        # 외부 비전 프레임이 일정 시간 끊기면 자동 재기동/재연결을 시도한다.
        calib_on_1 = bool(getattr(self, "_calibration_mode_enabled_1", False))
        calib_on_2 = bool(getattr(self, "_calibration_mode_enabled_2", False))
        calib_on_any = bool(calib_on_1 or calib_on_2)

        if YOLO_EXTERNAL_NODE and YOLO_AUTO_LAUNCH_NODE and (not calib_on_any):
            stale_vision = False
            if vision1_on:
                stale_vision = stale_vision or (
                    (self._last_vision_frame_at is None) or ((now - self._last_vision_frame_at) > 3.0)
                )
            if vision2_on:
                stale_vision = stale_vision or (
                    (self._last_vision_frame_at_2 is None) or ((now - self._last_vision_frame_at_2) > 3.0)
                )
            if stale_vision:
                retry_needed = True
                self._ensure_external_vision_process()
        if YOLO_EXTERNAL_NODE and calib_on_any:
            stale_vision = False
            if calib_on_1 and vision1_on:
                stale_vision = stale_vision or (
                    (self._last_vision_frame_at is None)
                    or ((now - self._last_vision_frame_at) > 3.0)
                )
            if calib_on_2 and vision2_on:
                stale_vision = stale_vision or (
                    (self._last_vision_frame_at_2 is None)
                    or ((now - self._last_vision_frame_at_2) > 3.0)
                )
            if stale_vision and ((now - self._vision_rebind_last_try_at) > 8.0):
                retry_needed = True
                self._vision_rebind_last_try_at = now
                self._sync_calibration_processes()
                self._rebind_external_vision_bridge_for_mode()

        if retry_needed:
            if not bool(getattr(self, "_vision_retry_notice_logged", False)):
                self.append_log("[비전] 연결 지연 판단: 재연결 시도를 계속 진행합니다.\n")
                self._vision_retry_notice_logged = True
        else:
            self._vision_retry_notice_logged = False

        if vision_text == "정상 수신 중":
            vision_severity = "normal"
        elif vision_text in ("오류", "실패", "끊김"):
            vision_severity = "error"
        else:
            vision_severity = "warning"
        if vision2_text == "정상 수신 중":
            vision2_severity = "normal"
        elif vision2_text in ("오류", "실패", "끊김"):
            vision2_severity = "error"
        else:
            vision2_severity = "warning"

        if self._last_yolo_qimage is None and vision_text in ("오류", "실패", "끊김", "지연"):
            if hasattr(self, "yolo_view") and self.yolo_view is not None:
                self.yolo_view.setAlignment(Qt.AlignCenter)
                self.yolo_view.setStyleSheet("background-color: #000000; color: #f3f3f3;")
                self.yolo_view.setText("이미지 데이터 없음")
        if self._last_yolo_qimage_2 is None and vision2_text in ("오류", "실패", "끊김", "지연"):
            if hasattr(self, "yolo_view_2") and self.yolo_view_2 is not None:
                self.yolo_view_2.setAlignment(Qt.AlignCenter)
                self.yolo_view_2.setStyleSheet("background-color: #000000; color: #f3f3f3;")
                self.yolo_view_2.setText("이미지 데이터 없음")
        self._set_signal_state_label(self._vision_state_label, vision_text, vision_severity, False)
        self._set_top_status("vision", vision_text, vision_severity)
        self._set_top_status("vision2", vision2_text, vision2_severity)
        self._update_cycle_time_labels()

    def _set_signal_state_label(self, label: QLabel, text: str, severity: str, is_bold: bool = False):
        if label is None:
            return
        label.setText(text)
        if severity == "normal":
            color = STATE_COLOR_NORMAL
        elif severity == "error":
            color = STATE_COLOR_ERROR
        else:
            color = STATE_COLOR_WARNING
        weight = 700 if is_bold else 500
        label.setStyleSheet(f"color: {color}; font-weight: {weight};")

    def _set_table_value(self, row: int, value: str, severity: str = "", flash: bool = True):
        if not hasattr(self, "_robot_state_table") or self._robot_state_table is None:
            return
        now = time.monotonic()
        text_value = str(value)
        old_value = self._table_value_cache.get(row)
        if flash and old_value != text_value:
            self._table_flash_until[row] = now + STATE_FLASH_SEC
        self._table_value_cache[row] = text_value

        item = self._robot_state_table.item(row, 1)
        if item is None:
            item = QTableWidgetItem("")
            self._robot_state_table.setItem(row, 1, item)
        item.setText(text_value)

        if severity == "normal":
            color = QColor(STATE_COLOR_NORMAL)
        elif severity == "error":
            color = QColor(STATE_COLOR_ERROR)
        elif severity == "warning":
            color = QColor(STATE_COLOR_WARNING)
        else:
            color = self.palette().text().color()
        item.setForeground(QBrush(color))

        is_bold = now < self._table_flash_until.get(row, 0.0)
        font = item.font()
        font.setBold(is_bold)
        item.setFont(font)

    def _update_robot_state_table(
        self,
        state_code,
        state_name,
        stale,
        mode_text,
        robot_mode_value,
        robot_mode_seen_at,
        control_mode_value,
        control_mode_seen_at,
        app_ready,
        app_busy,
        has_error,
    ):
        connection_text = {
            "REAL": "실제로봇",
            "VIRTUAL": "가상로봇",
            "UNKNOWN": "알수없음",
        }.get(str(mode_text or "UNKNOWN").upper(), str(mode_text or "UNKNOWN"))
        connection_sev = "normal" if str(mode_text or "").upper() in ("REAL", "VIRTUAL") else "warning"

        comm_text = "끊김" if stale else "정상 수신 중"
        comm_sev = "error" if stale else "normal"

        if state_code is None:
            robot_state_text = "초기화" if not app_ready else "대기"
            robot_state_sev = "warning" if not app_ready else ("error" if has_error else "warning")
        else:
            normalized_name = str(state_name or f"STATE_{state_code}").strip()
            robot_state_text = ROBOT_STATE_KR_MAP.get(normalized_name, normalized_name)
            if int(state_code) in ROBOT_STATE_ERROR_CODES:
                robot_state_sev = "error"
            elif int(state_code) in ROBOT_STATE_NORMAL_CODES:
                robot_state_sev = "normal"
            else:
                robot_state_sev = "warning"

        robot_mode_map = {0: "메뉴얼모드", 1: "오토모드", 2: "측정모드"}
        if robot_mode_value is None or robot_mode_seen_at is None:
            robot_mode_text = "-"
            robot_mode_sev = "warning"
        else:
            try:
                robot_mode_text = robot_mode_map.get(int(robot_mode_value), str(int(robot_mode_value)))
            except Exception:
                robot_mode_text = str(robot_mode_value)
            robot_mode_sev = "normal"

        if control_mode_value is None or control_mode_seen_at is None:
            control_mode_text = "-"
            control_mode_sev = "warning"
        else:
            try:
                control_mode_text = str(int(control_mode_value))
            except Exception:
                control_mode_text = str(control_mode_value)
            control_mode_sev = "normal"

        if has_error:
            work_text = "오류"
            work_sev = "error"
        elif not app_ready:
            work_text = "초기화"
            work_sev = "warning"
        elif app_busy:
            work_text = "동작중"
            work_sev = "normal"
        else:
            work_text = "대기"
            work_sev = "normal"

        self._set_table_value(0, connection_text, connection_sev, flash=False)
        self._set_table_value(1, comm_text, comm_sev)
        self._set_table_value(2, robot_state_text, robot_state_sev)
        self._set_table_value(3, robot_mode_text, robot_mode_sev, flash=False)
        self._set_table_value(4, control_mode_text, control_mode_sev, flash=False)
        self._set_table_value(5, work_text, work_sev)

    def _run_vision_target_move(self, vision_xyz_mm, robot_xyz_mm, source_panel: int = 1):
        src = 2 if int(source_panel) == 2 else 1
        tag = f"[비전{src}이동]"
        if self.backend is None:
            self.append_log(f"{tag} 백엔드 초기화 중입니다.\n")
            return
        if not hasattr(self.backend, "send_move_vision_point"):
            self.append_log(f"{tag} 백엔드가 비전 좌표 이동 기능을 지원하지 않습니다.\n")
            return

        rx, ry, rz = robot_xyz_mm
        z_margin = self._get_vision_move_z_margin_mm()
        target_z = float(rz) + float(z_margin)
        ok, msg = self.backend.send_move_vision_point(rx, ry, target_z, dwell_sec=1.0)
        if vision_xyz_mm is not None:
            vx, vy, vz = vision_xyz_mm
            self.append_log(
                f"{tag} 비전 XYZ(mm): X={vx:.1f}, Y={vy:.1f}, Z={vz:.1f} -> "
                f"Robot XYZ(mm): X={rx:.2f}, Y={ry:.2f}, Z={target_z:.2f} [Z+{z_margin:.1f}]\n"
            )
        self.append_log(f"{tag} {msg}\n")

    def on_move_to_last_vision_point(self):
        if self._last_clicked_robot_xyz_mm is None or self._last_clicked_vision_xyz_mm is None:
            self.append_log("[비전이동] 마지막 클릭 좌표가 없습니다. 비전1/2 화면을 먼저 클릭하세요.\n")
            return
        self._run_vision_target_move(
            self._last_clicked_vision_xyz_mm,
            self._last_clicked_robot_xyz_mm,
            source_panel=getattr(self, "_last_clicked_source_vision", 1),
        )

    def _detect_robot_mode_text(self):
        if hasattr(self.backend, "get_connection_mode_snapshot"):
            mode_text, _ = self.backend.get_connection_mode_snapshot()
            mode_text = str(mode_text or "").upper().strip()
            if mode_text in ("REAL", "VIRTUAL", "UNKNOWN"):
                self._mode_text_cached = mode_text
                return self._mode_text_cached
        if not self._mode_text_cached:
            self._mode_text_cached = "UNKNOWN"
        return self._mode_text_cached

    def _setup_rosout_bridge(self):
        if RosLogMsg is None:
            self.append_log("[시스템] /rosout 브리지 비활성화: rcl_interfaces.msg.Log import 실패\n")
            return
        node = getattr(self.backend, "robot_controller", None)
        if node is None:
            self.append_log("[시스템] /rosout 브리지 비활성화: robot_controller 노드 없음\n")
            return
        try:
            self._rosout_sub = node.create_subscription(RosLogMsg, "/rosout", self._on_rosout_msg, 200)
            self.append_log("[시스템] /rosout 로그 미러링 시작\n")
        except Exception as e:
            self.append_log(f"[시스템] /rosout 브리지 생성 실패: {e}\n")

    def _on_rosout_msg(self, msg):
        name = str(getattr(msg, "name", ""))
        if name not in ("bartender_backend", "bartender_robot_app"):
            return

        level = int(getattr(msg, "level", 0))
        if level >= 50:
            level_text = "FATAL"
        elif level >= 40:
            level_text = "ERROR"
        elif level >= 30:
            level_text = "WARN"
        elif level >= 20:
            level_text = "INFO"
        else:
            level_text = "DEBUG"

        text = str(getattr(msg, "msg", ""))
        self.ros_log_received.emit(f"[ROS:{level_text}] [{name}] {text}\n")

    def _setup_robot_state_table(self):
        self._robot_state_table = getattr(self, "robot_state_table", None)
        if self._robot_state_table is None:
            return

        self._robot_state_table.setRowCount(6)
        self._robot_state_table.setColumnCount(2)
        self._robot_state_table.setHorizontalHeaderLabels(["항목", "값"])
        self._robot_state_table.verticalHeader().setVisible(False)
        self._robot_state_table.horizontalHeader().setMinimumSectionSize(20)
        self._robot_state_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._robot_state_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._robot_state_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._robot_state_table.setSelectionMode(QTableWidget.NoSelection)
        self._robot_state_table.setFocusPolicy(Qt.NoFocus)
        self._robot_state_table.setWordWrap(False)
        self._robot_state_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._robot_state_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._robot_state_table.setFont(QFont(UI_FONT_FAMILY, UI_TABLE_FONT_SIZE))
        self._robot_state_table.verticalHeader().setDefaultSectionSize(UI_PANEL_TABLE_ROW_HEIGHT)

        keys = ["연결타입", "통신상태", "로봇상태", "로봇모드", "제어모드", "작업상태"]
        for row, key in enumerate(keys):
            self._robot_state_table.setItem(row, 0, QTableWidgetItem(key))
            self._robot_state_table.setItem(row, 1, QTableWidgetItem("-"))
        self._apply_table_font_scale(self._robot_state_table, 1.0)
        self._fit_table_height(self._robot_state_table)

    def _setup_position_tables(self):
        hide_names = [
            "j1_label", "j2_label", "j3_label", "j4_label", "j5_label", "j6_label",
            "j1_value", "j2_value", "j3_value", "j4_value", "j5_value", "j6_value",
            "x_label", "y_label", "z_label", "a_label", "b_label", "c_label",
            "x_value", "y_value", "z_value", "a_value", "b_value", "c_value",
        ]
        for name in hide_names:
            w = getattr(self, name, None)
            if w is not None:
                w.hide()

        self._joint_table = getattr(self, "joint_table", None)
        self._cart_table = getattr(self, "cart_table", None)
        if self._joint_table is None or self._cart_table is None:
            return

        self._setup_position_table_common(self._joint_table, ["J1", "J2", "J3", "J4", "J5", "J6"])
        self._setup_position_table_common(self._cart_table, ["X", "Y", "Z", "A", "B", "C"])
        self._setup_current_tool_label()

    def _refresh_positions(self):
        if getattr(self, "_joint_table", None) is None or getattr(self, "_cart_table", None) is None:
            return
        if self.backend is None or (not bool(self._top_status_enabled.get("robot", True))):
            self._clear_position_views()
            return
        if hasattr(self.backend, "get_position_snapshot"):
            data, seen_at = self.backend.get_position_snapshot()
        else:
            data = self.backend.get_current_positions() if hasattr(self.backend, "get_current_positions") else None
            seen_at = None
        if not data or len(data) < 2:
            self._clear_position_views()
            return
        now = time.monotonic()
        if seen_at is not None and (now - float(seen_at)) > POSITION_STALE_SEC:
            self._clear_position_views()
            return
        posj, posx = data
        if posj is None or posx is None or len(posj) < 6 or len(posx) < 6:
            self._clear_position_views()
            return

        def _update_table(table, prefix, values):
            for row in range(6):
                item = table.item(row, 1)
                if item is None:
                    item = QTableWidgetItem("-")
                    table.setItem(row, 1, item)
                text_value = self._fmt_ui_float(values[row], 2)
                cache_key = (prefix, row)
                if self._pos_value_cache.get(cache_key) != text_value:
                    self._pos_flash_until[cache_key] = now + POSITION_FLASH_SEC
                self._pos_value_cache[cache_key] = text_value
                item.setText(text_value)
                item.setForeground(QBrush(self.palette().text().color()))
                font = item.font()
                font.setBold(now < self._pos_flash_until.get(cache_key, 0.0))
                item.setFont(font)

        _update_table(self._joint_table, "joint", [float(v) for v in list(posj)[:6]])
        _update_table(self._cart_table, "cart", [float(v) for v in list(posx)[:6]])
        self._last_positions_seen_at = seen_at if seen_at is not None else now
        if hasattr(self.backend, "get_current_tcp_snapshot"):
            tcp_name, _ = self.backend.get_current_tcp_snapshot()
            if tcp_name is not None:
                self._set_current_tool_text(tcp_name)

    def _setup_current_tool_label(self):
        if self._cart_table is None:
            return
        if self._current_tool_label is not None:
            return

        ui_label = getattr(self, "current_tool_label", None)
        if isinstance(ui_label, QLabel):
            self._current_tool_label = ui_label
            self._current_tool_label.show()
            self._current_tool_label.raise_()
            return

        label = QLabel("현재툴(TCP): -", self._cart_table.parentWidget())
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        inserted = False
        parent = self._cart_table.parentWidget()
        if parent is not None and parent.layout() is not None:
            lay = parent.layout()
            idx = lay.indexOf(self._cart_table)
            if idx >= 0:
                lay.insertWidget(idx, label)
                inserted = True

        if not inserted:
            try:
                g = self._cart_table.geometry()
                h = 14
                y = max(0, g.y() - h - 2)
                label.setGeometry(g.x(), y, g.width(), h)
            except Exception:
                pass

        self._current_tool_label = label
        self._current_tool_label.show()

    def _set_current_tool_text(self, tcp_name):
        tcp = str(tcp_name or "").strip()
        display = "flange" if tcp == "" else tcp
        text = f"현재툴(TCP): {display}"
        if text == self._current_tool_text_cache:
            return
        self._current_tool_text_cache = text
        if self._current_tool_label is not None:
            self._current_tool_label.setText(text)

    def _refresh_current_tool_label_once(self, log_fail: bool = False):
        if self.backend is None:
            self._set_current_tool_text("-")
            return False
        if not hasattr(self.backend, "sync_current_tcp_once") or not hasattr(self.backend, "get_current_tcp_snapshot"):
            self._set_current_tool_text("-")
            return False
        ok_tcp, msg_tcp = self.backend.sync_current_tcp_once(force=True, timeout_sec=1.0)
        if not ok_tcp:
            if log_fail:
                self.append_log(f"[툴표시] 현재 TCP 조회 실패: {msg_tcp}\n")
            return False
        tcp_name, _tcp_seen_at = self.backend.get_current_tcp_snapshot()
        self._set_current_tool_text(tcp_name)
        return True

    def _setup_cycle_time_labels(self):
        self._vision_cycle_label = getattr(self, "vision_cycle_label", None)
        self._vision_cycle_label_2 = getattr(self, "vision_cycle_label_2", None)
        self._robot_cycle_label = getattr(self, "robot_cycle_label", None)
        style = f"color: #555; font-size: {max(7, UI_TERMINAL_FONT_SIZE - 1)}pt;"
        if self._vision_cycle_label is not None:
            self._vision_cycle_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._vision_cycle_label.setStyleSheet(style)
            self._vision_cycle_label.show()
        if self._vision_cycle_label_2 is not None:
            self._vision_cycle_label_2.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._vision_cycle_label_2.setStyleSheet(style)
            self._vision_cycle_label_2.show()
        if self._robot_cycle_label is not None:
            self._robot_cycle_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._robot_cycle_label.setStyleSheet(style)

        serial_style = style
        self._vision_serial_label = getattr(self, "vision_serial_label", None)
        self._vision_serial_label_2 = getattr(self, "vision_serial_label_2", None)
        self._vision_serial_change_button = getattr(self, "vision_serial_change_button", None)
        self._vision_serial_change_button_2 = getattr(self, "vision_serial_change_button_2", None)
        self._vision_rotate_left_button = getattr(self, "vision_rotate_left_button", None)
        self._vision_rotate_zero_button = getattr(self, "vision_rotate_zero_button", None)
        self._vision_rotate_right_button = getattr(self, "vision_rotate_right_button", None)
        self._vision_rotation_label = getattr(self, "vision_rotation_label", None)
        self._vision_rotate_left_button_2 = getattr(self, "vision_rotate_left_button_2", None)
        self._vision_rotate_zero_button_2 = getattr(self, "vision_rotate_zero_button_2", None)
        self._vision_rotate_right_button_2 = getattr(self, "vision_rotate_right_button_2", None)
        self._vision_rotation_label_2 = getattr(self, "vision_rotation_label_2", None)

        if self._vision_serial_label is not None:
            self._vision_serial_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._vision_serial_label.setStyleSheet(serial_style)
        if self._vision_serial_label_2 is not None:
            self._vision_serial_label_2.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._vision_serial_label_2.setStyleSheet(serial_style)
        if self._vision_rotation_label is not None:
            self._vision_rotation_label.setStyleSheet(serial_style)
        if self._vision_rotation_label_2 is not None:
            self._vision_rotation_label_2.setStyleSheet(serial_style)
        if self._vision_serial_change_button is not None:
            try:
                self._vision_serial_change_button.clicked.disconnect()
            except Exception:
                pass
            self._vision_serial_change_button.clicked.connect(lambda: self._on_change_vision_camera(1))
        if self._vision_serial_change_button_2 is not None:
            try:
                self._vision_serial_change_button_2.clicked.disconnect()
            except Exception:
                pass
            self._vision_serial_change_button_2.clicked.connect(lambda: self._on_change_vision_camera(2))
        if self._vision_rotate_left_button is not None:
            try:
                self._vision_rotate_left_button.clicked.disconnect()
            except Exception:
                pass
            self._vision_rotate_left_button.clicked.connect(lambda: self._set_vision_rotation(1, -90))
        if self._vision_rotate_right_button is not None:
            try:
                self._vision_rotate_right_button.clicked.disconnect()
            except Exception:
                pass
            self._vision_rotate_right_button.clicked.connect(lambda: self._set_vision_rotation(1, 90))
        if self._vision_rotate_zero_button is not None:
            try:
                self._vision_rotate_zero_button.clicked.disconnect()
            except Exception:
                pass
            self._vision_rotate_zero_button.clicked.connect(lambda: self._set_vision_rotation_absolute(1, 0))
        if self._vision_rotate_left_button_2 is not None:
            try:
                self._vision_rotate_left_button_2.clicked.disconnect()
            except Exception:
                pass
            self._vision_rotate_left_button_2.clicked.connect(lambda: self._set_vision_rotation(2, -90))
        if self._vision_rotate_zero_button_2 is not None:
            try:
                self._vision_rotate_zero_button_2.clicked.disconnect()
            except Exception:
                pass
            self._vision_rotate_zero_button_2.clicked.connect(lambda: self._set_vision_rotation_absolute(2, 0))
        if self._vision_rotate_right_button_2 is not None:
            try:
                self._vision_rotate_right_button_2.clicked.disconnect()
            except Exception:
                pass
            self._vision_rotate_right_button_2.clicked.connect(lambda: self._set_vision_rotation(2, 90))
        self._refresh_vision_rotation_labels()

        self._reposition_cycle_labels()
        self._refresh_vision_serial_labels()

    def _hide_robot_control_title_label(self):
        panel = getattr(self, "frame_2", None)
        if panel is None:
            return
        try:
            for w in panel.findChildren(QLabel):
                txt = (w.text() or "").strip()
                if txt in ("[로봇 조작]", "로봇 조작"):
                    w.hide()
        except Exception:
            return

    def _build_vision_to_robot_affine(self):
        try:
            uvz = VISION_CALIB_UVZ_MM
            xyz = ROBOT_CALIB_XYZ_MM
            if uvz.shape != xyz.shape or uvz.shape[0] < 4 or uvz.shape[1] != 3:
                self.append_log("[비전] 보정 데이터 형식 오류: 변환행렬 생성 실패\n")
                return
            result, err = self._compute_rigid_uvz_to_xyz(uvz, xyz, require_intrinsics=False)
            if result is None:
                self.append_log(f"[비전] 기본 회전행렬 생성 실패: {err}\n")
                return
            affine, _, _, rmse = result
            self._vision_to_robot_affine = affine
            self._vision_to_robot_rmse = rmse
            self._vision_to_robot_affine_1 = np.asarray(affine, dtype=np.float64)
            self._vision_to_robot_affine_2 = np.asarray(affine, dtype=np.float64)
            self._vision_to_robot_rmse_1 = rmse
            self._vision_to_robot_rmse_2 = rmse
            self.append_log(f"[비전] 비전 XYZ->로봇 회전행렬(R,t) 적용 (보정 RMSE={rmse:.3f}mm)\n")
        except Exception as e:
            self._vision_to_robot_affine = None
            self._vision_to_robot_rmse = None
            self._vision_to_robot_affine_1 = None
            self._vision_to_robot_affine_2 = None
            self._vision_to_robot_rmse_1 = None
            self._vision_to_robot_rmse_2 = None
            self.append_log(f"[비전] 변환행렬 생성 실패: {e}\n")

    def _vision_uvz_to_robot_xyz_mm(self, u: float, v: float, z_mm: float, panel_index: int = 1):
        panel = 2 if int(panel_index) == 2 else 1
        affine = self._vision_to_robot_affine_2 if panel == 2 else self._vision_to_robot_affine_1
        if affine is None:
            affine = self._vision_to_robot_affine
        if affine is None:
            return None
        # 비전 클릭 이동은 반드시 UVZ -> Camera XYZ(mm) 변환이 되어야 한다.
        cam_xyz = self._uvz_to_camera_xyz_mm(u, v, z_mm, require_intrinsics=True, panel_index=panel)
        if cam_xyz is None:
            return None
        cx, cy, cz = cam_xyz
        vec = np.array([float(cx), float(cy), float(cz), 1.0], dtype=np.float64)
        out = vec @ affine
        return float(out[0]), float(out[1]), float(out[2])

    def _apply_vision_to_robot_affine(self, mat, rmse=None, path=None, panel_index: int = 1):
        arr = np.asarray(mat, dtype=np.float64)
        if arr.shape != (4, 3):
            raise ValueError("vision affine must be 4x3")
        panel = 2 if int(panel_index) == 2 else 1
        resolved_path = _resolve_repo_file(path, fallback_dirs=(CALIB_DIR, CALIB_ROTMAT_DIR)) if path else None
        if panel == 2:
            self._vision_to_robot_affine_2 = np.asarray(arr, dtype=np.float64)
            self._vision_to_robot_rmse_2 = None if rmse is None else float(rmse)
            self._calib_matrix_path_2 = resolved_path
        else:
            self._vision_to_robot_affine_1 = np.asarray(arr, dtype=np.float64)
            self._vision_to_robot_rmse_1 = None if rmse is None else float(rmse)
            self._calib_matrix_path_1 = resolved_path
            self._vision_to_robot_affine = np.asarray(arr, dtype=np.float64)
            self._vision_to_robot_rmse = None if rmse is None else float(rmse)
            self._calib_matrix_path = resolved_path
        if self._vision_to_robot_affine is None:
            self._vision_to_robot_affine = np.asarray(arr, dtype=np.float64)
            self._vision_to_robot_rmse = None if rmse is None else float(rmse)
            self._calib_matrix_path = resolved_path
        if resolved_path:
            self._save_active_calibration_path(resolved_path, panel_index=panel)
        self.calibration_ui_refresh_requested.emit()

    def _reposition_cycle_labels(self):
        if hasattr(self, "_vision_cycle_label") and self._vision_cycle_label is not None:
            parent = self._vision_cycle_label.parentWidget()
            w = parent.width() if parent is not None else 511
            right_x = max(220, w - 180)
            self._vision_cycle_label.setGeometry(right_x, 8, 168, 14)
            if self._vision_serial_label is not None:
                self._vision_serial_label.setGeometry(right_x - 52, 24, 220, 14)
            if self._vision_rotate_left_button is not None:
                self._vision_rotate_left_button.raise_()
            if self._vision_rotate_zero_button is not None:
                self._vision_rotate_zero_button.raise_()
            if self._vision_rotate_right_button is not None:
                self._vision_rotate_right_button.raise_()
        if hasattr(self, "_vision_cycle_label_2") and self._vision_cycle_label_2 is not None:
            parent = self._vision_cycle_label_2.parentWidget()
            w = parent.width() if parent is not None else 511
            right_x2 = max(220, w - 180)
            self._vision_cycle_label_2.setGeometry(right_x2, 8, 168, 14)
            if self._vision_serial_label_2 is not None:
                self._vision_serial_label_2.setGeometry(right_x2 - 52, 24, 220, 14)
            if self._vision_rotate_left_button_2 is not None:
                self._vision_rotate_left_button_2.raise_()
            if self._vision_rotate_zero_button_2 is not None:
                self._vision_rotate_zero_button_2.raise_()
            if self._vision_rotate_right_button_2 is not None:
                self._vision_rotate_right_button_2.raise_()
        if hasattr(self, "_robot_cycle_label") and self._robot_cycle_label is not None:
            parent = self._robot_cycle_label.parentWidget()
            w = parent.width() if parent is not None else 531
            self._robot_cycle_label.setGeometry(max(240, w - 190), 8, 178, 14)

    def _update_cycle_time_labels(self):
        if hasattr(self, "_vision_cycle_label") and self._vision_cycle_label is not None:
            if self._vision_cycle_ms is None:
                self._vision_cycle_label.setText("업데이트: - ms")
            else:
                self._vision_cycle_label.setText(f"업데이트: {self._vision_cycle_ms:.1f} ms")
        if hasattr(self, "_vision_cycle_label_2") and self._vision_cycle_label_2 is not None:
            if self._vision_cycle_ms_2 is None:
                self._vision_cycle_label_2.setText("업데이트: - ms")
            else:
                self._vision_cycle_label_2.setText(f"업데이트: {self._vision_cycle_ms_2:.1f} ms")
        if hasattr(self, "_robot_cycle_label") and self._robot_cycle_label is not None:
            if self._robot_cycle_ms is None:
                self._robot_cycle_label.setText("업데이트: - ms")
            else:
                self._robot_cycle_label.setText(f"업데이트: {self._robot_cycle_ms:.1f} ms")
        self._refresh_vision_serial_labels()

    def performance_snapshot(self):
        return {
            "vision1": {
                "frontend_receive_interval_ms": self._vision_cycle_ms,
                "frontend_decode_ms": self._vision_decode_ms,
                "frontend_render_delay_ms": self._vision_render_delay_ms,
                "frontend_render_interval_ms": self._vision_render_interval_ms,
                "calib_input_interval_ms": self._calib_proc_input_ms_1,
                "calib_processing_ms": self._calib_proc_ms_1,
                "calib_publish_interval_ms": self._calib_proc_publish_ms_1,
                "calib_frame_wait_ms": self._calib_proc_wait_ms_1,
                "calib_reason": self._calib_last_reason_1,
            },
            "vision2": {
                "frontend_receive_interval_ms": self._vision_cycle_ms_2,
                "frontend_decode_ms": self._vision_decode_ms_2,
                "frontend_render_delay_ms": self._vision_render_delay_ms_2,
                "frontend_render_interval_ms": self._vision_render_interval_ms_2,
                "calib_input_interval_ms": self._calib_proc_input_ms_2,
                "calib_processing_ms": self._calib_proc_ms_2,
                "calib_publish_interval_ms": self._calib_proc_publish_ms_2,
                "calib_frame_wait_ms": self._calib_proc_wait_ms_2,
                "calib_reason": self._calib_last_reason_2,
            },
        }

    def _on_change_vision_camera(self, panel_index: int):
        discovered = self._discover_connected_camera_serials()
        for s in discovered:
            if s not in self._available_camera_serials:
                self._available_camera_serials.append(s)
        options = [s for s in self._available_camera_serials if s]
        if not options:
            QMessageBox.warning(self, "카메라 변경", "검색된 카메라 시리얼이 없습니다.")
            return

        current = self._vision_assigned_serial_1 if int(panel_index) == 1 else self._vision_assigned_serial_2
        try:
            idx = options.index(current)
        except Exception:
            idx = 0
        selected, ok = QInputDialog.getItem(
            self,
            "카메라 변경",
            f"비전{int(panel_index)}에 할당할 카메라 시리얼을 선택하세요.",
            options,
            idx,
            False,
        )
        if not ok:
            return
        picked = self._normalize_serial_text(selected)
        if not picked:
            return

        if int(panel_index) == 1:
            prev = self._vision_assigned_serial_1
            self._vision_assigned_serial_1 = picked
            if picked == self._vision_assigned_serial_2:
                self._vision_assigned_serial_2 = prev
        else:
            prev = self._vision_assigned_serial_2
            self._vision_assigned_serial_2 = picked
            if picked == self._vision_assigned_serial_1:
                self._vision_assigned_serial_1 = prev

        self._save_vision_serial_settings()
        self._refresh_vision_serial_labels()
        self._clear_calibration_panel_state(1)
        self._clear_calibration_panel_state(2)
        self._clear_vision_view_data(1)
        self._clear_vision_view_data(2)
        self._advance_vision_stream_token(1)
        self._advance_vision_stream_token(2)
        now = time.monotonic()
        self._vision_drop_frames_until_1 = now + 1.2
        self._vision_drop_frames_until_2 = now + 1.2
        self._vision_state_text = "전환중"
        self._vision_state_text_2 = "전환중"
        self._sync_calibration_processes()
        self._rebind_external_vision_bridge_for_mode()
        self.append_log(
            f"[비전] 카메라 할당 변경: 비전1={self._vision_assigned_serial_1}, 비전2={self._vision_assigned_serial_2} (런타임 camera={self._runtime_camera_serial_1}, camera2={self._runtime_camera_serial_2})\n"
        )

    def _setup_position_table_common(self, table: QTableWidget, keys):
        table.setRowCount(len(keys))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["항목", "값"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setMinimumSectionSize(20)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setFocusPolicy(Qt.NoFocus)
        table.setWordWrap(False)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setFont(QFont(UI_FONT_FAMILY, UI_TABLE_FONT_SIZE))
        table.verticalHeader().setDefaultSectionSize(UI_PANEL_TABLE_ROW_HEIGHT)
        for row, key in enumerate(keys):
            table.setItem(row, 0, QTableWidgetItem(key))
            table.setItem(row, 1, QTableWidgetItem("-"))
        self._apply_table_font_scale(table, 1.0)
        self._fit_table_height(table)

    def _fit_table_height(self, table: QTableWidget):
        if table is None:
            return
        try:
            total_h = table.horizontalHeader().height()
            for row in range(table.rowCount()):
                total_h += table.rowHeight(row)
            total_h += table.frameWidth() * 2 + 2
            table.setMinimumHeight(total_h)
            table.setMaximumHeight(total_h)
        except Exception:
            pass

    def _apply_table_font_scale(self, table: QTableWidget, scale: float):
        if table is None:
            return
        try:
            scale = float(scale)
        except Exception:
            scale = 1.0
        scale = max(0.3, min(1.0, scale))

        base_font = table.font()
        base_pt = base_font.pointSizeF() if base_font.pointSizeF() > 0 else float(UI_FONT_SIZE)
        new_pt = max(6.0, base_pt * scale)

        f = QFont(base_font)
        f.setPointSizeF(new_pt)
        table.setFont(f)

        hh = table.horizontalHeader()
        if hh is not None:
            hf = QFont(hh.font())
            hpt = hf.pointSizeF() if hf.pointSizeF() > 0 else base_pt
            hf.setPointSizeF(max(6.0, hpt * scale))
            hh.setFont(hf)

        for r in range(table.rowCount()):
            for c in range(table.columnCount()):
                item = table.item(r, c)
                if item is None:
                    continue
                if c == 1:
                    # 값 컬럼은 상태 강조(bold) 업데이트 로직이 있으므로 크기만 고정.
                    if item.font().pointSizeF() <= 0:
                        item.setFont(QFont(f))
                    else:
                        itf = QFont(item.font())
                        itf.setPointSizeF(new_pt)
                        item.setFont(itf)
                else:
                    item.setFont(QFont(f))

    def _setup_control_buttons(self):
        def _pick_button(panel, names, text_keywords):
            for n in names:
                w = getattr(self, n, None)
                if isinstance(w, QPushButton):
                    return w
            if panel is not None:
                for w in panel.findChildren(QPushButton):
                    txt = str(w.text())
                    if any(k in txt for k in text_keywords):
                        return w
            return None

        def _pick_line_edit(panel, names, hint_keywords):
            for n in names:
                w = getattr(self, n, None)
                if isinstance(w, QLineEdit):
                    return w
            if panel is not None:
                for w in panel.findChildren(QLineEdit):
                    blob = f"{w.placeholderText()} {w.toolTip()} {w.objectName()}".lower()
                    if any(k in blob for k in hint_keywords):
                        return w
            return None

        def _pick_label(panel, names, text_keywords):
            for n in names:
                w = getattr(self, n, None)
                if isinstance(w, QLabel):
                    return w
            if panel is not None:
                for w in panel.findChildren(QLabel):
                    txt = str(w.text())
                    if any(k in txt for k in text_keywords):
                        return w
            return None

        def _pick_checkbox(panel, names, text_keywords):
            for n in names:
                w = getattr(self, n, None)
                if isinstance(w, QCheckBox):
                    return w
            if panel is not None:
                for w in panel.findChildren(QCheckBox):
                    txt = str(w.text())
                    if any(k in txt for k in text_keywords):
                        return w
            return None

        panel = getattr(self, "frame_2", None)
        self.pushButton = _pick_button(panel, ["pushButton", "start_button", "work_start_button"], ["작업 시작", "시작"])
        start_btn = self.pushButton
        self._print_pos_button = _pick_button(
            panel,
            ["print_pos_button", "print_button", "position_print_button"],
            ["좌표 출력", "좌표", "출력"],
        )
        self._reset_button = _pick_button(panel, ["reset_button", "robot_reset_button"], ["리셋", "초기화"])
        self._home_button = _pick_button(panel, ["home_button", "move_home_button"], ["홈위치", "홈", "원점"])
        self._home_save_button = _pick_button(
            panel,
            ["home_button_2", "home_save_button", "save_home_button"],
            ["홈위치 저장", "홈 저장"],
        )
        self._robot_mode_button = _pick_button(
            panel,
            ["print_pos_button_2", "robot_mode_button", "mode_change_button"],
            ["모드변경", "모드 변경", "로봇모드"],
        )
        self._tool_change_button = _pick_button(
            panel,
            ["tool_change_button", "tcp_change_button"],
            ["툴 변경", "tcp 변경", "tool"],
        )

        start_btn = self.pushButton
        if start_btn is not None:
            try:
                start_btn.clicked.disconnect()
            except Exception:
                pass
            start_btn.clicked.connect(self.on_move_xyzabc_dialog)
        self._motion_group_box = getattr(self, "motion_group_box", None)
        self._motion_group_title = _pick_label(panel, ["motion_group_title"], ["기본 동작", "기본동작"])
        self._vision_group_box = getattr(self, "vision_group_box", None)
        self._vision_group_title = _pick_label(panel, ["vision_group_title"], ["비전 이동", "비전이동"])
        self._gripper_manual_box = getattr(self, "gripper_manual_box", None)
        self._gripper_range_title_label = _pick_label(
            panel,
            ["gripper_range_title_label"],
            ["그리퍼 사이 거리 입력", "그리퍼", "집게"],
        )
        self._gripper_range_value_label = _pick_label(
            panel,
            ["gripper_range_value_label"],
            ["허용 범위", "0.00 ~ 109.00"],
        )
        self._gripper_stroke_input = _pick_line_edit(
            panel,
            ["gripper_stroke_input", "gripper_input"],
            ["gripper", "그리퍼", "집게", "109.00"],
        )
        self._gripper_move_button = _pick_button(
            panel,
            ["gripper_move_button", "gripper_button"],
            ["그리퍼 이동", "집게 이동", "그리퍼"],
        )
        self._vision_move_button = _pick_button(
            panel,
            ["vision_move_button", "vision_coord_move_button"],
            ["비전 좌표 이동", "비전 이동", "좌표 이동"],
        )
        self._vision_z_margin_input = _pick_line_edit(
            panel,
            ["vision_z_margin_input", "z_margin_input", "vision_z_offset_input"],
            ["z", "옵셋", "offset", "vision"],
        )
        self._vision_z_title_label = _pick_label(
            panel,
            ["vision_z_title_label"],
            ["Z 옵셋 입력", "Z+ 옵셋", "옵셋"],
        )
        self._vision_z_range_label = _pick_label(
            panel,
            ["vision_z_range_label"],
            ["-300.0 ~ 300.0", "허용 범위"],
        )
        self._vision_dialog_toggle_button = _pick_button(
            panel,
            ["vision_dialog_toggle_button", "vision_click_toggle_button"],
            ["비전클릭이동", "비전좌표 클릭"],
        )
        self._vision_dialog_toggle_switch = _pick_checkbox(
            panel,
            ["vision_dialog_toggle_switch", "vision_click_dialog_switch"],
            ["비전클릭이동", "비전좌표 클릭"],
        )
        self._calibration_mode_switch = _pick_checkbox(
            panel,
            ["calibration_mode_switch", "vision_calibration_mode_switch"],
            ["캘리브레이션 모드", "calibration mode"],
        )
        self._calibration_mode_switch_2 = getattr(self, "calibration_mode_switch_2", None)
        self._calibration_transform_button = _pick_button(
            panel,
            ["calibration_transform_button", "vision_robot_transform_button"],
            ["비전/로봇 좌표변환", "좌표변환", "변환행렬"],
        )
        self._calibration_transform_button_2 = getattr(self, "calibration_transform_button_2", None)
        self._calibration_load_button = _pick_button(
            panel,
            ["calibration_load_button", "load_rotation_matrix_button"],
            ["회전행렬 불러오기", "행렬 불러오기", "불러오기"],
        )
        self._calibration_load_button_2 = getattr(self, "calibration_load_button_2", None)
        self._calibration_status_label = _pick_label(
            panel,
            ["calibration_status_label"],
            ["캘리브레이션", "calibration"],
        )
        self._calibration_status_label_2 = getattr(self, "calibration_status_label_2", None)
        self._calibration_matrix_file_label = _pick_label(
            panel,
            ["calibration_matrix_file_label"],
            ["행렬:", "matrix"],
        )
        self._calibration_matrix_file_label_2 = getattr(self, "calibration_matrix_file_label_2", None)

        if self._print_pos_button is not None:
            try:
                self._print_pos_button.clicked.disconnect()
            except Exception:
                pass
            self._print_pos_button.clicked.connect(self.on_move_joint_dialog)
        if self._reset_button is not None:
            self._reset_button.clicked.connect(self.on_reset_robot)
        if self._home_button is not None:
            self._home_button.clicked.connect(self.on_move_home)
        if self._home_save_button is not None:
            try:
                self._home_save_button.clicked.disconnect()
            except Exception:
                pass
            self._home_save_button.clicked.connect(self.on_save_home_position_dialog)
        if self._robot_mode_button is not None:
            try:
                self._robot_mode_button.clicked.disconnect()
            except Exception:
                pass
            self._robot_mode_button.clicked.connect(self.on_change_robot_mode_dialog)
        if self._tool_change_button is not None:
            try:
                self._tool_change_button.clicked.disconnect()
            except Exception:
                pass
            self._tool_change_button.clicked.connect(self.on_change_tool_dialog)
        if self._gripper_stroke_input is not None:
            validator = QDoubleValidator(0.0, 109.0, 2, self._gripper_stroke_input)
            validator.setNotation(QDoubleValidator.StandardNotation)
            self._gripper_stroke_input.setValidator(validator)
        if self._gripper_move_button is not None:
            self._gripper_move_button.clicked.connect(self.on_gripper_move)

        if self._vision_move_button is not None:
            self._vision_move_button.clicked.connect(self.on_move_to_last_vision_point)
        if self._vision_z_margin_input is not None:
            self._vision_z_margin_input.setText(f"{VISION_MOVE_Z_MARGIN_MM:.1f}")
            z_validator = QDoubleValidator(-1000.0, 3000.0, 1, self._vision_z_margin_input)
            z_validator.setNotation(QDoubleValidator.StandardNotation)
            self._vision_z_margin_input.setValidator(z_validator)
        if self._vision_dialog_toggle_button is not None:
            self._vision_dialog_toggle_button.hide()
        if self._vision_dialog_toggle_switch is not None:
            self._vision_dialog_toggle_switch.setChecked(bool(getattr(self, "_vision_click_dialog_enabled", True)))
            self._vision_dialog_toggle_switch.toggled.connect(self._on_vision_click_dialog_toggled)
        if self._calibration_mode_switch is not None:
            self._calibration_mode_switch.setChecked(bool(self._calibration_mode_enabled_1))
            self._calibration_mode_switch.toggled.connect(self._on_calibration_mode_toggled)
        if self._calibration_mode_switch_2 is not None:
            self._calibration_mode_switch_2.setChecked(bool(self._calibration_mode_enabled_2))
            self._calibration_mode_switch_2.toggled.connect(self._on_calibration_mode_toggled)
        if self._calibration_transform_button is not None:
            self._calibration_transform_button.clicked.connect(self.on_calibration_transform)
        if self._calibration_transform_button_2 is not None:
            self._calibration_transform_button_2.clicked.connect(self.on_calibration_transform)
        if self._calibration_load_button is not None:
            self._calibration_load_button.clicked.connect(self.on_calibration_load_matrix)
        if self._calibration_load_button_2 is not None:
            self._calibration_load_button_2.clicked.connect(self.on_calibration_load_matrix)
        if self._motion_group_box is not None:
            self._motion_group_box.lower()
        if self._gripper_manual_box is not None:
            self._gripper_manual_box.lower()
        if self._vision_group_box is not None:
            self._vision_group_box.lower()
        for w in [
            self._motion_group_title,
            self._vision_group_title,
            self._home_button,
            self._home_save_button,
            self._robot_mode_button,
            self._tool_change_button,
            self.pushButton,
            self._print_pos_button,
            self._reset_button,
            self._gripper_range_title_label,
            self._gripper_range_value_label,
            self._gripper_stroke_input,
            self._gripper_move_button,
            self._vision_dialog_toggle_switch,
            self._vision_z_title_label,
            self._vision_z_range_label,
            self._vision_z_margin_input,
            self._vision_move_button,
            self._calibration_mode_switch,
            self._calibration_mode_switch_2,
            self._calibration_transform_button,
            self._calibration_transform_button_2,
            self._calibration_load_button,
            self._calibration_load_button_2,
            self._calibration_matrix_file_label,
            self._calibration_matrix_file_label_2,
            self._calibration_status_label,
            self._calibration_status_label_2,
        ]:
            if w is not None:
                w.raise_()
        self._update_calibration_mode_ui()
        self._layout_control_buttons()

    def _setup_gripper_manual_controls(self):
        # UI widgets are now defined in developer_frontend.ui
        return

    def _layout_control_buttons(self):
        # Geometry is controlled by developer_frontend.ui; enforce only z-order here.
        if self._calibration_mode_switch is not None and self._calibration_mode_switch_2 is not None:
            try:
                g1 = self._calibration_mode_switch.geometry()
                g2 = self._calibration_mode_switch_2.geometry()
                # Keep switch_2 on the same row as switch_1.
                self._calibration_mode_switch_2.setGeometry(g2.x(), g1.y(), g2.width(), g1.height())
            except Exception:
                pass
        if self._motion_group_box is not None:
            self._motion_group_box.lower()
        if self._gripper_manual_box is not None:
            self._gripper_manual_box.lower()
        if self._vision_group_box is not None:
            self._vision_group_box.lower()
        for w in [
            self._motion_group_title,
            self._vision_group_title,
            self._home_button,
            self._home_save_button,
            self.pushButton,
            self._print_pos_button,
            self._reset_button,
            self._tool_change_button,
            self._gripper_range_title_label,
            self._gripper_range_value_label,
            self._gripper_stroke_input,
            self._gripper_move_button,
            self._vision_dialog_toggle_switch,
            self._vision_z_title_label,
            self._vision_z_range_label,
            self._vision_z_margin_input,
            self._vision_move_button,
            self._vision_dialog_toggle_button,
            self._calibration_mode_switch,
            self._calibration_mode_switch_2,
            self._calibration_transform_button,
            self._calibration_transform_button_2,
            self._calibration_load_button,
            self._calibration_load_button_2,
            self._calibration_matrix_file_label,
            self._calibration_matrix_file_label_2,
            self._calibration_status_label,
            self._calibration_status_label_2,
        ]:
            if w is not None:
                w.raise_()

    def _get_vision_move_z_margin_mm(self):
        default_v = float(VISION_MOVE_Z_MARGIN_MM)
        inp = getattr(self, "_vision_z_margin_input", None)
        if inp is None:
            self._vision_move_z_margin_mm = default_v
            return self._vision_move_z_margin_mm
        raw = inp.text().strip()
        if not raw:
            self._vision_move_z_margin_mm = default_v
            inp.setText(f"{default_v:.1f}")
            return self._vision_move_z_margin_mm
        try:
            v = float(raw)
        except Exception:
            v = default_v
        self._vision_move_z_margin_mm = v
        inp.setText(f"{v:.1f}")
        return self._vision_move_z_margin_mm

    def _toggle_vision_click_dialog(self):
        self._vision_click_dialog_enabled = not bool(self._vision_click_dialog_enabled)
        if hasattr(self, "_vision_dialog_toggle_button") and self._vision_dialog_toggle_button is not None:
            self._vision_dialog_toggle_button.setText(
                "비전클릭이동 ON" if self._vision_click_dialog_enabled else "비전클릭이동 OFF"
            )
        if hasattr(self, "_vision_dialog_toggle_switch") and self._vision_dialog_toggle_switch is not None:
            self._vision_dialog_toggle_switch.blockSignals(True)
            self._vision_dialog_toggle_switch.setChecked(bool(self._vision_click_dialog_enabled))
            self._vision_dialog_toggle_switch.blockSignals(False)
        self.append_log(
            f"[비전] 클릭 이동 다이얼로그: {'ON' if self._vision_click_dialog_enabled else 'OFF'}\n"
        )
        if YOLO_EXTERNAL_NODE and self.backend is not None:
            self._rebind_external_vision_bridge_for_mode()

    def _on_vision_click_dialog_toggled(self, checked):
        self._vision_click_dialog_enabled = bool(checked)
        self.append_log(
            f"[비전] 클릭 이동 다이얼로그: {'ON' if self._vision_click_dialog_enabled else 'OFF'}\n"
        )
        if YOLO_EXTERNAL_NODE and self.backend is not None:
            self._rebind_external_vision_bridge_for_mode()

    def on_change_robot_mode_dialog(self):
        if self.backend is None:
            self.append_log("[로봇모드] 백엔드 초기화 중입니다.\n")
            return
        if not hasattr(self.backend, "sync_robot_mode_once") or not hasattr(self.backend, "set_robot_mode"):
            self.append_log("[로봇모드] 백엔드가 모드 변경 기능을 지원하지 않습니다.\n")
            return

        ok_mode, mode_msg = self.backend.sync_robot_mode_once(force=False, timeout_sec=1.0)
        if not ok_mode:
            self.append_log(f"[로봇모드] 현재 모드 조회 실패: {mode_msg}\n")
        mode_value, _seen_at = self.backend.get_robot_mode_snapshot()
        cur_value = -1 if mode_value is None else int(mode_value)
        cur_text = {
            0: "메뉴얼모드",
            1: "오토모드",
            2: "측정모드",
        }.get(cur_value, "알수없음")

        dialog = QDialog(self)
        dialog.setWindowTitle("로봇 모드 변경")
        dialog.setModal(True)
        dialog.resize(500, 280)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(14)
        info = QLabel(f"현재 모드: {cur_text} ({cur_value})\n변경할 모드를 선택하세요.", dialog)
        layout.addWidget(info)
        layout.addSpacing(8)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(16)
        mode_row.addStretch(1)
        btn_manual = QPushButton("메뉴얼모드", dialog)
        btn_auto = QPushButton("오토모드", dialog)
        for b in (btn_manual, btn_auto):
            b.setCheckable(True)
            b.setMinimumSize(150, 58)
            b.setStyleSheet(
                "QPushButton { border: 1px solid #9aa0a6; border-radius: 6px; padding: 8px 12px; }"
                "QPushButton:checked { background: #1f6feb; color: #ffffff; border: 1px solid #1f6feb; }"
            )
        group = QButtonGroup(dialog)
        group.setExclusive(True)
        group.addButton(btn_manual, 0)
        group.addButton(btn_auto, 1)
        if cur_value == 1:
            btn_auto.setChecked(True)
        else:
            btn_manual.setChecked(True)
        mode_row.addWidget(btn_manual)
        mode_row.addWidget(btn_auto)
        mode_row.addStretch(1)
        layout.addStretch(1)
        layout.addLayout(mode_row)
        layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.setCenterButtons(False)

        def _on_accept():
            if group.checkedId() not in (0, 1):
                QMessageBox.warning(dialog, "로봇 모드 변경", "메뉴얼모드 또는 오토모드를 선택하세요.")
                return
            dialog.accept()

        buttons.accepted.connect(_on_accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QDialog.Accepted:
            self.append_log("[로봇모드] 변경 취소\n")
            return

        target = int(group.checkedId())
        target_name = "메뉴얼모드" if target == 0 else "오토모드"

        def _format_mode_result_details(raw_msg: str) -> str:
            text = str(raw_msg or "").strip()
            parts = [p.strip() for p in text.split(";") if p.strip()]
            set_part = "-"
            get_part = "-"
            extra_parts = []

            def _simplify_status(value: str) -> str:
                v = str(value or "").strip()
                up = v.upper()
                if up.startswith("OK"):
                    return "OK"
                if up.startswith("FAIL"):
                    if "(" in v and ")" in v:
                        try:
                            reason = v[v.find("(") + 1 : v.rfind(")")].strip()
                            if reason:
                                return reason
                        except Exception:
                            pass
                    reason = v[4:].strip()
                    return reason if reason else "실패"
                return v if v else "-"

            for p in parts:
                up = p.upper()
                if up.startswith("SET="):
                    set_part = _simplify_status(p.split("=", 1)[1].strip())
                elif up.startswith("GET="):
                    get_part = _simplify_status(p.split("=", 1)[1].strip())
                else:
                    extra_parts.append(p)
            lines = [f"- SET: {set_part}", f"- GET: {get_part}"]
            if extra_parts:
                lines.append(f"- 기타: {'; '.join(extra_parts)}")
            return "\n".join(lines)

        wait = QDialog(self)
        wait.setWindowTitle("로봇 모드 변경")
        wait.setWindowModality(Qt.WindowModal)
        wait.setModal(True)
        wait_layout = QVBoxLayout(wait)
        wait_layout.addWidget(QLabel("모드 변경 적용중입니다...\n모드 변경 확인중입니다...", wait))
        wait.setFixedSize(260, 90)
        wait.show()
        QApplication.processEvents()
        try:
            ok, msg = self.backend.set_robot_mode(target, timeout_sec=8.0)
        finally:
            wait.done(0)
            wait.hide()
            wait.deleteLater()
            QApplication.processEvents()
        if ok:
            self.append_log(f"[로봇모드] {target_name}({target})로 변경을 성공하였습니다. ({msg})\n")
            QMessageBox.information(
                self,
                "로봇 모드 변경",
                (
                    f"{target_name}({target})로 변경을 성공하였습니다.\n\n"
                    f"상세\n{_format_mode_result_details(msg)}"
                ),
                QMessageBox.Ok,
            )
        else:
            self.append_log(f"[로봇모드] {target_name}({target})로 변경을 실패하였습니다. ({msg})\n")
            QMessageBox.warning(
                self,
                "로봇 모드 변경",
                (
                    f"{target_name}({target})로 변경을 실패하였습니다.\n\n"
                    f"상세\n{_format_mode_result_details(msg)}"
                ),
                QMessageBox.Ok,
            )

    def on_change_tool_dialog(self):
        if self.backend is None:
            self.append_log("[툴변경] 백엔드 초기화 중입니다.\n")
            return
        if not hasattr(self.backend, "sync_robot_mode_once") or not hasattr(self.backend, "get_robot_mode_snapshot"):
            self.append_log("[툴변경] 백엔드가 로봇모드 조회 기능을 지원하지 않습니다.\n")
            return
        if not hasattr(self.backend, "sync_current_tcp_once") or not hasattr(self.backend, "set_current_tcp_name"):
            self.append_log("[툴변경] 백엔드가 TCP 변경 기능을 지원하지 않습니다.\n")
            return

        ok_mode, mode_msg = self.backend.sync_robot_mode_once(force=False, timeout_sec=1.0)
        if not ok_mode:
            self.append_log(f"[툴변경] 로봇모드 조회 실패: {mode_msg}\n")
            QMessageBox.warning(self, "툴 변경", f"로봇모드 조회 실패\n\n{mode_msg}")
            return

        mode_value, _mode_seen_at = self.backend.get_robot_mode_snapshot()
        try:
            mode_int = int(mode_value) if mode_value is not None else -1
        except Exception:
            mode_int = -1
        if mode_int != 0:
            self.append_log(f"[툴변경] 메뉴얼모드(0)에서만 변경할 수 있습니다. (현재 mode={mode_int})\n")
            QMessageBox.warning(
                self,
                "툴 변경",
                f"메뉴얼모드(0)에서만 진입할 수 있습니다.\n현재 모드: {mode_int}",
            )
            return

        ok_tcp, tcp_msg = self.backend.sync_current_tcp_once(force=True, timeout_sec=1.0)
        if not ok_tcp:
            self.append_log(f"[툴변경] 현재 TCP 조회 실패: {tcp_msg}\n")
        cur_name, _cur_seen = self.backend.get_current_tcp_snapshot() if hasattr(self.backend, "get_current_tcp_snapshot") else ("", None)
        current_display = "flange" if str(cur_name or "").strip() == "" else str(cur_name).strip()

        dialog = QDialog(self)
        dialog.setWindowTitle("툴 변경")
        dialog.setModal(True)
        dialog.resize(520, 280)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(14)

        info = QLabel(
            f"현재 툴(TCP): {current_display}\n변경할 항목을 선택하세요.\n(메뉴얼모드에서만 변경 가능)",
            dialog,
        )
        layout.addWidget(info)
        layout.addSpacing(8)

        options = [("flange", ""), ("tool_gripper", "tool_gripper"), ("tool_checkerboard", "tool_checkerboard")]
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch(1)
        group = QButtonGroup(dialog)
        group.setExclusive(True)
        buttons = {}
        for idx, (label, _target) in enumerate(options):
            btn = QPushButton(label, dialog)
            btn.setCheckable(True)
            btn.setMinimumSize(140, 54)
            btn.setStyleSheet(
                "QPushButton { border: 1px solid #9aa0a6; border-radius: 6px; padding: 8px 12px; }"
                "QPushButton:checked { background: #1f6feb; color: #ffffff; border: 1px solid #1f6feb; }"
            )
            group.addButton(btn, idx)
            buttons[label] = btn
            btn_row.addWidget(btn)
        btn_row.addStretch(1)
        layout.addStretch(1)
        layout.addLayout(btn_row)
        layout.addStretch(1)

        if current_display in buttons:
            buttons[current_display].setChecked(True)
        else:
            buttons["flange"].setChecked(True)

        buttons_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons_box.setCenterButtons(False)
        buttons_box.accepted.connect(dialog.accept)
        buttons_box.rejected.connect(dialog.reject)
        layout.addWidget(buttons_box)

        if dialog.exec_() != QDialog.Accepted:
            self.append_log("[툴변경] 변경 취소\n")
            return

        selected_id = int(group.checkedId())
        if selected_id < 0 or selected_id >= len(options):
            QMessageBox.warning(self, "툴 변경", "변경할 툴을 선택하세요.")
            return

        selected_label, target_name = options[selected_id]
        wait = QDialog(self)
        wait.setWindowTitle("툴 변경")
        wait.setWindowModality(Qt.WindowModal)
        wait.setModal(True)
        wait_layout = QVBoxLayout(wait)
        wait_layout.addWidget(QLabel("툴(TCP) 적용중입니다...\n결과 확인중입니다...", wait))
        wait.setFixedSize(280, 90)
        wait.show()
        QApplication.processEvents()
        try:
            ok_set, msg_set = self.backend.set_current_tcp_name(target_name, timeout_sec=5.0)
            self.backend.sync_current_tcp_once(force=True, timeout_sec=1.0)
        finally:
            wait.done(0)
            wait.hide()
            wait.deleteLater()
            QApplication.processEvents()

        now_name, _now_seen = self.backend.get_current_tcp_snapshot() if hasattr(self.backend, "get_current_tcp_snapshot") else ("", None)
        now_display = "flange" if str(now_name or "").strip() == "" else str(now_name).strip()
        self._set_current_tool_text(now_name)

        if ok_set:
            self.append_log(f"[툴변경] 적용 성공: 요청={selected_label}, 현재={now_display} ({msg_set})\n")
            QMessageBox.information(
                self,
                "툴 변경",
                f"툴(TCP) 적용 성공\n\n요청: {selected_label}\n현재: {now_display}\n\n상세: {msg_set}",
                QMessageBox.Ok,
            )
        else:
            self.append_log(f"[툴변경] 적용 실패: 요청={selected_label}, 현재={now_display} ({msg_set})\n")
            QMessageBox.warning(
                self,
                "툴 변경",
                f"툴(TCP) 적용 실패\n\n요청: {selected_label}\n현재: {now_display}\n\n상세: {msg_set}",
                QMessageBox.Ok,
            )

    def _set_manual_mode_and_checkerboard_tcp(self, parent=None):
        if self.backend is None:
            return False, "백엔드 초기화 중입니다."
        if not hasattr(self.backend, "sync_robot_mode_once") or not hasattr(self.backend, "get_robot_mode_snapshot"):
            return False, "백엔드가 로봇모드 조회 기능을 지원하지 않습니다."
        if not hasattr(self.backend, "set_robot_mode"):
            return False, "백엔드가 로봇모드 변경 기능을 지원하지 않습니다."
        if not hasattr(self.backend, "sync_current_tcp_once") or not hasattr(self.backend, "set_current_tcp_name"):
            return False, "백엔드가 TCP 변경 기능을 지원하지 않습니다."

        holder = parent if parent is not None else self
        wait = QDialog(holder)
        wait.setWindowTitle("모드/TCP 설정")
        wait.setWindowModality(Qt.WindowModal)
        wait.setModal(True)
        wait_layout = QVBoxLayout(wait)
        wait_layout.addWidget(QLabel("수동모드 및 체커보드 TCP 적용중입니다...", wait))
        wait.setFixedSize(320, 86)
        wait.show()
        QApplication.processEvents()
        try:
            ok_mode_sync, msg_mode_sync = self.backend.sync_robot_mode_once(force=True, timeout_sec=1.2)
            if not ok_mode_sync:
                return False, f"로봇모드 조회 실패: {msg_mode_sync}"

            mode_value, _mode_seen_at = self.backend.get_robot_mode_snapshot()
            try:
                mode_int = int(mode_value) if mode_value is not None else -1
            except Exception:
                mode_int = -1

            mode_detail = "SKIP(이미 메뉴얼모드)"
            if mode_int != 0:
                ok_mode_set, msg_mode_set = self.backend.set_robot_mode(0, timeout_sec=8.0)
                mode_detail = msg_mode_set
                if not ok_mode_set:
                    return False, f"수동모드 전환 실패: {msg_mode_set}"

            ok_tcp_set, msg_tcp_set = self.backend.set_current_tcp_name("tool_checkerboard", timeout_sec=5.0)
            if not ok_tcp_set:
                return False, f"체커보드 TCP 적용 실패: {msg_tcp_set}"

            ok_tcp_sync, msg_tcp_sync = self.backend.sync_current_tcp_once(force=True, timeout_sec=1.2)
            if not ok_tcp_sync:
                return False, f"TCP 재조회 실패: {msg_tcp_sync}"
            tcp_name, _tcp_seen_at = self.backend.get_current_tcp_snapshot()
            tcp_name = str(tcp_name or "").strip()
            if tcp_name != "tool_checkerboard":
                return False, f"TCP 검증 실패: 현재 tcp='{tcp_name or 'flange'}'"

            self._refresh_current_tool_label_once(log_fail=False)
            return True, f"MODE={mode_detail}; TCP={msg_tcp_set}"
        finally:
            wait.done(0)
            wait.hide()
            wait.deleteLater()
            QApplication.processEvents()

    def _on_calibration_mode_toggled(self, checked):
        sender = self.sender()
        src_panel = 1
        if sender is getattr(self, "_calibration_mode_switch_2", None):
            checked = bool(self._calibration_mode_switch_2.isChecked())
            self._calibration_mode_enabled_2 = bool(checked)
            src_panel = 2
        elif hasattr(self, "_calibration_mode_switch") and self._calibration_mode_switch is not None:
            checked = bool(self._calibration_mode_switch.isChecked())
            self._calibration_mode_enabled_1 = bool(checked)
            src_panel = 1
        self._calibration_mode_enabled = bool(self._calibration_mode_enabled_1 or self._calibration_mode_enabled_2)
        self._clear_calibration_panel_state(src_panel)
        self._clear_vision_view_data(src_panel)
        if src_panel == 2:
            self._vision_state_text_2 = "전환중"
        else:
            self._vision_state_text = "전환중"
        now = time.monotonic()
        self._advance_vision_stream_token(src_panel)
        if src_panel == 2:
            self._vision_drop_frames_until_2 = now + 1.2
        else:
            self._vision_drop_frames_until_1 = now + 1.2
        if (not checked) and self._vision_to_robot_affine is None:
            self._try_load_calibration_matrix_on_startup()
        self._mode_switch_grace_until = now + MODE_SWITCH_GRACE_SEC
        self._rebind_external_vision_bridge_for_mode()
        self._update_calibration_mode_ui()
        base_enabled = False
        cache = self._robot_controls_enabled_cache
        if isinstance(cache, tuple):
            base_enabled = bool(cache[0])
        elif cache is not None:
            base_enabled = bool(cache)
        self._set_robot_controls_enabled(base_enabled)
        self.append_log(f"[캘리브레이션{src_panel}] 모드: {'ON' if bool(checked) else 'OFF'}\n")
        self._save_vision_serial_settings()

    def _tick_calibration_status_blink(self):
        self._calib_status_blink_on = not bool(getattr(self, "_calib_status_blink_on", False))
        if bool(getattr(self, "_calibration_mode_enabled_1", False) or getattr(self, "_calibration_mode_enabled_2", False)):
            self._update_calibration_mode_ui()

    def _is_calibration_points_ready(self, panel_index=None):
        if panel_index is None:
            panel = 0
        else:
            try:
                panel = int(panel_index)
            except Exception:
                panel = 0
        if panel == 2:
            data = getattr(self, "_calib_last_points_uvz_mm_2", None)
        elif panel == 1:
            data = getattr(self, "_calib_last_points_uvz_mm_1", None)
        else:
            data = getattr(self, "_calib_last_points_uvz_mm_1", None)
            if not isinstance(data, dict):
                data = getattr(self, "_calib_last_points_uvz_mm_2", None)
        if self._is_calibration_full_data_ready(panel_index=panel if panel in (1, 2) else None):
            return True
        if not isinstance(data, dict):
            return False
        center = data.get("center")
        if center is None or len(center) < 3:
            return False
        try:
            return np.isfinite(float(center[2]))
        except Exception:
            return False

    def _is_calibration_board_detected(self, panel_index=None):
        try:
            panel = int(panel_index) if panel_index is not None else 1
        except Exception:
            panel = 1
        if panel == 2:
            pts = getattr(self, "_calib_last_grid_pts_2", None)
            seen_at = getattr(self, "_calib_last_points_at_2", None)
        else:
            pts = getattr(self, "_calib_last_grid_pts_1", None)
            seen_at = getattr(self, "_calib_last_points_at_1", None)
        if pts is None:
            return False
        if seen_at is None:
            return True
        return (time.monotonic() - float(seen_at)) <= max(0.2, float(CALIB_DETECTION_HOLD_SEC) + 0.2)

    def _is_calibration_full_data_ready(self, panel_index=None):
        try:
            panel = int(panel_index) if panel_index is not None else 1
        except Exception:
            panel = 1
        if panel == 2:
            ready = bool(getattr(self, "_calib_full_ready_2", False))
            seen_at = getattr(self, "_calib_last_points_at_2", None)
        else:
            ready = bool(getattr(self, "_calib_full_ready_1", False))
            seen_at = getattr(self, "_calib_last_points_at_1", None)
        if not ready:
            return False
        if seen_at is None:
            return True
        return (time.monotonic() - float(seen_at)) <= max(0.2, float(CALIB_DETECTION_HOLD_SEC) + 0.2)

    def _update_calibration_mode_ui(self):
        calib_on_1 = bool(getattr(self, "_calibration_mode_enabled_1", False))
        calib_on_2 = bool(getattr(self, "_calibration_mode_enabled_2", False))
        calib_on = bool(calib_on_1 or calib_on_2)
        self._calibration_mode_enabled = calib_on
        calib_ready_1 = bool(self._is_calibration_points_ready(panel_index=1))
        calib_ready_2 = bool(self._is_calibration_points_ready(panel_index=2))
        calib_detected_1 = bool(self._is_calibration_full_data_ready(panel_index=1))
        calib_detected_2 = bool(self._is_calibration_full_data_ready(panel_index=2))
        if hasattr(self, "_calibration_mode_switch") and self._calibration_mode_switch is not None:
            self._calibration_mode_switch.blockSignals(True)
            self._calibration_mode_switch.setChecked(calib_on_1)
            self._calibration_mode_switch.setText(f"켈리브레이션활성화 ({'ON' if calib_on_1 else 'OFF'})")
            self._calibration_mode_switch.setToolTip("켈리브레이션 활성화 스위치")
            self._calibration_mode_switch.blockSignals(False)
        if hasattr(self, "_calibration_mode_switch_2") and self._calibration_mode_switch_2 is not None:
            self._calibration_mode_switch_2.blockSignals(True)
            self._calibration_mode_switch_2.setChecked(calib_on_2)
            self._calibration_mode_switch_2.setText(f"켈리브레이션활성화 ({'ON' if calib_on_2 else 'OFF'})")
            self._calibration_mode_switch_2.setToolTip("켈리브레이션 활성화 스위치")
            self._calibration_mode_switch_2.blockSignals(False)
        if hasattr(self, "_calibration_transform_button") and self._calibration_transform_button is not None:
            self._calibration_transform_button.setVisible(calib_on_1)
            self._calibration_transform_button.setEnabled(calib_on_1 and bool(self._top_status_enabled.get("robot", True)))
        if hasattr(self, "_calibration_transform_button_2") and self._calibration_transform_button_2 is not None:
            self._calibration_transform_button_2.setVisible(calib_on_2)
            self._calibration_transform_button_2.setEnabled(calib_on_2 and bool(self._top_status_enabled.get("robot", True)))
        if hasattr(self, "_calibration_load_button") and self._calibration_load_button is not None:
            self._calibration_load_button.setVisible(calib_on_1)
            self._calibration_load_button.setEnabled(calib_on_1)
        if hasattr(self, "_calibration_load_button_2") and self._calibration_load_button_2 is not None:
            self._calibration_load_button_2.setVisible(calib_on_2)
            self._calibration_load_button_2.setEnabled(calib_on_2)
        if hasattr(self, "_calibration_matrix_file_label") and self._calibration_matrix_file_label is not None:
            base = os.path.basename(self._calib_matrix_path_1) if self._calib_matrix_path_1 else "(없음)"
            self._calibration_matrix_file_label.setText(f"현재 적용행렬 : {base}")
            self._calibration_matrix_file_label.setStyleSheet("color: #666666; font-size: 8pt;")
        if hasattr(self, "_calibration_matrix_file_label_2") and self._calibration_matrix_file_label_2 is not None:
            base = os.path.basename(self._calib_matrix_path_2) if self._calib_matrix_path_2 else "(없음)"
            self._calibration_matrix_file_label_2.setText(f"현재 적용행렬 : {base}")
            self._calibration_matrix_file_label_2.setStyleSheet("color: #666666; font-size: 8pt;")
        if hasattr(self, "_calibration_status_label") and self._calibration_status_label is not None:
            self._calibration_status_label.setVisible(calib_on_1)
            if calib_on_1:
                status = "[실패] 켈리브레이션 보드 인식 실패"
                if calib_detected_1:
                    status = "[성공] 켈리브레이션 보드 인식 성공"
                self._calibration_status_label.setText(status)
                if calib_detected_1:
                    self._calibration_status_label.setStyleSheet(
                        "color: #2e7d32; font-size: 10pt; font-weight: 800;"
                    )
                else:
                    blink_on = bool(getattr(self, "_calib_status_blink_on", False))
                    self._calibration_status_label.setStyleSheet(
                        "color: #d32f2f; font-size: 10pt; font-weight: 800;"
                        if blink_on
                        else "color: #f9a825; font-size: 10pt; font-weight: 800;"
                    )
            else:
                self._calibration_status_label.setText("")
                self._calibration_status_label.setStyleSheet(
                    "color: #5f5a1e; font-size: 9pt; font-weight: 700;"
                )
        if hasattr(self, "_calibration_status_label_2") and self._calibration_status_label_2 is not None:
            self._calibration_status_label_2.setVisible(calib_on_2)
            if calib_on_2:
                status = "[실패] 켈리브레이션 보드 인식 실패"
                if calib_detected_2:
                    status = "[성공] 켈리브레이션 보드 인식 성공"
                self._calibration_status_label_2.setText(status)
                if calib_detected_2:
                    self._calibration_status_label_2.setStyleSheet(
                        "color: #2e7d32; font-size: 10pt; font-weight: 800;"
                    )
                else:
                    blink_on = bool(getattr(self, "_calib_status_blink_on", False))
                    self._calibration_status_label_2.setStyleSheet(
                        "color: #d32f2f; font-size: 10pt; font-weight: 800;"
                        if blink_on
                        else "color: #f9a825; font-size: 10pt; font-weight: 800;"
                    )
            else:
                self._calibration_status_label_2.setText("")
                self._calibration_status_label_2.setStyleSheet(
                    "color: #5f5a1e; font-size: 9pt; font-weight: 700;"
                )

        # yolo_view 배경색은 프레임 렌더링/미수신 상태에서만 관리한다.
        # (캘리브레이션 상태 갱신 타이머와 충돌해 깜박이는 현상 방지)

        # Normal vision move UI is hidden in calibration mode.
        for w in [
            getattr(self, "_vision_dialog_toggle_switch", None),
            getattr(self, "_vision_z_title_label", None),
            getattr(self, "_vision_z_range_label", None),
            getattr(self, "_vision_z_margin_input", None),
            getattr(self, "_vision_move_button", None),
        ]:
            if w is not None:
                w.setVisible(True)
        if hasattr(self, "_vision_dialog_toggle_button") and self._vision_dialog_toggle_button is not None:
            self._vision_dialog_toggle_button.hide()

    def _parse_point_file_sections(self, file_path):
        sections = {}
        current = "default"
        sections[current] = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    head = line.lstrip("#").strip().lower()
                    if head.startswith("robot"):
                        current = "robot"
                    elif head.startswith("vision"):
                        current = "vision"
                    elif head.startswith("rotationmat"):
                        current = "rotationmat"
                    else:
                        current = "default"
                    sections.setdefault(current, {})
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = k.strip().lower()
                vals = [x.strip() for x in v.split(",")]
                if len(vals) < 1:
                    continue
                try:
                    numbers = [float(x) for x in vals]
                except Exception:
                    continue
                sections.setdefault(current, {})[key] = numbers
        return sections

    def _points_dict_to_array(self, points_dict):
        names = ["p1", "p2", "p3", "p4", "p5"]
        arr = []
        for n in names:
            if n not in points_dict:
                return None
            arr.append(points_dict[n][:3])
        return np.asarray(arr, dtype=np.float64)

    def _compute_rigid_uvz_to_xyz(self, uvz_arr, xyz_arr, require_intrinsics=False, panel_index: int = 1):
        if uvz_arr is None or xyz_arr is None:
            return None, "P1~P5 데이터가 부족합니다."
        if uvz_arr.shape != xyz_arr.shape or uvz_arr.shape[0] < 4:
            return None, "비전/로봇 포인트 형식이 맞지 않습니다."
        if not np.isfinite(uvz_arr).all():
            return None, "비전 포인트에 유효하지 않은 깊이(Z)가 있습니다."
        if not np.isfinite(xyz_arr).all():
            return None, "로봇 포인트 파일에 유효하지 않은 값이 있습니다."

        src_uvz = np.asarray(uvz_arr, dtype=np.float64)
        src_rows = []
        for row in src_uvz:
            cam_xyz = self._uvz_to_camera_xyz_mm(
                row[0], row[1], row[2], require_intrinsics=require_intrinsics, panel_index=panel_index
            )
            if cam_xyz is None:
                return None, "camera_info 미수신: Eye-to-Hand 계산을 위해 카메라 내부파라미터가 필요합니다."
            src_rows.append(cam_xyz)
        src = np.asarray(src_rows, dtype=np.float64)
        dst = np.asarray(xyz_arr, dtype=np.float64)

        src_c = np.mean(src, axis=0)
        dst_c = np.mean(dst, axis=0)
        src0 = src - src_c
        dst0 = dst - dst_c

        # Kabsch (column-vector) -> row-vector 변환으로 저장
        h = src0.T @ dst0
        u, _, vt = np.linalg.svd(h)
        r_col = vt.T @ u.T
        if np.linalg.det(r_col) < 0:
            vt[-1, :] *= -1.0
            r_col = vt.T @ u.T
        r_row = r_col.T
        t_row = dst_c - (src_c @ r_row)

        pred = (src @ r_row) + t_row
        err = np.linalg.norm(pred - xyz_arr, axis=1)
        rmse = float(np.sqrt(np.mean(err ** 2)))

        affine = np.vstack([r_row, t_row.reshape(1, 3)])
        return (affine, r_row, t_row, rmse), None

    def _compute_rigid_xyz_to_xyz(self, src_xyz_arr, dst_xyz_arr):
        if src_xyz_arr is None or dst_xyz_arr is None:
            return None, "소스/타겟 XYZ 데이터가 부족합니다."
        src = np.asarray(src_xyz_arr, dtype=np.float64)
        dst = np.asarray(dst_xyz_arr, dtype=np.float64)
        if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 4:
            return None, "XYZ 포인트 형식이 맞지 않거나 데이터 개수가 부족합니다."
        if not np.isfinite(src).all():
            return None, "비전 XYZ 포인트에 유효하지 않은 값이 있습니다."
        if not np.isfinite(dst).all():
            return None, "로봇 XYZ 포인트에 유효하지 않은 값이 있습니다."

        src_c = np.mean(src, axis=0)
        dst_c = np.mean(dst, axis=0)
        src0 = src - src_c
        dst0 = dst - dst_c
        h = src0.T @ dst0
        u, _, vt = np.linalg.svd(h)
        r_col = vt.T @ u.T
        if np.linalg.det(r_col) < 0:
            vt[-1, :] *= -1.0
            r_col = vt.T @ u.T
        r_row = r_col.T
        t_row = dst_c - (src_c @ r_row)
        pred = (src @ r_row) + t_row
        err = np.linalg.norm(pred - dst, axis=1)
        rmse = float(np.sqrt(np.mean(err ** 2)))
        affine = np.vstack([r_row, t_row.reshape(1, 3)])
        return (affine, r_row, t_row, rmse), None

    def _extract_calib_grid_camera_xyz(self, panel_index: int = 1):
        panel = 2 if int(panel_index) == 2 else 1
        cached_xyz = self._calib_grid_camera_xyz_2 if panel == 2 else self._calib_grid_camera_xyz_1
        if cached_xyz is not None:
            try:
                arr = np.asarray(cached_xyz, dtype=np.float64).reshape(-1, 3)
                if arr.size > 0 and np.isfinite(arr).all():
                    return arr, None
            except Exception:
                pass
        pts_grid = self._calib_last_grid_pts_2 if panel == 2 else self._calib_last_grid_pts_1
        if pts_grid is None:
            return None, "체커보드 그리드 데이터가 없습니다."
        try:
            grid = np.asarray(pts_grid, dtype=np.float64)
        except Exception:
            return None, "체커보드 그리드 데이터 파싱 실패"
        if grid.ndim != 3 or grid.shape[2] < 2 or grid.shape[0] < 2 or grid.shape[1] < 2:
            return None, "체커보드 그리드 형식이 올바르지 않습니다."

        cam_rows = []
        bad = []
        flat = grid.reshape(-1, 2)
        for idx, pt in enumerate(flat):
            u_f = float(pt[0])
            v_f = float(pt[1])
            z_m = self._depth_m_from_image_coord(int(round(u_f)), int(round(v_f)), panel_index=panel)
            if z_m is None or (not np.isfinite(z_m)) or float(z_m) <= 0.0:
                bad.append(f"g{idx + 1}")
                continue
            z_mm = float(z_m) * 1000.0
            cxyz = self._uvz_to_camera_xyz_mm(u_f, v_f, z_mm, require_intrinsics=True, panel_index=panel)
            if cxyz is None:
                bad.append(f"g{idx + 1}")
                continue
            cxyz = np.asarray(cxyz, dtype=np.float64).reshape(3,)
            if not np.isfinite(cxyz).all():
                bad.append(f"g{idx + 1}")
                continue
            cam_rows.append(cxyz)
        total = int(flat.shape[0])
        if len(cam_rows) != total:
            sample = ", ".join(bad[:8])
            if len(bad) > 8:
                sample += ", ..."
            return None, f"체커보드 전체 XYZ 취득 실패: {len(bad)}/{total}개 깊이 미수신 또는 변환 실패({sample})"
        return np.asarray(cam_rows, dtype=np.float64), None

    def _robust_center_from_grid_xyz(self, grid_xyz):
        pts = np.asarray(grid_xyz, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 4:
            return None, 0
        if not np.isfinite(pts).all():
            return None, 0
        # 전체 코너를 최대한 활용하되, depth 튐(outlier) 영향은 줄인다.
        med = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - med.reshape(1, 3), axis=1)
        mad = np.median(np.abs(dist - np.median(dist)))
        if not np.isfinite(mad) or mad <= 1e-9:
            inlier = pts
        else:
            thr = max(5.0, 3.5 * 1.4826 * mad)
            inlier = pts[dist <= thr]
            if inlier.shape[0] < max(4, int(0.5 * pts.shape[0])):
                inlier = pts
        center = np.mean(inlier, axis=0)
        if not np.isfinite(center).all():
            return None, int(inlier.shape[0])
        return center.astype(np.float64), int(inlier.shape[0])

    def _save_calibration_file(self, out_path, vision_uvz_arr, robot_xyz_arr, affine, rmse, r_row=None, t_row=None, panel_index: int = 1):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        names = ["p1", "p2", "p3", "p4", "p5"]
        aff = np.asarray(affine, dtype=np.float64)
        if r_row is None and aff.shape == (4, 3):
            r_row = aff[:3, :]
        if t_row is None and aff.shape == (4, 3):
            t_row = aff[3, :]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("#VisionUVZ (u,v,z_mm)\n")
            for i, n in enumerate(names):
                u, v, z = vision_uvz_arr[i]
                f.write(f"{n}={u:.3f},{v:.3f},{z:.3f}\n")
            f.write("\n#Vision (x,y,z_mm)\n")
            for i, n in enumerate(names):
                u, v, z = vision_uvz_arr[i]
                cxyz = self._uvz_to_camera_xyz_mm(u, v, z, require_intrinsics=False, panel_index=panel_index)
                if cxyz is None:
                    f.write(f"{n}=nan,nan,nan\n")
                else:
                    x, y, zz = cxyz
                    f.write(f"{n}={x:.3f},{y:.3f},{zz:.3f}\n")
            f.write("\n#Robot (x,y,z_mm)\n")
            for i, n in enumerate(names):
                x, y, z = robot_xyz_arr[i]
                f.write(f"{n}={x:.3f},{y:.3f},{z:.3f}\n")
            f.write("\n#RotationMat (rigid: R(3x3), t(1x3), row-vector form)\n")
            if r_row is not None:
                r = np.asarray(r_row, dtype=np.float64).reshape(3, 3)
                for i in range(3):
                    f.write(f"r{i}={r[i,0]:.12g},{r[i,1]:.12g},{r[i,2]:.12g}\n")
            if t_row is not None:
                t = np.asarray(t_row, dtype=np.float64).reshape(3,)
                f.write(f"t={t[0]:.12g},{t[1]:.12g},{t[2]:.12g}\n")
            # backward compatibility
            for r in range(4):
                f.write(f"m{r}={aff[r,0]:.12g},{aff[r,1]:.12g},{aff[r,2]:.12g}\n")
            f.write(f"rmse={rmse:.6f}\n")

    def _load_affine_from_file(self, file_path):
        sections = self._parse_point_file_sections(file_path)
        rot = sections.get("rotationmat", {})
        if all((f"r{i}" in rot) for i in range(3)) and ("t" in rot):
            try:
                r_row = np.asarray([rot["r0"][:3], rot["r1"][:3], rot["r2"][:3]], dtype=np.float64)
                t_row = np.asarray(rot["t"][:3], dtype=np.float64).reshape(1, 3)
                mat = np.vstack([r_row, t_row])
                rmse = None
                if "rmse" in rot and len(rot["rmse"]) > 0:
                    try:
                        rmse = float(rot["rmse"][0])
                    except Exception:
                        rmse = None
                if rmse is None:
                    base = sections.get("default", {})
                    if "rmse" in base and len(base["rmse"]) > 0:
                        try:
                            rmse = float(base["rmse"][0])
                        except Exception:
                            rmse = None
                return mat, rmse
            except Exception:
                pass
        rows = []
        for i in range(4):
            k = f"m{i}"
            if k in rot:
                rows.append(rot[k][:3])
        if len(rows) == 4:
            mat = np.asarray(rows, dtype=np.float64)
            rmse = None
            if "rmse" in rot and len(rot["rmse"]) > 0:
                try:
                    rmse = float(rot["rmse"][0])
                except Exception:
                    rmse = None
            if rmse is None:
                base = sections.get("default", {})
                if "rmse" in base and len(base["rmse"]) > 0:
                    try:
                        rmse = float(base["rmse"][0])
                    except Exception:
                        rmse = None
            return mat, rmse
        return None, None

    def _save_active_calibration_path(self, src_path, panel_index: int = 1):
        try:
            if not src_path:
                return
            resolved_path = _resolve_repo_file(src_path, fallback_dirs=(CALIB_DIR, CALIB_ROTMAT_DIR))
            stored_path = _project_relative_path(resolved_path or src_path)
            rows = self._load_parameter_rows()
            panel = 2 if int(panel_index) == 2 else 1
            if panel == 2:
                rows["active_calibration_path_2"] = [stored_path]
            else:
                rows["active_calibration_path_1"] = [stored_path]
                # backward compatibility
                rows["active_calibration_path"] = [stored_path]
            self._save_parameter_rows(rows)
            if os.path.isfile(LEGACY_CALIB_ACTIVE_PATH_FILE):
                try:
                    os.remove(LEGACY_CALIB_ACTIVE_PATH_FILE)
                except Exception:
                    pass
        except Exception:
            return

    def on_calibration_load_matrix(self):
        sender = self.sender()
        panel = 2 if sender is getattr(self, "_calibration_load_button_2", None) else 1
        start_dir = CALIB_ROTMAT_DIR if os.path.isdir(CALIB_ROTMAT_DIR) else CALIB_DIR
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"캘리브레이션 행렬 불러오기 (비전{panel})",
            start_dir,
            "Text Files (*.txt);;All Files (*)",
        )
        if not file_path:
            return
        mat, rmse = self._load_affine_from_file(file_path)
        if mat is None:
            self.append_log(f"[캘리브레이션{panel}] 행렬 불러오기 실패: {file_path}\n")
            QMessageBox.warning(self, "행렬 불러오기", "선택한 파일에서 변환행렬을 읽지 못했습니다.")
            return
        self._apply_vision_to_robot_affine(mat, rmse=rmse, path=file_path, panel_index=panel)
        self.append_log(f"[캘리브레이션{panel}] 행렬 불러오기 성공: {file_path}\n")

    def on_calibration_transform(self):
        sender = self.sender()
        panel = 2 if sender is getattr(self, "_calibration_transform_button_2", None) else 1
        points = self._calib_last_points_uvz_mm_2 if panel == 2 else self._calib_last_points_uvz_mm_1
        vision_uvz_arr = self._points_dict_to_array(points)
        if vision_uvz_arr is None:
            self.append_log(f"[캘리브레이션{panel}] P1~P5 데이터가 없어 좌표변환을 수행할 수 없습니다.\n")
            QMessageBox.warning(self, "좌표변환", "P1~P5 데이터가 준비되지 않았습니다.")
            return
        result, err = self._compute_rigid_uvz_to_xyz(
            vision_uvz_arr,
            np.asarray(ROBOT_CALIB_XYZ_MM, dtype=np.float64),
            require_intrinsics=True,
            panel_index=panel,
        )
        if result is None:
            self.append_log(f"[캘리브레이션{panel}] 좌표변환 실패: {err}\n")
            QMessageBox.warning(self, "좌표변환", str(err))
            return
        aff, r_row, t_row, rmse = result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(CALIB_ROTMAT_DIR, f"calib_matrix_panel{panel}_{timestamp}.txt")
        try:
            self._save_calibration_file(
                out_path,
                vision_uvz_arr,
                np.asarray(ROBOT_CALIB_XYZ_MM, dtype=np.float64),
                aff,
                rmse,
                r_row=r_row,
                t_row=t_row,
                panel_index=panel,
            )
        except Exception as e:
            self.append_log(f"[캘리브레이션{panel}] 행렬 저장 실패: {e}\n")
            QMessageBox.warning(self, "좌표변환", f"행렬 저장 실패: {e}")
            return
        self._apply_vision_to_robot_affine(aff, rmse=rmse, path=out_path, panel_index=panel)
        self.append_log(f"[캘리브레이션{panel}] 좌표변환 완료 (RMSE={rmse:.3f}mm): {out_path}\n")

    def _load_parameter_rows(self):
        rows = {}
        if not os.path.isfile(PARAM_FILE):
            return rows
        try:
            with open(PARAM_FILE, "r", encoding="utf-8", newline="") as f:
                for row in csv.reader(f):
                    if not row:
                        continue
                    key = str(row[0]).strip()
                    if not key or key.lower() == "name":
                        continue
                    rows[key] = [str(v).strip() for v in row[1:] if str(v).strip() != ""]
        except Exception:
            return {}
        return rows

    def _save_parameter_rows(self, rows):
        try:
            os.makedirs(PARAM_DIR, exist_ok=True)
            with open(PARAM_FILE, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "v1", "v2", "v3", "v4", "v5", "v6"])
                for key in sorted(rows.keys()):
                    vals = rows.get(key, [])
                    if not isinstance(vals, (list, tuple)):
                        vals = [str(vals)]
                    writer.writerow([key, *list(vals)])
        except Exception:
            return

    def _normalize_serial_text(self, value):
        s = str(value or "").strip()
        if not s:
            return ""
        if s.startswith("_"):
            s = s[1:]
        return s

    def _discover_connected_camera_serials(self):
        serials = []
        try:
            proc = subprocess.run(
                ["rs-enumerate-devices", "-s"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2.5,
                check=False,
            )
            text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            for line in text.splitlines():
                low = line.lower()
                if "serial number" not in low:
                    continue
                if ":" not in line:
                    continue
                raw = line.split(":", 1)[1].strip()
                serial = self._normalize_serial_text(raw)
                if serial and serial not in serials:
                    serials.append(serial)
        except Exception:
            pass
        return serials

    def _load_vision_serial_settings(self):
        rows = self._load_parameter_rows()
        def _as_bool(key: str, default: bool) -> bool:
            raw = (rows.get(key) or [str(int(default))])[0] if rows.get(key) else str(int(default))
            s = str(raw).strip().lower()
            if s in ("1", "true", "on", "yes", "y"):
                return True
            if s in ("0", "false", "off", "no", "n"):
                return False
            return bool(default)

        vis1 = self._normalize_serial_text((rows.get("vision1_serial") or [""])[0] if rows.get("vision1_serial") else "")
        vis2 = self._normalize_serial_text((rows.get("vision2_serial") or [""])[0] if rows.get("vision2_serial") else "")
        if not vis1:
            vis1 = DEFAULT_VISION1_SERIAL
        if not vis2:
            vis2 = DEFAULT_VISION2_SERIAL
        if vis1 == vis2:
            vis2 = DEFAULT_VISION2_SERIAL if vis1 != DEFAULT_VISION2_SERIAL else DEFAULT_VISION1_SERIAL
        self._vision_assigned_serial_1 = vis1
        self._vision_assigned_serial_2 = vis2
        self._runtime_camera_serial_1 = vis1
        self._runtime_camera_serial_2 = vis2

        discovered = self._discover_connected_camera_serials()
        merged = []
        for s in discovered + [vis1, vis2]:
            if s and s not in merged:
                merged.append(s)
        self._available_camera_serials = merged
        try:
            rot1 = int(float((rows.get("vision1_view_rotation_deg") or ["0"])[0]))
        except Exception:
            rot1 = 0
        try:
            rot2 = int(float((rows.get("vision2_view_rotation_deg") or ["0"])[0]))
        except Exception:
            rot2 = 0
        self._vision_rotation_deg_1 = self._normalize_rotation_deg(rot1)
        self._vision_rotation_deg_2 = self._normalize_rotation_deg(rot2)
        self._top_status_enabled["vision"] = _as_bool("top_status_vision_enabled", True)
        self._top_status_enabled["vision2"] = _as_bool("top_status_vision2_enabled", True)
        self._top_status_enabled["robot"] = _as_bool("top_status_robot_enabled", True)
        # Do not auto-restore calibration ON state on startup.
        self._calibration_mode_enabled_1 = False
        self._calibration_mode_enabled_2 = False
        self._calibration_mode_enabled = bool(self._calibration_mode_enabled_1 or self._calibration_mode_enabled_2)
        for k, toggle in self._top_status_toggles.items():
            if toggle is None:
                continue
            on = bool(self._top_status_enabled.get(k, True))
            toggle.blockSignals(True)
            toggle.setChecked(on)
            toggle.setText("ON" if on else "OFF")
            toggle.blockSignals(False)

    def _save_vision_serial_settings(self):
        rows = self._load_parameter_rows()
        rows["vision1_serial"] = [self._normalize_serial_text(self._vision_assigned_serial_1)]
        rows["vision2_serial"] = [self._normalize_serial_text(self._vision_assigned_serial_2)]
        rows["vision1_view_rotation_deg"] = [str(int(self._vision_rotation_deg_1))]
        rows["vision2_view_rotation_deg"] = [str(int(self._vision_rotation_deg_2))]
        rows["top_status_vision_enabled"] = ["1" if bool(self._top_status_enabled.get("vision", True)) else "0"]
        rows["top_status_vision2_enabled"] = ["1" if bool(self._top_status_enabled.get("vision2", True)) else "0"]
        rows["top_status_robot_enabled"] = ["1" if bool(self._top_status_enabled.get("robot", True)) else "0"]
        # Keep startup behavior deterministic: calibration mode starts OFF after restart.
        rows["vision1_calibration_enabled"] = ["0"]
        rows["vision2_calibration_enabled"] = ["0"]
        self._save_parameter_rows(rows)

    def _normalize_rotation_deg(self, value):
        try:
            v = int(value)
        except Exception:
            v = 0
        v = v % 360
        candidates = (0, 90, 180, 270)
        return min(candidates, key=lambda x: abs(x - v))

    def _rotate_image_for_panel(self, image: QImage, panel_index: int):
        if image is None:
            return None
        deg = self._vision_rotation_deg_2 if int(panel_index) == 2 else self._vision_rotation_deg_1
        deg = self._normalize_rotation_deg(deg)
        if deg == 0:
            return image
        tf = QTransform()
        tf.rotate(float(deg))
        return image.transformed(tf, Qt.FastTransformation)

    def _map_rotated_to_source_coords(self, x_rot: int, y_rot: int, src_w: int, src_h: int, rot_deg: int):
        rot = self._normalize_rotation_deg(rot_deg)
        x_r = int(max(0, min(max(0, (src_h - 1) if rot in (90, 270) else (src_w - 1)), x_rot)))
        y_r = int(max(0, min(max(0, (src_w - 1) if rot in (90, 270) else (src_h - 1)), y_rot)))
        if rot == 90:  # clockwise
            return int(max(0, min(src_w - 1, y_r))), int(max(0, min(src_h - 1, (src_h - 1) - x_r)))
        if rot == 180:
            return int(max(0, min(src_w - 1, (src_w - 1) - x_r))), int(max(0, min(src_h - 1, (src_h - 1) - y_r)))
        if rot == 270:  # clockwise
            return int(max(0, min(src_w - 1, (src_w - 1) - y_r))), int(max(0, min(src_h - 1, x_r)))
        return int(max(0, min(src_w - 1, x_r))), int(max(0, min(src_h - 1, y_r)))

    def _map_source_to_rotated_coords(self, x_src: int, y_src: int, src_w: int, src_h: int, rot_deg: int):
        rot = self._normalize_rotation_deg(rot_deg)
        xs = int(max(0, min(max(0, src_w - 1), int(x_src))))
        ys = int(max(0, min(max(0, src_h - 1), int(y_src))))
        if rot == 90:   # clockwise
            return int(max(0, min(src_h - 1, (src_h - 1) - ys))), int(max(0, min(src_w - 1, xs)))
        if rot == 180:
            return int(max(0, min(src_w - 1, (src_w - 1) - xs))), int(max(0, min(src_h - 1, (src_h - 1) - ys)))
        if rot == 270:  # clockwise
            return int(max(0, min(src_h - 1, ys))), int(max(0, min(src_w - 1, (src_w - 1) - xs)))
        return xs, ys

    def _to_display_uv(self, u_src: float, v_src: float, panel_index: int = 1, src_w: int = None, src_h: int = None):
        panel = 2 if int(panel_index) == 2 else 1
        if src_w is None or src_h is None:
            size = self._last_yolo_image_size_2 if panel == 2 else self._last_yolo_image_size
            if size is None or len(size) < 2:
                return float(u_src), float(v_src)
            src_w = int(size[0])
            src_h = int(size[1])
        rot = self._vision_rotation_deg_2 if panel == 2 else self._vision_rotation_deg_1
        du, dv = self._map_source_to_rotated_coords(int(round(float(u_src))), int(round(float(v_src))), int(src_w), int(src_h), int(rot))
        return float(du), float(dv)

    def _map_source_to_canvas_coords(self, x_src: float, y_src: float, panel_index: int = 1):
        panel = 2 if int(panel_index) == 2 else 1
        m = self._yolo_view_map_2 if panel == 2 else self._yolo_view_map
        if not m:
            return None
        src_orig_w = int(m.get("src_orig_w", m.get("src_w", 0)))
        src_orig_h = int(m.get("src_orig_h", m.get("src_h", 0)))
        if src_orig_w <= 0 or src_orig_h <= 0:
            return None
        rot_deg = int(m.get("rotation_deg", 0))
        xr, yr = self._map_source_to_rotated_coords(
            int(round(float(x_src))),
            int(round(float(y_src))),
            src_orig_w,
            src_orig_h,
            rot_deg,
        )
        scale = float(m.get("scale", 1.0))
        ax = float(xr) * scale
        ay = float(yr) * scale
        draw_w = int(m.get("draw_w", 0))
        draw_h = int(m.get("draw_h", 0))
        view_w = int(m.get("view_w", 0))
        view_h = int(m.get("view_h", 0))
        crop_x = int(m.get("crop_x", 0))
        crop_y = int(m.get("crop_y", 0))
        pad_x = int(m.get("pad_x", 0))
        pad_y = int(m.get("pad_y", 0))
        vx = (ax - float(crop_x)) if draw_w > view_w else (ax + float(pad_x))
        vy = (ay - float(crop_y)) if draw_h > view_h else (ay + float(pad_y))
        return int(round(vx)), int(round(vy))

    def _set_vision_rotation(self, panel_index: int, delta_deg: int):
        if int(panel_index) == 2:
            self._vision_rotation_deg_2 = self._normalize_rotation_deg(self._vision_rotation_deg_2 + int(delta_deg))
            # 회전 후 원점이 보이도록 포커스를 원점 기준으로 초기화한다.
            self._yolo_pan_x_2 = -1000000.0
            self._yolo_pan_y_2 = -1000000.0
            self._render_yolo_view_2()
            self.append_log(f"[비전2] 화면 회전: {self._vision_rotation_deg_2}도\n")
        else:
            self._vision_rotation_deg_1 = self._normalize_rotation_deg(self._vision_rotation_deg_1 + int(delta_deg))
            self._yolo_pan_x = -1000000.0
            self._yolo_pan_y = -1000000.0
            self._render_yolo_view()
            self.append_log(f"[비전1] 화면 회전: {self._vision_rotation_deg_1}도\n")
        self._refresh_vision_rotation_labels()
        self._save_vision_serial_settings()

    def _set_vision_rotation_absolute(self, panel_index: int, absolute_deg: int):
        target = self._normalize_rotation_deg(absolute_deg)
        if int(panel_index) == 2:
            self._vision_rotation_deg_2 = target
            self._yolo_pan_x_2 = -1000000.0
            self._yolo_pan_y_2 = -1000000.0
            self._render_yolo_view_2()
            self.append_log(f"[비전2] 화면 회전: {self._vision_rotation_deg_2}도\n")
        else:
            self._vision_rotation_deg_1 = target
            self._yolo_pan_x = -1000000.0
            self._yolo_pan_y = -1000000.0
            self._render_yolo_view()
            self.append_log(f"[비전1] 화면 회전: {self._vision_rotation_deg_1}도\n")
        self._refresh_vision_rotation_labels()
        self._save_vision_serial_settings()

    def _runtime_camera_slot_for_serial(self, serial):
        s = self._normalize_serial_text(serial)
        if s and s == self._normalize_serial_text(self._runtime_camera_serial_1):
            return 1
        if s and s == self._normalize_serial_text(self._runtime_camera_serial_2):
            return 2
        return 1

    def _assigned_topic_for_serial(self, serial):
        slot = self._runtime_camera_slot_for_serial(serial)
        return YOLO_EXTERNAL_TOPIC_2 if slot == 2 else YOLO_EXTERNAL_TOPIC

    def _assigned_depth_topic_for_serial(self, serial):
        slot = self._runtime_camera_slot_for_serial(serial)
        return "/camera2/camera/aligned_depth_to_color/image_raw" if slot == 2 else YOLO_EXTERNAL_DEPTH_TOPIC

    def _assigned_info_topic_for_serial(self, serial, node):
        slot = self._runtime_camera_slot_for_serial(serial)
        if slot == 2:
            for t in ("/camera2/camera/color/camera_info", "/camera2/color/camera_info"):
                if self._is_image_topic_alive(node, t.replace("/camera_info", "/image_raw")):
                    return t
            return "/camera2/camera/color/camera_info"
        if self._is_image_topic_alive(node, CALIB_VISION_TOPIC_PRIMARY):
            return CALIB_CAMERA_INFO_TOPIC_PRIMARY
        return CALIB_CAMERA_INFO_TOPIC_FALLBACK

    def _assigned_calib_topic_for_serial(self, serial, node):
        slot = self._runtime_camera_slot_for_serial(serial)
        if slot == 2:
            topic_primary = "/camera2/camera/color/image_raw"
            topic_fallback = "/camera2/color/image_raw"
            for t in (topic_primary, topic_fallback):
                if self._is_image_topic_alive(node, t):
                    return t
            return YOLO_EXTERNAL_TOPIC_2 if self._is_image_topic_alive(node, YOLO_EXTERNAL_TOPIC_2) else topic_primary
        return self._resolve_calib_vision_topic(node)

    def _vision_panel_needs_depth(self, panel_index: int = 1):
        panel = 2 if int(panel_index) == 2 else 1
        calib_on = bool(self._calibration_mode_enabled_2) if panel == 2 else bool(self._calibration_mode_enabled_1)
        return bool(calib_on or self._vision_click_dialog_enabled)

    def _vision_panel_needs_camera_info(self, panel_index: int = 1):
        # camera_info는 저부하이며 축/좌표 정확도에 직접 필요하므로 active panel에서는 유지한다.
        _ = panel_index
        return True

    def _refresh_vision_serial_labels(self):
        text1 = f"시리얼: {self._normalize_serial_text(self._vision_assigned_serial_1) or '-'}"
        text2 = f"시리얼: {self._normalize_serial_text(self._vision_assigned_serial_2) or '-'}"
        if self._vision_serial_label is not None:
            self._vision_serial_label.setText(text1)
        if self._vision_serial_label_2 is not None:
            self._vision_serial_label_2.setText(text2)

    def _refresh_vision_rotation_labels(self):
        if self._vision_rotation_label is not None:
            self._vision_rotation_label.setText(f"각도: {int(self._vision_rotation_deg_1)}°")
        if self._vision_rotation_label_2 is not None:
            self._vision_rotation_label_2.setText(f"각도: {int(self._vision_rotation_deg_2)}°")

    def _try_load_calibration_matrix_on_startup(self):
        try:
            # Legacy(calibration/*) -> New(config/calibration/*) one-way migration
            try:
                if os.path.isdir(LEGACY_CALIB_ROBOT_DIR):
                    os.makedirs(CALIB_ROBOT_DIR, exist_ok=True)
                    for p in glob.glob(os.path.join(LEGACY_CALIB_ROBOT_DIR, "*")):
                        if not os.path.isfile(p):
                            continue
                        dst = os.path.join(CALIB_ROBOT_DIR, os.path.basename(p))
                        if not os.path.isfile(dst):
                            try:
                                import shutil
                                shutil.copy2(p, dst)
                            except Exception:
                                pass
                if os.path.isdir(LEGACY_CALIB_ROTMAT_DIR):
                    os.makedirs(CALIB_ROTMAT_DIR, exist_ok=True)
                    for p in glob.glob(os.path.join(LEGACY_CALIB_ROTMAT_DIR, "*")):
                        if not os.path.isfile(p):
                            continue
                        dst = os.path.join(CALIB_ROTMAT_DIR, os.path.basename(p))
                        if not os.path.isfile(dst):
                            try:
                                import shutil
                                shutil.copy2(p, dst)
                            except Exception:
                                pass
            except Exception:
                pass

            env_path = os.environ.get("CALIB_MATRIX_PATH", "").strip()
            candidates = []
            if env_path:
                candidates.append(env_path)
            migrated_legacy_path = ""
            if os.path.isfile(LEGACY_CALIB_ACTIVE_PATH_FILE):
                try:
                    with open(LEGACY_CALIB_ACTIVE_PATH_FILE, "r", encoding="utf-8") as f:
                        p = f.readline().strip()
                    if p:
                        migrated_legacy_path = p
                    os.remove(LEGACY_CALIB_ACTIVE_PATH_FILE)
                except Exception:
                    pass
            if migrated_legacy_path:
                self._save_active_calibration_path(migrated_legacy_path)
                candidates.append(migrated_legacy_path)
            rows = self._load_parameter_rows()
            active_path_1 = (rows.get("active_calibration_path_1") or [""])[0] if rows.get("active_calibration_path_1") else ""
            active_path_2 = (rows.get("active_calibration_path_2") or [""])[0] if rows.get("active_calibration_path_2") else ""
            legacy_active = (rows.get("active_calibration_path") or [""])[0] if rows.get("active_calibration_path") else ""
            if not active_path_1:
                active_path_1 = legacy_active
            if not active_path_2:
                active_path_2 = active_path_1

            last_matrix = None
            if os.path.isdir(CALIB_ROTMAT_DIR):
                matrix_files = sorted(
                    glob.glob(os.path.join(CALIB_ROTMAT_DIR, "calib_matrix_*.txt")),
                    key=lambda p: os.path.getmtime(p),
                    reverse=True,
                )
                if matrix_files:
                    last_matrix = matrix_files[0]

            path_for_panel = {
                1: active_path_1 if active_path_1 else (candidates[0] if candidates else last_matrix),
                2: active_path_2 if active_path_2 else (active_path_1 if active_path_1 else last_matrix),
            }
            loaded_any = False
            for panel, raw_path in path_for_panel.items():
                p = _resolve_repo_file(raw_path, fallback_dirs=(CALIB_DIR, CALIB_ROTMAT_DIR))
                if not p:
                    continue
                mat, rmse = self._load_affine_from_file(p)
                if mat is None:
                    continue
                self._apply_vision_to_robot_affine(mat, rmse=rmse, path=p, panel_index=panel)
                self.append_log(f"[캘리브레이션{panel}] 시작 시 행렬 자동 로드: {p}\n")
                loaded_any = True
            if not loaded_any:
                # 시작 시점에 로드 가능한 행렬이 없으면 명시적으로 없음 상태를 유지/표시
                self._calib_matrix_path = None
                self._calib_matrix_path_1 = None
                self._calib_matrix_path_2 = None
                self.append_log("[캘리브레이션] 시작 시 적용 가능한 행렬 없음 (현재 적용행렬: 없음)\n")
        except Exception:
            return

    def _build_vision_image_qos(self):
        if (
            QoSProfile is not None
            and QoSHistoryPolicy is not None
            and QoSReliabilityPolicy is not None
        ):
            return QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
            )
        return qos_profile_sensor_data if qos_profile_sensor_data is not None else 1

    def _is_image_topic_alive(self, node, topic_name):
        if node is None or not topic_name:
            return False
        try:
            if int(node.count_publishers(str(topic_name))) > 0:
                return True
        except Exception:
            pass
        try:
            infos = node.get_publishers_info_by_topic(str(topic_name))
            if infos:
                return True
        except Exception:
            pass
        return False

    def _resolve_calib_vision_topic(self, node):
        for topic_name in (CALIB_VISION_TOPIC_PRIMARY, CALIB_VISION_TOPIC_FALLBACK):
            if self._is_image_topic_alive(node, topic_name):
                return topic_name
        return CALIB_VISION_TOPIC_PRIMARY

    def _calibration_output_meta_topic(self, panel_index: int = 1):
        return CALIB_OUTPUT_META_TOPIC_2 if int(panel_index) == 2 else CALIB_OUTPUT_META_TOPIC_1

    def _current_vision_image_topic(self):
        if bool(getattr(self, "_calibration_mode_enabled_1", False)):
            node = getattr(self.backend, "robot_controller", None) if self.backend is not None else None
            if node is not None:
                return self._assigned_calib_topic_for_serial(self._vision_assigned_serial_1, node)
            return CALIB_VISION_TOPIC_PRIMARY
        return self._assigned_topic_for_serial(self._vision_assigned_serial_1)

    def _current_vision_image_topic_2(self):
        if bool(getattr(self, "_calibration_mode_enabled_2", False)):
            node = getattr(self.backend, "robot_controller", None) if self.backend is not None else None
            if node is not None:
                return self._assigned_calib_topic_for_serial(self._vision_assigned_serial_2, node)
            return "/camera2/camera/color/image_raw"
        return self._assigned_topic_for_serial(self._vision_assigned_serial_2)

    def _teardown_external_vision_bridge_panel(self, panel_index: int):
        node = getattr(self.backend, "robot_controller", None) if self.backend is not None else None

        def _destroy(sub):
            if sub is None:
                return
            try:
                if node is not None:
                    node.destroy_subscription(sub)
                    return
            except Exception:
                pass
            try:
                sub.destroy()
            except Exception:
                pass

        panel = 2 if int(panel_index) == 2 else 1
        if panel == 2:
            _destroy(getattr(self, "_vision_sub_2", None))
            _destroy(getattr(self, "_calib_meta_sub_2", None))
            _destroy(getattr(self, "_vision_depth_sub_2", None))
            _destroy(getattr(self, "_vision_info_sub_2", None))
            self._vision_sub_2 = None
            self._calib_meta_sub_2 = None
            self._vision_depth_sub_2 = None
            self._vision_info_sub_2 = None
            self._vision_image_topic_in_use_2 = None
            self._calib_meta_topic_in_use_2 = None
        else:
            _destroy(getattr(self, "_vision_sub", None))
            _destroy(getattr(self, "_calib_meta_sub_1", None))
            _destroy(getattr(self, "_vision_depth_sub", None))
            _destroy(getattr(self, "_vision_info_sub", None))
            self._vision_sub = None
            self._calib_meta_sub_1 = None
            self._vision_depth_sub = None
            self._vision_info_sub = None
            self._vision_image_topic_in_use = None
            self._calib_meta_topic_in_use_1 = None

    def _teardown_external_vision_bridge(self):
        self._teardown_external_vision_bridge_panel(1)
        self._teardown_external_vision_bridge_panel(2)

    def _build_calibration_process_cmd(self, panel_index: int):
        panel = 2 if int(panel_index) == 2 else 1
        node = getattr(self.backend, "robot_controller", None) if self.backend is not None else None
        if node is None:
            return None
        serial = self._vision_assigned_serial_2 if panel == 2 else self._vision_assigned_serial_1
        return [
            sys.executable,
            os.path.join(PROJECT_ROOT, "src", "vision", "camera_eye_to_hand_robot_calibration.py"),
            "--image-topic",
            self._assigned_calib_topic_for_serial(serial, node),
            "--depth-topic",
            self._assigned_depth_topic_for_serial(serial),
            "--camera-info-topic",
            self._assigned_info_topic_for_serial(serial, node),
            "--output-meta-topic",
            self._calibration_output_meta_topic(panel),
            "--cols",
            str(int(self._calib_pattern_cols)),
            "--rows",
            str(int(self._calib_pattern_rows)),
            "--panel",
            str(panel),
            "--hold-sec",
            str(float(CALIB_DETECTION_HOLD_SEC)),
            "--detect-interval-sec",
            str(float(CALIB_DETECT_INTERVAL_SEC)),
            "--process-hz",
            str(float(CALIB_PROCESS_HZ)),
        ]

    def _ensure_calibration_process(self, panel_index: int):
        panel = 2 if int(panel_index) == 2 else 1
        node = getattr(self.backend, "robot_controller", None) if self.backend is not None else None
        if node is None:
            return False
        meta_topic = self._calibration_output_meta_topic(panel)
        if self._is_image_topic_alive(node, meta_topic):
            return True
        cmd = self._build_calibration_process_cmd(panel)
        if not cmd:
            return False
        proc = self._calib_proc_2 if panel == 2 else self._calib_proc_1
        prev_cmd = self._calib_proc_cmd_2 if panel == 2 else self._calib_proc_cmd_1
        if proc is not None and proc.poll() is None and prev_cmd == cmd:
            return True
        self._stop_calibration_process(panel)
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            self.append_log(f"[캘리브레이션{panel}] 프로세스 시작 실패: {e}\n")
            return False
        if panel == 2:
            self._calib_proc_2 = proc
            self._calib_proc_cmd_2 = list(cmd)
            self._calib_proc_started_by_ui_2 = True
        else:
            self._calib_proc_1 = proc
            self._calib_proc_cmd_1 = list(cmd)
            self._calib_proc_started_by_ui_1 = True
        self.append_log(f"[캘리브레이션{panel}] 프로세스 시작\n")
        return True

    def _stop_calibration_process(self, panel_index: int):
        panel = 2 if int(panel_index) == 2 else 1
        if panel == 2:
            proc = self._calib_proc_2
            started_by_ui = bool(self._calib_proc_started_by_ui_2)
        else:
            proc = self._calib_proc_1
            started_by_ui = bool(self._calib_proc_started_by_ui_1)
        if proc is not None and proc.poll() is None and started_by_ui:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=1.0)
                    except Exception:
                        pass
        if panel == 2:
            self._calib_proc_2 = None
            self._calib_proc_cmd_2 = None
            self._calib_proc_started_by_ui_2 = False
        else:
            self._calib_proc_1 = None
            self._calib_proc_cmd_1 = None
            self._calib_proc_started_by_ui_1 = False

    def _sync_calibration_processes(self):
        if self.backend is None:
            return
        if bool(getattr(self, "_calibration_mode_enabled_1", False)) and bool(self._top_status_enabled.get("vision", True)):
            self._ensure_calibration_process(1)
        else:
            self._stop_calibration_process(1)
        if bool(getattr(self, "_calibration_mode_enabled_2", False)) and bool(self._top_status_enabled.get("vision2", True)):
            self._ensure_calibration_process(2)
        else:
            self._stop_calibration_process(2)

    def _apply_calibration_meta(self, panel_index: int, raw: str, stream_token=None):
        panel = 2 if int(panel_index) == 2 else 1
        if stream_token is not None:
            current = int(getattr(self, "_vision_stream_token_2", 0)) if panel == 2 else int(getattr(self, "_vision_stream_token_1", 0))
            if int(stream_token) != current:
                return
        try:
            meta = json.loads(raw or "{}")
        except Exception:
            return
        now = time.monotonic()
        points = meta.get("points")
        if not isinstance(points, dict):
            points = {}
        center = meta.get("center")
        if center is not None and isinstance(center, (list, tuple)) and len(center) >= 3:
            points["center"] = [float(center[0]), float(center[1]), float(center[2])]
        grid_points = meta.get("grid_points_uv")
        grid_status = meta.get("grid_status")
        grid_xyz = meta.get("grid_camera_xyz_mm")
        detected = bool(meta.get("detected", False))
        full_ready = bool(meta.get("full_ready", False))
        reason = str(meta.get("reason", "대기"))
        full_reason = str(meta.get("full_reason", reason))
        raw_input_interval_ms = meta.get("raw_input_interval_ms")
        processing_ms = meta.get("processing_ms")
        publish_interval_ms = meta.get("publish_interval_ms")
        frame_wait_ms = meta.get("frame_wait_ms")
        try:
            grid_points_arr = np.asarray(grid_points, dtype=np.float64) if grid_points is not None else None
        except Exception:
            grid_points_arr = None
        try:
            grid_xyz_arr = np.asarray(grid_xyz, dtype=np.float64) if grid_xyz is not None else None
        except Exception:
            grid_xyz_arr = None
        if panel == 2:
            self._calib_last_points_uvz_mm_2 = points if points else None
            self._calib_last_points_at_2 = now if detected else None
            self._calib_last_grid_pts_2 = grid_points_arr
            self._calib_last_grid_status_2 = list(grid_status) if isinstance(grid_status, list) else None
            self._calib_grid_camera_xyz_2 = grid_xyz_arr
            self._calib_last_reason_2 = reason
            self._calib_full_ready_2 = bool(full_ready)
            self._calib_full_reason_2 = full_reason
            self._calib_proc_input_ms_2 = float(raw_input_interval_ms) if raw_input_interval_ms is not None else None
            self._calib_proc_ms_2 = float(processing_ms) if processing_ms is not None else None
            self._calib_proc_publish_ms_2 = float(publish_interval_ms) if publish_interval_ms is not None else None
            self._calib_proc_wait_ms_2 = float(frame_wait_ms) if frame_wait_ms is not None else None
        else:
            self._calib_last_points_uvz_mm_1 = points if points else None
            self._calib_last_points_at_1 = now if detected else None
            self._calib_last_grid_pts_1 = grid_points_arr
            self._calib_last_grid_status_1 = list(grid_status) if isinstance(grid_status, list) else None
            self._calib_grid_camera_xyz_1 = grid_xyz_arr
            self._calib_last_reason_1 = reason
            self._calib_full_ready_1 = bool(full_ready)
            self._calib_full_reason_1 = full_reason
            self._calib_proc_input_ms_1 = float(raw_input_interval_ms) if raw_input_interval_ms is not None else None
            self._calib_proc_ms_1 = float(processing_ms) if processing_ms is not None else None
            self._calib_proc_publish_ms_1 = float(publish_interval_ms) if publish_interval_ms is not None else None
            self._calib_proc_wait_ms_1 = float(frame_wait_ms) if frame_wait_ms is not None else None
            self._calib_last_points_uvz_mm = self._calib_last_points_uvz_mm_1
            self._calib_last_points_at = self._calib_last_points_at_1
            self._calib_last_reason = self._calib_last_reason_1
        self.calibration_ui_refresh_requested.emit()

    def _on_calibration_meta_msg(self, msg, stream_token=None):
        try:
            self._apply_calibration_meta(1, getattr(msg, "data", ""), stream_token=stream_token)
        except Exception:
            return

    def _on_calibration_meta_msg_2(self, msg, stream_token=None):
        try:
            self._apply_calibration_meta(2, getattr(msg, "data", ""), stream_token=stream_token)
        except Exception:
            return

    def _ensure_external_vision_process(self):
        proc = getattr(self, "_external_vision_proc", None)
        if proc is not None and proc.poll() is None:
            return True
        if not YOLO_AUTO_LAUNCH_CMD:
            return False
        try:
            cmd = shlex.split(YOLO_AUTO_LAUNCH_CMD)
        except Exception as e:
            self.append_log(f"[비전] 자동실행 명령 파싱 실패: {e}\n")
            return False
        if not cmd:
            return False
        try:
            self._external_vision_proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._external_vision_cmd = cmd
            self._external_vision_started_by_ui = True
            self.append_log(f"[비전] 외부 비전 프로세스 시작: {' '.join(cmd)}\n")
            return True
        except Exception as e:
            self._external_vision_proc = None
            self._external_vision_cmd = None
            self._external_vision_started_by_ui = False
            self.append_log(f"[비전] 외부 비전 프로세스 시작 실패: {e}\n")
            return False

    def _stop_external_vision_process(self):
        proc = getattr(self, "_external_vision_proc", None)
        if proc is None:
            self._external_vision_cmd = None
            self._external_vision_started_by_ui = False
            return
        if proc.poll() is None and bool(getattr(self, "_external_vision_started_by_ui", False)):
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
        self._external_vision_proc = None
        self._external_vision_cmd = None
        self._external_vision_started_by_ui = False

    def _start_yolo_camera(self):
        if not UI_ENABLE_VISION or self.backend is None:
            return
        if YOLO_EXTERNAL_NODE:
            try:
                if YOLO_AUTO_LAUNCH_NODE and (YOLO_AUTO_LAUNCH_ALWAYS or (not self._calibration_mode_enabled)):
                    self._ensure_external_vision_process()
                self._setup_external_vision_bridge()
            except Exception as e:
                self.append_log(f"[비전] 시작 실패: {e}\n")
            return
        if YOLO_ALLOW_INTERNAL_WEBCAM:
            self.append_log("[비전] 내부 웹캠 모드는 현재 비활성화 상태입니다.\n")
        else:
            self.append_log("[비전] 외부 비전 노드 미사용 상태입니다.\n")

    def _setup_external_vision_bridge_panel(self, panel_index: int):
        if RosImageMsg is None or self.backend is None:
            return False
        node = getattr(self.backend, "robot_controller", None)
        if node is None:
            return False
        panel = 2 if int(panel_index) == 2 else 1
        image_qos = self._build_vision_image_qos()
        if panel == 2:
            sub_kwargs = {}
            if self._vision_cb_group_2 is not None:
                sub_kwargs["callback_group"] = self._vision_cb_group_2
            image_topic = self._current_vision_image_topic_2()
            self._teardown_external_vision_bridge_panel(2)
            stream_token = self._vision_stream_token_2
            self._vision_sub_2 = node.create_subscription(
                RosImageMsg, image_topic, lambda msg, token=stream_token: self._on_vision_image_msg_2(msg, token), image_qos, **sub_kwargs
            )
            self._vision_image_topic_in_use_2 = image_topic
            if bool(self._calibration_mode_enabled_2) and RosStringMsg is not None:
                meta_topic_2 = self._calibration_output_meta_topic(2)
                self._calib_meta_sub_2 = node.create_subscription(
                    RosStringMsg,
                    meta_topic_2,
                    lambda msg, token=stream_token: self._on_calibration_meta_msg_2(msg, token),
                    10,
                    **sub_kwargs,
                )
                self._calib_meta_topic_in_use_2 = meta_topic_2
            if self._vision_panel_needs_depth(2):
                depth_topic_2 = self._assigned_depth_topic_for_serial(self._vision_assigned_serial_2)
                self._vision_depth_sub_2 = node.create_subscription(
                    RosImageMsg, depth_topic_2, lambda msg, token=stream_token: self._on_vision_depth_msg_2(msg, token), image_qos, **sub_kwargs
                )
            else:
                depth_topic_2 = None
            if self._vision_panel_needs_camera_info(2) and RosCameraInfoMsg is not None:
                info_topic_2 = self._assigned_info_topic_for_serial(self._vision_assigned_serial_2, node)
                self._vision_info_sub_2 = node.create_subscription(
                    RosCameraInfoMsg,
                    info_topic_2,
                    lambda msg, token=stream_token: self._on_vision_camera_info_msg(msg, token, panel_index=2),
                    image_qos,
                    **sub_kwargs,
                )
                self.append_log(f"[비전2] 카메라정보 구독 시작: {info_topic_2}\n")
            self._vision_state_text_2 = "끊김"
            self.append_log(f"[비전2] 외부 비전 구독 시작({'CALIB' if self._calibration_mode_enabled_2 else 'YOLO'}): {image_topic}\n")
            if bool(self._calibration_mode_enabled_2) and self._calib_meta_topic_in_use_2:
                self.append_log(f"[캘리브레이션2] 메타 구독 시작: {self._calib_meta_topic_in_use_2}\n")
            if depth_topic_2:
                self.append_log(f"[비전2] 뎁스 구독 시작: {depth_topic_2}\n")
        else:
            sub_kwargs = {}
            if self._vision_cb_group_1 is not None:
                sub_kwargs["callback_group"] = self._vision_cb_group_1
            image_topic = self._current_vision_image_topic()
            self._teardown_external_vision_bridge_panel(1)
            stream_token = self._vision_stream_token_1
            self._vision_sub = node.create_subscription(
                RosImageMsg, image_topic, lambda msg, token=stream_token: self._on_vision_image_msg(msg, token), image_qos, **sub_kwargs
            )
            self._vision_image_topic_in_use = image_topic
            if bool(self._calibration_mode_enabled_1) and RosStringMsg is not None:
                meta_topic_1 = self._calibration_output_meta_topic(1)
                self._calib_meta_sub_1 = node.create_subscription(
                    RosStringMsg,
                    meta_topic_1,
                    lambda msg, token=stream_token: self._on_calibration_meta_msg(msg, token),
                    10,
                    **sub_kwargs,
                )
                self._calib_meta_topic_in_use_1 = meta_topic_1
            if self._vision_panel_needs_depth(1):
                depth_topic = self._assigned_depth_topic_for_serial(self._vision_assigned_serial_1)
                self._vision_depth_sub = node.create_subscription(
                    RosImageMsg, depth_topic, lambda msg, token=stream_token: self._on_vision_depth_msg(msg, token), image_qos, **sub_kwargs
                )
            else:
                depth_topic = None
            if self._vision_panel_needs_camera_info(1) and RosCameraInfoMsg is not None:
                info_topic = self._assigned_info_topic_for_serial(self._vision_assigned_serial_1, node)
                self._vision_info_sub = node.create_subscription(
                    RosCameraInfoMsg,
                    info_topic,
                    lambda msg, token=stream_token: self._on_vision_camera_info_msg(msg, token, panel_index=1),
                    image_qos,
                    **sub_kwargs,
                )
                self.append_log(f"[비전] 카메라정보 구독 시작: {info_topic}\n")
            self._vision_state_text = "끊김"
            self.append_log(f"[비전] 외부 비전 구독 시작({'CALIB' if self._calibration_mode_enabled_1 else 'YOLO'}): {image_topic}\n")
            if bool(self._calibration_mode_enabled_1) and self._calib_meta_topic_in_use_1:
                self.append_log(f"[캘리브레이션1] 메타 구독 시작: {self._calib_meta_topic_in_use_1}\n")
            if depth_topic:
                self.append_log(f"[비전] 뎁스 구독 시작: {depth_topic}\n")
        return True

    def _rebind_external_vision_bridge_for_mode(self):
        if not YOLO_EXTERNAL_NODE:
            return
        if self.backend is None:
            return
        node = getattr(self.backend, "robot_controller", None)
        if node is None:
            return
        self._teardown_external_vision_bridge()
        if not self._vision_panel_needs_depth(1):
            self._clear_vision_depth_cache(1)
        if not self._vision_panel_needs_depth(2):
            self._clear_vision_depth_cache(2)
        self._sync_calibration_processes()
        self._setup_external_vision_bridge()

    def _setup_external_vision_bridge(self):
        if RosImageMsg is None:
            self.append_log("[비전] sensor_msgs/Image import 실패\n")
            self._vision_state_text = "오류"
            self._vision_state_text_2 = "오류"
            return False

        node = getattr(self.backend, "robot_controller", None)
        if node is None:
            self.append_log("[비전] ROS 노드 없음: 외부 비전 구독 실패\n")
            self._vision_state_text = "오류"
            self._vision_state_text_2 = "오류"
            return False

        try:
            ok = False
            if bool(self._top_status_enabled.get("vision", True)):
                ok = self._setup_external_vision_bridge_panel(1) or ok
            else:
                self._teardown_external_vision_bridge_panel(1)
            if bool(self._top_status_enabled.get("vision2", True)):
                ok = self._setup_external_vision_bridge_panel(2) or ok
            else:
                self._teardown_external_vision_bridge_panel(2)
            return ok
        except Exception as e:
            now = time.monotonic()
            msg = str(e)
            if (msg != self._vision_bridge_fail_last_msg) or ((now - self._vision_bridge_fail_last_at) > 5.0):
                if self._calibration_mode_enabled_1 or self._calibration_mode_enabled_2:
                    self.append_log(f"[비전] 캘리브레이션 비전 대기: {e}\n")
                else:
                    self.append_log(f"[비전] 외부 비전 구독 실패: {e}\n")
                self._vision_bridge_fail_last_msg = msg
                self._vision_bridge_fail_last_at = now
            self._vision_state_text = "대기" if self._calibration_mode_enabled_1 else "오류"
            self._vision_state_text_2 = "대기" if self._calibration_mode_enabled_2 else "오류"
            return False


    def _detect_calib_checkerboard(self, gray):
        try:
            import cv2
        except Exception:
            return None
        cols = int(self._calib_pattern_cols)
        rows = int(self._calib_pattern_rows)
        # 현장에서는 "7x9"를 칸 수로 표현하는 경우가 많아
        # 내부 코너 수(6x8)도 함께 시도한다.
        candidate_sizes = []
        for c, r in (
            (cols, rows),
            (max(3, cols - 1), max(3, rows - 1)),
            (rows, cols),
            (max(3, rows - 1), max(3, cols - 1)),
        ):
            if (c, r) not in candidate_sizes:
                candidate_sizes.append((c, r))
        # 탐색 경우수를 줄이되, 실제 현장 조명/대비 편차를 고려해 최소한의 변형은 함께 시도한다.
        variants = [gray, cv2.equalizeHist(gray)]
        try:
            variants.append(cv2.GaussianBlur(gray, (5, 5), 0))
        except Exception:
            pass
        variants.append(255 - gray)
        for g in variants:
            for cand_cols, cand_rows in candidate_sizes:
                pattern_size = (cand_cols, cand_rows)
                if hasattr(cv2, "findChessboardCornersSB"):
                    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
                    if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
                        flags |= cv2.CALIB_CB_EXHAUSTIVE
                    if hasattr(cv2, "CALIB_CB_ACCURACY"):
                        flags |= cv2.CALIB_CB_ACCURACY
                    found, corners = cv2.findChessboardCornersSB(g, pattern_size, flags=flags)
                    if found and corners is not None:
                        return corners.reshape(cand_rows, cand_cols, 2)
                # SB가 있더라도 실패할 수 있어 classic 탐색을 항상 후순위로 시도한다.
                flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                found, corners = cv2.findChessboardCorners(g, pattern_size, flags=flags)
                if found and corners is not None:
                    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
                    corners = cv2.cornerSubPix(g, corners, (11, 11), (-1, -1), term)
                    return corners.reshape(cand_rows, cand_cols, 2)
        return None

    def _calib_outer_from_inner(self, pts_grid):
        rows, cols = pts_grid.shape[:2]
        p_tl = pts_grid[0, 0]
        p_tr = pts_grid[0, cols - 1]
        p_bl = pts_grid[rows - 1, 0]
        p_br = pts_grid[rows - 1, cols - 1]
        ux_top = (p_tr - p_tl) / max(1, cols - 1)
        ux_bot = (p_br - p_bl) / max(1, cols - 1)
        uy_left = (p_bl - p_tl) / max(1, rows - 1)
        uy_right = (p_br - p_tr) / max(1, rows - 1)
        ux = 0.5 * (ux_top + ux_bot)
        uy = 0.5 * (uy_left + uy_right)
        # 내부 코너 -> 보드 실제 가장자리로 확장할 때는 1칸이 아니라 반칸(0.5 step)이다.
        o_tl = p_tl - 0.5 * ux - 0.5 * uy
        o_tr = p_tr + 0.5 * ux - 0.5 * uy
        o_bl = p_bl - 0.5 * ux + 0.5 * uy
        o_br = p_br + 0.5 * ux + 0.5 * uy
        return [o_tl, o_tr, o_bl, o_br]

    def _calib_order_outer_screen(self, pts4):
        pts = [np.array(p, dtype=np.float32) for p in pts4]
        pts_sorted = sorted(pts, key=lambda p: (float(p[1]), float(p[0])))
        top2 = sorted(pts_sorted[:2], key=lambda p: float(p[0]))
        bot2 = sorted(pts_sorted[2:], key=lambda p: float(p[0]))
        tl, tr = top2[0], top2[1]
        bl, br = bot2[0], bot2[1]
        return tl, tr, bl, br

    def _calib_line_intersection(self, a1, a2, b1, b2):
        x1, y1 = float(a1[0]), float(a1[1])
        x2, y2 = float(a2[0]), float(a2[1])
        x3, y3 = float(b1[0]), float(b1[1])
        x4, y4 = float(b2[0]), float(b2[1])
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-9:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
        return np.array([px, py], dtype=np.float32)

    def _calib_pick_p5_upper_right(self, pts_grid, center):
        pts = pts_grid.reshape(-1, 2)
        cx, cy = float(center[0]), float(center[1])
        cands = [p for p in pts if float(p[0]) >= cx and float(p[1]) <= cy]
        if not cands:
            cands = [p for p in pts if float(p[0]) >= cx]
        if not cands:
            cands = [p for p in pts if float(p[1]) <= cy]
        if not cands:
            cands = list(pts)
        best = min(cands, key=lambda p: (float(p[0] - cx) ** 2 + float(p[1] - cy) ** 2))
        return np.array(best, dtype=np.float32)

    def _draw_vision_axes_overlay(self, bgr, panel_index: int = 1):
        try:
            import cv2
        except Exception:
            return bgr
        h, w = bgr.shape[:2]
        if h <= 2 or w <= 2:
            return bgr
        # 비전 XYZ(mm) 좌표계(카메라 광학좌표)를 영상 위에 투영해 표시한다.
        # 원점은 광학중심(cx, cy), 방향은 X+ 우측 / Y+ 하향.
        panel = 2 if int(panel_index) == 2 else 1
        intr = self._camera_intrinsics_2 if panel == 2 else self._camera_intrinsics_1
        if intr is None and panel == 1:
            intr = self._camera_intrinsics
        if intr is not None and len(intr) >= 4:
            fx, fy, cx, cy = [float(v) for v in intr[:4]]
            ox = int(round(cx))
            oy = int(round(cy))
        else:
            fx = fy = None
            ox = int(w // 2)
            oy = int(h // 2)
        ox = max(1, min(w - 2, ox))
        oy = max(1, min(h - 2, oy))
        color = (0, 255, 0)
        tip_color = (0, 0, 255)
        thickness = 2

        # 오버레이 크기는 사용자 인지용으로 고정하고, 축선은 점선으로 화면 끝까지 연장한다.
        axis_len_px = max(14, int(min(w, h) * (1.0 / 3.0)))
        x_end = min(w - 2, max(1, ox + axis_len_px))
        y_end = min(h - 2, max(1, oy + axis_len_px))

        # Dashed axis lines to both edges
        dash = max(6, int(min(w, h) * 0.015))
        gap = max(4, int(dash * 0.7))
        xx = 1
        while xx < (w - 1):
            x2 = min(w - 2, xx + dash)
            cv2.line(bgr, (xx, oy), (x2, oy), color, thickness, cv2.LINE_AA)
            xx = x2 + gap
        yy = 1
        while yy < (h - 1):
            y2 = min(h - 2, yy + dash)
            cv2.line(bgr, (ox, yy), (ox, y2), color, thickness, cv2.LINE_AA)
            yy = y2 + gap
        # 화살표 촉만 빨간색으로 강조
        tip_len_x = max(8, int(w * 0.04))
        tip_len_y = max(8, int(h * 0.04))
        cv2.arrowedLine(
            bgr,
            (max(ox, (w - 2) - tip_len_x), oy),
            (w - 2, oy),
            tip_color,
            thickness,
            cv2.LINE_AA,
            tipLength=0.45,
        )
        cv2.arrowedLine(
            bgr,
            (ox, max(oy, (h - 2) - tip_len_y)),
            (ox, h - 2),
            tip_color,
            thickness,
            cv2.LINE_AA,
            tipLength=0.45,
        )

        # 원점 표시(빨간색)
        cv2.circle(bgr, (ox, oy), 4, (0, 0, 255), -1, cv2.LINE_AA)

        cv2.putText(
            bgr,
            "X+",
            (max(8, min(w - 40, (w - 2) - 24)), max(18, min(h - 8, oy + 26))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            "X+",
            (max(8, min(w - 40, (w - 2) - 24)), max(18, min(h - 8, oy + 26))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            "Y+",
            (max(8, min(w - 40, ox + 8)), max(18, min(h - 8, (h - 2) - 8))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            "Y+",
            (max(8, min(w - 40, ox + 8)), max(18, min(h - 8, (h - 2) - 8))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            color,
            2,
            cv2.LINE_AA,
        )
        return bgr

    def _draw_calib_info_overlay(self, bgr, data_dict):
        # 캘리브레이션 좌표 텍스트는 Qt 캔버스(좌하단)에서 정방향으로 별도 표시한다.
        return bgr

    def _extract_calib_center_uvz(self, data_dict):
        if not isinstance(data_dict, dict):
            return None
        vals = data_dict.get("center")
        if vals is not None and len(vals) >= 3:
            try:
                return float(vals[0]), float(vals[1]), float(vals[2])
            except Exception:
                pass
        pts = []
        for k in ("p1", "p2", "p3", "p4"):
            v = data_dict.get(k)
            if v is None or len(v) < 2:
                return None
            try:
                pts.append((float(v[0]), float(v[1])))
            except Exception:
                return None
        if len(pts) != 4:
            return None
        cu = sum(p[0] for p in pts) / 4.0
        cv = sum(p[1] for p in pts) / 4.0
        z_vals = []
        for k in ("p1", "p2", "p3", "p4", "p5"):
            v = data_dict.get(k)
            if v is None or len(v) < 3:
                continue
            try:
                zz = float(v[2])
                if np.isfinite(zz):
                    z_vals.append(zz)
            except Exception:
                continue
        cz = float(np.mean(z_vals)) if z_vals else float("nan")
        return cu, cv, cz

    def _draw_calib_points_only(self, bgr, data_dict, panel_index: int = 1):
        try:
            import cv2
        except Exception:
            return bgr
        if not isinstance(data_dict, dict):
            return bgr
        panel = 2 if int(panel_index) == 2 else 1
        pts_grid = self._calib_last_grid_pts_2 if panel == 2 else self._calib_last_grid_pts_1
        grid_status = self._calib_last_grid_status_2 if panel == 2 else self._calib_last_grid_status_1
        # 체커보드 그물 구조(인접 코너 연결선)
        if pts_grid is not None:
            try:
                grid = np.asarray(pts_grid, dtype=np.float32)
                rows, cols = grid.shape[:2]
                # OpenCV 체커보드는 내부 코너만 반환하므로, 표시용으로 바깥 1칸을 외삽해
                # 실제 보드의 가장자리까지 포함한 그물 구조를 그린다.
                if rows >= 2 and cols >= 2:
                    expanded = np.zeros((rows + 2, cols + 2, 2), dtype=np.float32)
                    expanded[1 : rows + 1, 1 : cols + 1] = grid
                    # 내부 코너 기반이므로 바깥 가장자리선은 반칸(0.5 step) 외삽으로 그린다.
                    expanded[0, 1 : cols + 1] = 1.5 * grid[0, :] - 0.5 * grid[1, :]
                    expanded[rows + 1, 1 : cols + 1] = 1.5 * grid[rows - 1, :] - 0.5 * grid[rows - 2, :]
                    expanded[:, 0] = 1.5 * expanded[:, 1] - 0.5 * expanded[:, 2]
                    expanded[:, cols + 1] = 1.5 * expanded[:, cols] - 0.5 * expanded[:, cols - 1]
                    grid = expanded
                    rows, cols = grid.shape[:2]
                for rr in range(rows):
                    for cc in range(cols - 1):
                        p1 = grid[rr, cc]
                        p2 = grid[rr, cc + 1]
                        cv2.line(
                            bgr,
                            (int(round(float(p1[0]))), int(round(float(p1[1])))),
                            (int(round(float(p2[0]))), int(round(float(p2[1])))),
                            (0, 220, 0),
                            1,
                            cv2.LINE_AA,
                        )
                for cc in range(cols):
                    for rr in range(rows - 1):
                        p1 = grid[rr, cc]
                        p2 = grid[rr + 1, cc]
                        cv2.line(
                            bgr,
                            (int(round(float(p1[0]))), int(round(float(p1[1])))),
                            (int(round(float(p2[0]))), int(round(float(p2[1])))),
                            (0, 220, 0),
                            1,
                            cv2.LINE_AA,
                        )
            except Exception:
                pass
        # 실제 켈 시퀀스 데이터 취득점 상태 표시: 정상=초록, 비정상=빨강
        if grid_status:
            try:
                for u_f, v_f, ok_pt in list(grid_status):
                    cx = int(round(float(u_f)))
                    cy = int(round(float(v_f)))
                    color = (0, 220, 0) if bool(ok_pt) else (0, 0, 255)
                    cv2.circle(bgr, (cx, cy), 2, color, -1, cv2.LINE_AA)
            except Exception:
                pass
        center_uvz = self._extract_calib_center_uvz(data_dict)
        if center_uvz is None:
            return bgr
        cu, cv, _ = center_uvz
        cx = int(round(cu))
        cy = int(round(cv))
        # 중심점 1개만 강조
        cv2.circle(bgr, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)
        return bgr

    def _draw_vision_update_ms_overlay(self, bgr):
        try:
            import cv2
        except Exception:
            return bgr
        h, w = bgr.shape[:2]
        if self._vision_cycle_ms is None or (not np.isfinite(self._vision_cycle_ms)):
            txt = "-ms"
        else:
            txt = f"{self._vision_cycle_ms:.1f}ms"
        fscale = 0.52
        thick = 1
        tsize, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fscale, thick)
        x = max(6, w - tsize[0] - 10)
        y = 20
        cv2.putText(
            bgr,
            txt,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            fscale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            txt,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            fscale,
            (0, 255, 0),
            thick,
            cv2.LINE_AA,
        )
        return bgr

    def _draw_robot_frame_icon_overlay(self, bgr):
        try:
            import cv2
        except Exception:
            return bgr
        h, w = bgr.shape[:2]
        if h < 90 or w < 180:
            return bgr

        # 로봇 프레임 원점을 비전 좌표계 Y축 최상단 우측에 둔다.
        cx = int(w * 0.5)
        oy = 22
        ox = min(w - 28, cx + 64)

        frame_color = (0, 255, 255)  # yellow

        cv2.putText(
            bgr,
            "[ROBOT FRAME]",
            (max(8, ox - 82), max(18, oy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            frame_color,
            2,
            cv2.LINE_AA,
        )

        # axis origin
        cv2.circle(bgr, (ox, oy), 2, frame_color, -1, cv2.LINE_AA)
        # draw X+/Y+ directions from current vision->robot transform
        # row-vector model: robot = camera @ R_row + t
        # robot-axis in camera frame = columns of R_row
        r = None
        try:
            aff = getattr(self, "_vision_to_robot_affine", None)
            if aff is not None:
                arr = np.asarray(aff, dtype=np.float64)
                if arr.shape == (4, 3):
                    r = arr[:3, :]
        except Exception:
            r = None

        def _dir2_from_cam3(v3):
            vx = float(v3[0])
            vy = float(v3[1])
            n = float(np.hypot(vx, vy))
            if n < 1e-9:
                return None
            return np.array([vx / n, vy / n], dtype=np.float64)

        x2 = None
        y2 = None
        if r is not None:
            x_cam = r[:, 0]
            y_cam = r[:, 1]
            x2 = _dir2_from_cam3(x_cam)
            y2 = _dir2_from_cam3(y_cam)

        if x2 is None:
            x2 = np.array([1.0, 0.0], dtype=np.float64)
        if y2 is None:
            y2 = np.array([0.0, 1.0], dtype=np.float64)

        len_x = 62.0
        len_y = 50.0
        x_end = (int(round(ox + x2[0] * len_x)), int(round(oy + x2[1] * len_x)))
        y_end = (int(round(ox + y2[0] * len_y)), int(round(oy + y2[1] * len_y)))

        cv2.arrowedLine(bgr, (ox, oy), x_end, frame_color, 2, cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(bgr, (ox, oy), y_end, frame_color, 2, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(
            bgr,
            "X+",
            (x_end[0] + 4, x_end[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            frame_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            "Y+",
            (y_end[0] + 4, y_end[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            frame_color,
            2,
            cv2.LINE_AA,
        )
        return bgr

    def _calib_overlay_and_points(self, bgr, panel_index: int = 1, draw_overlay: bool = True):
        try:
            import cv2
        except Exception:
            return bgr
        now = time.monotonic()
        panel = 2 if int(panel_index) == 2 else 1
        if panel == 2:
            last_points = self._calib_last_points_uvz_mm_2
            last_points_at = self._calib_last_points_at_2
            last_detect_run_at = float(getattr(self, "_calib_last_detect_run_at_2", 0.0) or 0.0)
        else:
            last_points = self._calib_last_points_uvz_mm_1
            last_points_at = self._calib_last_points_at_1
            last_detect_run_at = float(getattr(self, "_calib_last_detect_run_at_1", 0.0) or 0.0)
        detect_interval_sec = max(0.0, float(CALIB_DETECT_INTERVAL_SEC))
        if detect_interval_sec > 0.0 and (now - last_detect_run_at) < detect_interval_sec:
            if draw_overlay:
                if last_points is not None:
                    bgr = self._draw_calib_points_only(bgr, last_points, panel_index=panel)
                bgr = self._draw_calib_info_overlay(bgr, last_points)
            return bgr
        if panel == 2:
            self._calib_last_detect_run_at_2 = now
        else:
            self._calib_last_detect_run_at_1 = now
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # 큰 해상도 프레임은 축소 탐색 후 좌표를 복원해 처리한다.
        h, w = gray.shape[:2]
        scale = 1.0
        max_side = max(h, w)
        if max_side > 960:
            scale = 960.0 / float(max_side)
            small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = gray
        pts_grid = self._detect_calib_checkerboard(small)
        if pts_grid is not None and scale != 1.0:
            pts_grid = pts_grid / float(scale)
        if pts_grid is None:
            # 한두 프레임 미검출로 상태가 깜박이지 않도록 짧게 유지한다.
            if (
                last_points is not None
                and last_points_at is not None
                and (now - float(last_points_at)) <= CALIB_DETECTION_HOLD_SEC
            ):
                if panel == 2:
                    self._calib_last_reason_2 = "체커보드 추적중"
                else:
                    self._calib_last_reason_1 = "체커보드 추적중"
                self._calib_last_reason = "체커보드 추적중"
                bgr = self._draw_calib_points_only(bgr, last_points, panel_index=panel)
                bgr = self._draw_calib_info_overlay(bgr, last_points)
                return bgr
            if panel == 2:
                self._calib_last_points_uvz_mm_2 = None
                self._calib_last_points_at_2 = None
                self._calib_last_grid_pts_2 = None
                self._calib_last_grid_status_2 = None
                self._calib_last_reason_2 = "체커보드 미검출"
                self._calib_full_ready_2 = False
                self._calib_full_reason_2 = "체커보드 미검출"
            else:
                self._calib_last_points_uvz_mm_1 = None
                self._calib_last_points_at_1 = None
                self._calib_last_grid_pts_1 = None
                self._calib_last_grid_status_1 = None
                self._calib_last_reason_1 = "체커보드 미검출"
                self._calib_full_ready_1 = False
                self._calib_full_reason_1 = "체커보드 미검출"
            self._calib_last_reason = "체커보드 미검출"
            self._calib_last_points_uvz_mm = self._calib_last_points_uvz_mm_1
            self._calib_last_points_at = self._calib_last_points_at_1
            self.calibration_ui_refresh_requested.emit()
            if draw_overlay:
                bgr = self._draw_calib_info_overlay(bgr, None)
            return bgr

        o_tl, o_tr, o_bl, o_br = self._calib_order_outer_screen(self._calib_outer_from_inner(pts_grid))
        center = self._calib_line_intersection(o_tl, o_br, o_tr, o_bl)
        if center is None:
            center = 0.25 * (o_tl + o_tr + o_bl + o_br)
        p5 = self._calib_pick_p5_upper_right(pts_grid, center)

        names = [("p1", o_br), ("p2", o_bl), ("p3", o_tr), ("p4", o_tl), ("p5", p5)]
        data = {}
        invalid_depth_count = 0
        for name, p in names:
            u_f = float(p[0])
            v_f = float(p[1])
            u = int(round(u_f))
            v = int(round(v_f))
            z_m = self._depth_m_from_image_coord(u, v, panel_index=panel)
            if z_m is None:
                z_mm = float("nan")
                invalid_depth_count += 1
            else:
                z_mm = float(z_m * 1000.0)
            data[name] = [u_f, v_f, float(z_mm)]
        center_u = float(center[0])
        center_v = float(center[1])
        center_z_m = self._depth_m_from_image_coord(int(round(center_u)), int(round(center_v)), panel_index=panel)
        if center_z_m is None:
            center_z_mm = float("nan")
        else:
            center_z_mm = float(center_z_m * 1000.0)
        data["center"] = [center_u, center_v, center_z_mm]
        reason_txt = "깊이 미수신" if invalid_depth_count > 0 else "P1~P5 + 깊이 수신 완료"

        # 켈 시퀀스에서 실제 사용하는 전체 그리드 코너 데이터의 취득 상태(정상/비정상) 산출
        grid_status = []
        grid_bad_count = 0
        try:
            flat = np.asarray(pts_grid, dtype=np.float64).reshape(-1, 2)
        except Exception:
            flat = np.empty((0, 2), dtype=np.float64)
        for pt in flat:
            u_f = float(pt[0])
            v_f = float(pt[1])
            z_m = self._depth_m_from_image_coord(int(round(u_f)), int(round(v_f)), panel_index=panel)
            ok_pt = False
            if z_m is not None and np.isfinite(z_m) and float(z_m) > 0.0:
                z_mm = float(z_m) * 1000.0
                cxyz = self._uvz_to_camera_xyz_mm(u_f, v_f, z_mm, require_intrinsics=True, panel_index=panel)
                ok_pt = cxyz is not None and np.isfinite(np.asarray(cxyz, dtype=np.float64)).all()
            if not ok_pt:
                grid_bad_count += 1
            grid_status.append((u_f, v_f, bool(ok_pt)))

        if panel == 2:
            self._calib_last_points_uvz_mm_2 = data
            self._calib_last_points_at_2 = now
            self._calib_last_grid_pts_2 = np.array(pts_grid, copy=True)
            self._calib_last_grid_status_2 = list(grid_status)
            self._calib_last_reason_2 = reason_txt
        else:
            self._calib_last_points_uvz_mm_1 = data
            self._calib_last_points_at_1 = now
            self._calib_last_grid_pts_1 = np.array(pts_grid, copy=True)
            self._calib_last_grid_status_1 = list(grid_status)
            self._calib_last_reason_1 = reason_txt
        # UI 인식 성공 기준은 "켈 시퀀스에서 실제 필요한 전체 비전 데이터 준비" 여부로 판정한다.
        full_ready = len(grid_status) > 0 and grid_bad_count == 0
        full_err = None if full_ready else f"전체 코너 데이터 미수집({grid_bad_count}/{len(grid_status)})"
        if panel == 2:
            self._calib_full_ready_2 = bool(full_ready)
            self._calib_full_reason_2 = "전체 데이터 준비 완료" if full_ready else str(full_err or "전체 데이터 미준비")
        else:
            self._calib_full_ready_1 = bool(full_ready)
            self._calib_full_reason_1 = "전체 데이터 준비 완료" if full_ready else str(full_err or "전체 데이터 미준비")
        self._calib_last_points_uvz_mm = self._calib_last_points_uvz_mm_1
        self._calib_last_points_at = self._calib_last_points_at_1
        self._calib_last_reason = reason_txt
        self.calibration_ui_refresh_requested.emit()

        if draw_overlay:
            bgr = self._draw_calib_points_only(bgr, data, panel_index=panel)
            bgr = self._draw_calib_info_overlay(bgr, data)
        return bgr

    def _on_vision_image_msg(self, msg, stream_token=None):
        if not bool(self._top_status_enabled.get("vision", True)):
            return
        try:
            decode_started_at = time.monotonic()
            if stream_token is not None and int(stream_token) != int(getattr(self, "_vision_stream_token_1", 0)):
                return
            now = time.monotonic()
            if now < float(getattr(self, "_vision_drop_frames_until_1", 0.0)):
                return

            h = int(msg.height)
            w = int(msg.width)
            step = int(msg.step)
            if h <= 0 or w <= 0:
                return

            buf = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding in ("rgb8", "bgr8"):
                need = w * 3
                if step < need or buf.size < h * step:
                    return
                arr = buf.reshape((h, step))[:, :need].reshape((h, w, 3))
                bgr = arr[:, :, ::-1] if msg.encoding == "rgb8" else arr
                bgr = np.ascontiguousarray(bgr)
                self._last_yolo_image_size = (w, h)
                bgr = self._draw_vision_axes_overlay(bgr, panel_index=1)
                if self._calibration_mode_enabled_1:
                    calib_data = self._calib_last_points_uvz_mm_1
                    if calib_data is not None:
                        bgr = self._draw_calib_points_only(bgr, calib_data, panel_index=1)
                    bgr = self._draw_calib_info_overlay(bgr, calib_data)
                rgb = np.ascontiguousarray(bgr[:, :, ::-1])
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            elif msg.encoding in ("rgba8", "bgra8"):
                need = w * 4
                if step < need or buf.size < h * step:
                    return
                arr = buf.reshape((h, step))[:, :need].reshape((h, w, 4))
                if msg.encoding == "bgra8":
                    arr = arr[:, :, [2, 1, 0, 3]]
                arr = np.ascontiguousarray(arr)
                bgr = np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
                self._last_yolo_image_size = (w, h)
                bgr = self._draw_vision_axes_overlay(bgr, panel_index=1)
                if self._calibration_mode_enabled_1:
                    calib_data = self._calib_last_points_uvz_mm_1
                    if calib_data is not None:
                        bgr = self._draw_calib_points_only(bgr, calib_data, panel_index=1)
                    bgr = self._draw_calib_info_overlay(bgr, calib_data)
                rgba = np.concatenate([bgr[:, :, ::-1], arr[:, :, 3:4]], axis=2)
                rgba = np.ascontiguousarray(rgba)
                qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
            else:
                return

            self._vision_decode_ms = (time.monotonic() - decode_started_at) * 1000.0
            self._enqueue_vision_frame(qimg, now=now, stream_token=stream_token)
        except Exception:
            return

    def _on_vision_image_msg_2(self, msg, stream_token=None):
        if not bool(self._top_status_enabled.get("vision2", True)):
            return
        try:
            decode_started_at = time.monotonic()
            if stream_token is not None and int(stream_token) != int(getattr(self, "_vision_stream_token_2", 0)):
                return
            now = time.monotonic()
            if now < float(getattr(self, "_vision_drop_frames_until_2", 0.0)):
                return
            h = int(msg.height)
            w = int(msg.width)
            step = int(msg.step)
            if h <= 0 or w <= 0:
                return

            buf = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding in ("rgb8", "bgr8"):
                need = w * 3
                if step < need or buf.size < h * step:
                    return
                arr = buf.reshape((h, step))[:, :need].reshape((h, w, 3))
                bgr = arr[:, :, ::-1] if msg.encoding == "rgb8" else arr
                bgr = np.ascontiguousarray(bgr)
                self._last_yolo_image_size_2 = (w, h)
                bgr = self._draw_vision_axes_overlay(bgr, panel_index=2)
                if self._calibration_mode_enabled_2:
                    calib_data = self._calib_last_points_uvz_mm_2
                    if calib_data is not None:
                        bgr = self._draw_calib_points_only(bgr, calib_data, panel_index=2)
                    bgr = self._draw_calib_info_overlay(bgr, calib_data)
                rgb = np.ascontiguousarray(bgr[:, :, ::-1])
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            elif msg.encoding in ("rgba8", "bgra8"):
                need = w * 4
                if step < need or buf.size < h * step:
                    return
                arr = buf.reshape((h, step))[:, :need].reshape((h, w, 4))
                if msg.encoding == "bgra8":
                    arr = arr[:, :, [2, 1, 0, 3]]
                arr = np.ascontiguousarray(arr)
                bgr = np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
                self._last_yolo_image_size_2 = (w, h)
                bgr = self._draw_vision_axes_overlay(bgr, panel_index=2)
                if self._calibration_mode_enabled_2:
                    calib_data = self._calib_last_points_uvz_mm_2
                    if calib_data is not None:
                        bgr = self._draw_calib_points_only(bgr, calib_data, panel_index=2)
                    bgr = self._draw_calib_info_overlay(bgr, calib_data)
                rgba = np.concatenate([bgr[:, :, ::-1], arr[:, :, 3:4]], axis=2)
                rgba = np.ascontiguousarray(rgba)
                qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
            else:
                return

            self._vision_decode_ms_2 = (time.monotonic() - decode_started_at) * 1000.0
            self._enqueue_vision_frame_2(qimg, now=now, stream_token=stream_token)
        except Exception:
            return

    def _on_vision_depth_msg(self, msg, stream_token=None):
        try:
            if stream_token is not None and int(stream_token) != int(getattr(self, "_vision_stream_token_1", 0)):
                return
            self._last_camera_frame_at_1 = time.monotonic()
            h = int(msg.height)
            w = int(msg.width)
            step = int(msg.step)
            if h <= 0 or w <= 0:
                return

            enc = str(msg.encoding).lower()
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            if enc in ("16uc1", "mono16"):
                need = w * 2
                if step < need or buf.size < h * step:
                    return
                arr = buf.reshape((h, step))[:, :need].reshape((h, w, 2))
                depth = arr[:, :, 0].astype(np.uint16) | (arr[:, :, 1].astype(np.uint16) << 8)
                self._vision_depth_image = np.ascontiguousarray(depth)
                self._vision_depth_encoding = "16UC1"
                self._vision_depth_shape = (h, w)
                return

            if enc == "32fc1":
                need = w * 4
                if step < need or buf.size < h * step:
                    return
                depth = np.frombuffer(msg.data, dtype=np.float32, count=h * w)
                depth = depth.reshape((h, w))
                self._vision_depth_image = np.ascontiguousarray(depth)
                self._vision_depth_encoding = "32FC1"
                self._vision_depth_shape = (h, w)
                return
        except Exception:
            return

    def _on_vision_depth_msg_2(self, msg, stream_token=None):
        try:
            if stream_token is not None and int(stream_token) != int(getattr(self, "_vision_stream_token_2", 0)):
                return
            self._last_camera_frame_at_2 = time.monotonic()
            h = int(msg.height)
            w = int(msg.width)
            step = int(msg.step)
            if h <= 0 or w <= 0:
                return

            enc = str(msg.encoding).lower()
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            if enc in ("16uc1", "mono16"):
                need = w * 2
                if step < need or buf.size < h * step:
                    return
                arr = buf.reshape((h, step))[:, :need].reshape((h, w, 2))
                depth = arr[:, :, 0].astype(np.uint16) | (arr[:, :, 1].astype(np.uint16) << 8)
                self._vision_depth_image_2 = np.ascontiguousarray(depth)
                self._vision_depth_encoding_2 = "16UC1"
                self._vision_depth_shape_2 = (h, w)
                return

            if enc == "32fc1":
                need = w * 4
                if step < need or buf.size < h * step:
                    return
                depth = np.frombuffer(msg.data, dtype=np.float32, count=h * w)
                depth = depth.reshape((h, w))
                self._vision_depth_image_2 = np.ascontiguousarray(depth)
                self._vision_depth_encoding_2 = "32FC1"
                self._vision_depth_shape_2 = (h, w)
                return
        except Exception:
            return

    def _on_vision_camera_info_msg(self, msg, stream_token=None, panel_index: int = 1):
        try:
            panel = 2 if int(panel_index) == 2 else 1
            current_token = int(getattr(self, "_vision_stream_token_2", 0)) if panel == 2 else int(getattr(self, "_vision_stream_token_1", 0))
            if stream_token is not None and int(stream_token) != current_token:
                return
            k = getattr(msg, "k", None)
            if k is None or len(k) < 9:
                return
            fx = float(k[0])
            fy = float(k[4])
            cx = float(k[2])
            cy = float(k[5])
            if fx <= 1e-9 or fy <= 1e-9:
                return
            intr = (fx, fy, cx, cy)
            if panel == 2:
                self._camera_intrinsics_2 = intr
            else:
                self._camera_intrinsics_1 = intr
                self._camera_intrinsics = intr
        except Exception:
            return

    def _uvz_to_camera_xyz_mm(self, u, v, z_mm, require_intrinsics=False, panel_index: int = 1):
        panel = 2 if int(panel_index) == 2 else 1
        intr = self._camera_intrinsics_2 if panel == 2 else self._camera_intrinsics_1
        if intr is None and panel == 1:
            intr = self._camera_intrinsics
        if intr is None:
            if require_intrinsics:
                return None
            return float(u), float(v), float(z_mm)
        fx, fy, cx, cy = intr
        z = float(z_mm)
        x = (float(u) - cx) * z / fx
        # Camera optical frame: X right(+), Y down(+), Z forward(+)
        y = (float(v) - cy) * z / fy
        return float(x), float(y), float(z)

    def _depth_m_from_image_coord(self, ix: int, iy: int, panel_index: int = 1):
        if int(panel_index) == 2:
            depth_image = self._vision_depth_image_2
            depth_shape = self._vision_depth_shape_2
            depth_encoding = self._vision_depth_encoding_2
            image_size = self._last_yolo_image_size_2
        else:
            depth_image = self._vision_depth_image
            depth_shape = self._vision_depth_shape
            depth_encoding = self._vision_depth_encoding
            image_size = self._last_yolo_image_size
        if depth_image is None or depth_shape is None:
            return None
        if image_size is None:
            return None

        src_w, src_h = image_size
        dep_h, dep_w = depth_shape
        if src_w <= 0 or src_h <= 0 or dep_w <= 0 or dep_h <= 0:
            return None

        dx = int(round(ix * (dep_w - 1) / max(1, src_w - 1)))
        dy = int(round(iy * (dep_h - 1) / max(1, src_h - 1)))
        dx = max(0, min(dep_w - 1, dx))
        dy = max(0, min(dep_h - 1, dy))

        half = 2
        x1 = max(0, dx - half)
        x2 = min(dep_w, dx + half + 1)
        y1 = max(0, dy - half)
        y2 = min(dep_h, dy + half + 1)
        roi = depth_image[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        valid = roi[np.isfinite(roi) & (roi > 0)]
        if valid.size < 3:
            return None

        z = float(np.median(valid))
        if depth_encoding == "16UC1":
            z = z / 1000.0
        elif depth_encoding == "32FC1":
            z = z
        else:
            return None

        if z < 0.05 or z > 5.0:
            return None
        return z

    def _on_selected_point_msg(self, msg):
        try:
            self._last_vision_point_at = time.monotonic()
            if self._vision_coord_label is not None:
                self._vision_coord_label.setText(
                    f"선택점 X,Y[m] / Z[mm] : {msg.point.x:+.4f}, {msg.point.y:+.4f}, {msg.point.z * 1000.0:.1f}"
                )
                self._vision_coord_label.show()
        except Exception:
            return

    def _on_preview_point_msg(self, msg):
        try:
            # selected_point 직후 잠깐은 선택 좌표 표기를 유지한다.
            now = time.monotonic()
            if self._last_vision_point_at is not None and (now - self._last_vision_point_at) < 0.7:
                return
            if self._vision_coord_label is not None:
                self._vision_coord_label.setText(
                    f"프리뷰 X,Y[m] / Z[mm] : {msg.point.x:+.4f}, {msg.point.y:+.4f}, {msg.point.z * 1000.0:.1f}"
                )
                self._vision_coord_label.show()
        except Exception:
            return

    def _update_yolo_view(self, image: QImage):
        self._enqueue_vision_frame(image)

    def _enqueue_vision_frame(self, image: QImage, now=None, stream_token=None):
        if not hasattr(self, "yolo_view"):
            return
        t_now = time.monotonic() if now is None else float(now)
        if self._vision_prev_update_at is not None:
            dt = t_now - self._vision_prev_update_at
            if dt > 0.0:
                self._vision_cycle_ms = dt * 1000.0
        self._vision_prev_update_at = t_now
        self._last_yolo_image_size = (image.width(), image.height())
        with self._vision_frame_lock_1:
            self._pending_yolo_qimage = image
            self._pending_vision_token_1 = int(self._vision_stream_token_1 if stream_token is None else stream_token)
            self._pending_vision_enqueued_at_1 = t_now
            self._vision_frame_pending = True
        self._last_vision_frame_at = t_now
        self._vision_state_text = "정상 수신 중"

    def _enqueue_vision_frame_2(self, image: QImage, now=None, stream_token=None):
        if not hasattr(self, "yolo_view_2"):
            return
        t_now = time.monotonic() if now is None else float(now)
        if self._vision_prev_update_at_2 is not None:
            dt = t_now - self._vision_prev_update_at_2
            if dt > 0.0:
                self._vision_cycle_ms_2 = dt * 1000.0
        self._vision_prev_update_at_2 = t_now
        self._last_yolo_image_size_2 = (image.width(), image.height())
        with self._vision_frame_lock_2:
            self._pending_yolo_qimage_2 = image
            self._pending_vision_token_2 = int(self._vision_stream_token_2 if stream_token is None else stream_token)
            self._pending_vision_enqueued_at_2 = t_now
            self._vision_frame_pending_2 = True
        self._last_vision_frame_at_2 = t_now
        self._vision_state_text_2 = "정상 수신 중"

    def _drain_pending_vision_frame_1(self):
        with self._vision_frame_lock_1:
            image1 = self._pending_yolo_qimage if self._vision_frame_pending else None
            token1 = self._pending_vision_token_1 if self._vision_frame_pending else 0
            enqueued_at_1 = self._pending_vision_enqueued_at_1 if self._vision_frame_pending else None
            self._pending_yolo_qimage = None
            self._pending_vision_token_1 = 0
            self._pending_vision_enqueued_at_1 = None
            self._vision_frame_pending = False
        if image1 is not None and int(token1) == int(getattr(self, "_vision_stream_token_1", 0)):
            now = time.monotonic()
            if enqueued_at_1 is not None:
                self._vision_render_delay_ms = max(0.0, (now - float(enqueued_at_1)) * 1000.0)
            if self._vision_render_prev_at is not None:
                dt = now - self._vision_render_prev_at
                if dt > 0.0:
                    self._vision_render_interval_ms = dt * 1000.0
            self._vision_render_prev_at = now
            self._last_yolo_qimage = image1
            self._render_yolo_view()
            self._update_cycle_time_labels()

    def _drain_pending_vision_frame_2(self):
        with self._vision_frame_lock_2:
            image2 = self._pending_yolo_qimage_2 if self._vision_frame_pending_2 else None
            token2 = self._pending_vision_token_2 if self._vision_frame_pending_2 else 0
            enqueued_at_2 = self._pending_vision_enqueued_at_2 if self._vision_frame_pending_2 else None
            self._pending_yolo_qimage_2 = None
            self._pending_vision_token_2 = 0
            self._pending_vision_enqueued_at_2 = None
            self._vision_frame_pending_2 = False
        if image2 is not None and int(token2) == int(getattr(self, "_vision_stream_token_2", 0)):
            now = time.monotonic()
            if enqueued_at_2 is not None:
                self._vision_render_delay_ms_2 = max(0.0, (now - float(enqueued_at_2)) * 1000.0)
            if self._vision_render_prev_at_2 is not None:
                dt = now - self._vision_render_prev_at_2
                if dt > 0.0:
                    self._vision_render_interval_ms_2 = dt * 1000.0
            self._vision_render_prev_at_2 = now
            self._last_yolo_qimage_2 = image2
            self._render_yolo_view_2()
            self._update_cycle_time_labels()

    def _render_yolo_view(self):
        if not hasattr(self, "yolo_view"):
            return
        if self._last_yolo_qimage is None:
            self.yolo_view.clear()
            self.yolo_view.setPixmap(QPixmap())
            self.yolo_view.setAlignment(Qt.AlignCenter)
            self.yolo_view.setStyleSheet("background-color: #000000; color: #f3f3f3;")
            self.yolo_view.setText("이미지 데이터 없음")
            return
        view_main = getattr(self, "yolo_view", None)
        view_w = max(1, view_main.width())
        view_h = max(1, view_main.height())
        src_orig_w = max(1, self._last_yolo_qimage.width())
        src_orig_h = max(1, self._last_yolo_qimage.height())
        img_disp = self._rotate_image_for_panel(self._last_yolo_qimage, 1)
        src_w = max(1, img_disp.width())
        src_h = max(1, img_disp.height())

        fit_scale = max(view_w / float(src_w), view_h / float(src_h))
        scale = fit_scale * float(self._yolo_zoom)
        draw_w = int(round(src_w * scale))
        draw_h = int(round(src_h * scale))
        scaled = QPixmap.fromImage(img_disp).scaled(
            max(1, draw_w), max(1, draw_h), Qt.KeepAspectRatio, Qt.FastTransformation
        )

        canvas = QPixmap(view_w, view_h)
        canvas.fill(Qt.black)
        painter_x = 0
        painter_y = 0
        crop_x = 0
        crop_y = 0
        if self._yolo_pan_x <= -999999.0 or self._yolo_pan_y <= -999999.0:
            rot = self._normalize_rotation_deg(self._vision_rotation_deg_1)
            if rot == 90:
                org_x, org_y = src_w - 1, 0
            elif rot == 180:
                org_x, org_y = src_w - 1, src_h - 1
            elif rot == 270:
                org_x, org_y = 0, src_h - 1
            else:
                org_x, org_y = 0, 0
            base_x = float(max(0, draw_w - view_w)) / 2.0
            base_y = float(max(0, draw_h - view_h)) / 2.0
            desired_x = max(0.0, min(max(0.0, float(draw_w - view_w)), (org_x * scale) - 18.0))
            desired_y = max(0.0, min(max(0.0, float(draw_h - view_h)), (org_y * scale) - 18.0))
            self._yolo_pan_x = desired_x - base_x
            self._yolo_pan_y = desired_y - base_y
        if draw_w > view_w:
            base_x = float(draw_w - view_w) / 2.0
            crop_x = int(round(base_x + self._yolo_pan_x))
            crop_x = max(0, min(draw_w - view_w, crop_x))
        else:
            painter_x = int((view_w - draw_w) / 2)
            self._yolo_pan_x = 0.0
        if draw_h > view_h:
            base_y = float(draw_h - view_h) / 2.0
            crop_y = int(round(base_y + self._yolo_pan_y))
            crop_y = max(0, min(draw_h - view_h, crop_y))
        else:
            painter_y = int((view_h - draw_h) / 2)
            self._yolo_pan_y = 0.0

        if draw_w > view_w or draw_h > view_h:
            shown = scaled.copy(crop_x, crop_y, min(view_w, draw_w), min(view_h, draw_h))
            from PyQt5.QtGui import QPainter
            p = QPainter(canvas)
            p.drawPixmap(painter_x, painter_y, shown)
            p.end()
        else:
            from PyQt5.QtGui import QPainter
            p = QPainter(canvas)
            p.drawPixmap(painter_x, painter_y, scaled)
            p.end()

        self._yolo_view_map = {
            "scale": scale,
            "draw_w": draw_w,
            "draw_h": draw_h,
            "view_w": view_w,
            "view_h": view_h,
            "crop_x": crop_x,
            "crop_y": crop_y,
            "pad_x": painter_x,
            "pad_y": painter_y,
            "src_w": src_w,
            "src_h": src_h,
            "src_orig_w": src_orig_w,
            "src_orig_h": src_orig_h,
            "rotation_deg": int(self._vision_rotation_deg_1),
        }
        self._draw_calib_text_overlay_on_canvas(canvas, panel_index=1)
        view_main.setPixmap(canvas)

    def _render_yolo_view_2(self):
        view2 = getattr(self, "yolo_view_2", None)
        if view2 is None:
            return
        if self._last_yolo_qimage_2 is None:
            view2.clear()
            view2.setPixmap(QPixmap())
            view2.setAlignment(Qt.AlignCenter)
            view2.setStyleSheet("background-color: #000000; color: #f3f3f3;")
            view2.setText("이미지 데이터 없음")
            return
        view2_w = max(1, view2.width())
        view2_h = max(1, view2.height())
        src2_orig_w = max(1, self._last_yolo_qimage_2.width())
        src2_orig_h = max(1, self._last_yolo_qimage_2.height())
        img_disp_2 = self._rotate_image_for_panel(self._last_yolo_qimage_2, 2)
        src2_w = max(1, img_disp_2.width())
        src2_h = max(1, img_disp_2.height())
        fit_scale2 = max(view2_w / float(src2_w), view2_h / float(src2_h))
        scale2 = fit_scale2 * float(self._yolo_zoom_2)
        draw2_w = int(round(src2_w * scale2))
        draw2_h = int(round(src2_h * scale2))
        scaled2 = QPixmap.fromImage(img_disp_2).scaled(
            max(1, draw2_w), max(1, draw2_h), Qt.KeepAspectRatio, Qt.FastTransformation
        )
        canvas2 = QPixmap(view2_w, view2_h)
        canvas2.fill(Qt.black)
        painter2_x = 0
        painter2_y = 0
        crop2_x = 0
        crop2_y = 0
        if self._yolo_pan_x_2 <= -999999.0 or self._yolo_pan_y_2 <= -999999.0:
            rot2 = self._normalize_rotation_deg(self._vision_rotation_deg_2)
            if rot2 == 90:
                org2_x, org2_y = src2_w - 1, 0
            elif rot2 == 180:
                org2_x, org2_y = src2_w - 1, src2_h - 1
            elif rot2 == 270:
                org2_x, org2_y = 0, src2_h - 1
            else:
                org2_x, org2_y = 0, 0
            base2_x = float(max(0, draw2_w - view2_w)) / 2.0
            base2_y = float(max(0, draw2_h - view2_h)) / 2.0
            desired2_x = max(0.0, min(max(0.0, float(draw2_w - view2_w)), (org2_x * scale2) - 18.0))
            desired2_y = max(0.0, min(max(0.0, float(draw2_h - view2_h)), (org2_y * scale2) - 18.0))
            self._yolo_pan_x_2 = desired2_x - base2_x
            self._yolo_pan_y_2 = desired2_y - base2_y
        if draw2_w > view2_w:
            base2_x = float(draw2_w - view2_w) / 2.0
            crop2_x = int(round(base2_x + self._yolo_pan_x_2))
            crop2_x = max(0, min(draw2_w - view2_w, crop2_x))
        else:
            painter2_x = int((view2_w - draw2_w) / 2)
            self._yolo_pan_x_2 = 0.0
        if draw2_h > view2_h:
            base2_y = float(draw2_h - view2_h) / 2.0
            crop2_y = int(round(base2_y + self._yolo_pan_y_2))
            crop2_y = max(0, min(draw2_h - view2_h, crop2_y))
        else:
            painter2_y = int((view2_h - draw2_h) / 2)
            self._yolo_pan_y_2 = 0.0

        from PyQt5.QtGui import QPainter
        p2 = QPainter(canvas2)
        if draw2_w > view2_w or draw2_h > view2_h:
            shown2 = scaled2.copy(crop2_x, crop2_y, min(view2_w, draw2_w), min(view2_h, draw2_h))
            p2.drawPixmap(painter2_x, painter2_y, shown2)
        else:
            p2.drawPixmap(painter2_x, painter2_y, scaled2)
        p2.end()
        self._yolo_view_map_2 = {
            "scale": scale2,
            "draw_w": draw2_w,
            "draw_h": draw2_h,
            "view_w": view2_w,
            "view_h": view2_h,
            "crop_x": crop2_x,
            "crop_y": crop2_y,
            "pad_x": painter2_x,
            "pad_y": painter2_y,
            "src_w": src2_w,
            "src_h": src2_h,
            "src_orig_w": src2_orig_w,
            "src_orig_h": src2_orig_h,
            "rotation_deg": int(self._vision_rotation_deg_2),
        }
        self._draw_calib_text_overlay_on_canvas(canvas2, panel_index=2)
        view2.setPixmap(canvas2)

    def _map_view_to_image_coords(self, vx: int, vy: int, panel_index: int = 1):
        if int(panel_index) == 2:
            image_size = self._last_yolo_image_size_2
            view_map = self._yolo_view_map_2
        else:
            image_size = self._last_yolo_image_size
            view_map = self._yolo_view_map
        if image_size is None or view_map is None:
            return None
        m = view_map
        scale = float(m["scale"])
        draw_w = int(m["draw_w"])
        draw_h = int(m["draw_h"])
        view_w = int(m["view_w"])
        view_h = int(m["view_h"])
        crop_x = int(m["crop_x"])
        crop_y = int(m["crop_y"])
        pad_x = int(m["pad_x"])
        pad_y = int(m["pad_y"])
        src_w = int(m["src_w"])
        src_h = int(m["src_h"])

        if draw_w > view_w:
            ax = vx + crop_x
        else:
            if vx < pad_x or vx >= (pad_x + draw_w):
                return None
            ax = vx - pad_x

        if draw_h > view_h:
            ay = vy + crop_y
        else:
            if vy < pad_y or vy >= (pad_y + draw_h):
                return None
            ay = vy - pad_y

        ix = int(max(0, min(src_w - 1, ax / max(1e-6, scale))))
        iy = int(max(0, min(src_h - 1, ay / max(1e-6, scale))))
        src_orig_w = int(m.get("src_orig_w", src_w))
        src_orig_h = int(m.get("src_orig_h", src_h))
        rot_deg = int(m.get("rotation_deg", 0))
        return self._map_rotated_to_source_coords(ix, iy, src_orig_w, src_orig_h, rot_deg)

    def _draw_calib_text_overlay_on_canvas(self, canvas: QPixmap, panel_index: int = 1):
        if int(panel_index) == 2:
            calib_on = bool(getattr(self, "_calibration_mode_enabled_2", False))
            data = getattr(self, "_calib_last_points_uvz_mm_2", None)
        else:
            calib_on = bool(getattr(self, "_calibration_mode_enabled_1", False))
            data = getattr(self, "_calib_last_points_uvz_mm_1", None)
        if not calib_on:
            return
        if not isinstance(data, dict) or not data:
            return
        lines = []
        center_uvz = self._extract_calib_center_uvz(data)
        center_canvas_xy = None
        if center_uvz is not None:
            u, v, z = center_uvz
            center_canvas_xy = self._map_source_to_canvas_coords(u, v, panel_index=int(panel_index))
            cxyz = self._uvz_to_camera_xyz_mm(
                u, v, z, require_intrinsics=True, panel_index=int(panel_index)
            ) if np.isfinite(z) else None
            if cxyz is not None and np.isfinite(np.asarray(cxyz, dtype=np.float64)).all():
                cx, cy, cz = cxyz
                lines.append(f"CENTER XYZ(mm): X={cx:.1f}, Y={cy:.1f}, Z={cz:.1f}")
            elif np.isfinite(z):
                lines.append(f"CENTER Z(mm): Z={z:.1f} (camera_info 미수신: XY 환산 불가)")
            else:
                lines.append("CENTER: 깊이 미수신")
        if not lines:
            return
        from PyQt5.QtGui import QPainter, QPen
        p = QPainter(canvas)
        try:
            font = QFont(UI_FONT_FAMILY, max(7, UI_TERMINAL_FONT_SIZE - 3))
            p.setFont(font)
            line_h = 13
            base_x = 8
            start_y = max(14, canvas.height() - 8 - ((len(lines) - 1) * line_h))
            shadow_pen = QPen(QColor(0, 0, 0))
            text_pen = QPen(QColor(230, 230, 230))
            for i, line in enumerate(lines):
                y = start_y + i * line_h
                p.setPen(shadow_pen)
                p.drawText(base_x + 1, y + 1, line)
                p.setPen(text_pen)
                p.drawText(base_x, y, line)
            if center_canvas_xy is not None:
                cx, cy = center_canvas_xy
                center_line = lines[0]
                tx = max(6, min(canvas.width() - 260, int(cx) + 10))
                ty = max(18, min(canvas.height() - 10, int(cy) - 8))
                p.setPen(shadow_pen)
                p.drawText(tx + 1, ty + 1, center_line)
                p.setPen(text_pen)
                p.drawText(tx, ty, center_line)
        finally:
            p.end()

    def _set_yolo_message_if_needed(self, msg: str):
        if not hasattr(self, "yolo_view"):
            return
        if "실패" in msg or "없음" in msg:
            self.yolo_view.setText(msg)
            self._vision_state_text = "오류"
        elif "시작" in msg:
            self._vision_state_text = "실행 중"
        elif "종료" in msg:
            self._vision_state_text = "종료"
        if self._last_yolo_qimage is None:
            self.yolo_view.setAlignment(Qt.AlignCenter)
            self.yolo_view.setText("비전1 프레임 대기중...")

    def _update_bottom_status(self):
        return

    def eventFilter(self, obj, event):
        panel_index = 1 if obj is self.yolo_view else (2 if obj is getattr(self, "yolo_view_2", None) else 0)
        if panel_index in (1, 2):
            log_tag = f"[비전{panel_index}]"
            if event.type() == QEvent.MouseMove:
                if panel_index == 1 and self._yolo_panning and self._yolo_pan_last_pos is not None:
                    px, py = self._yolo_pan_last_pos
                    cx, cy = event.pos().x(), event.pos().y()
                    self._yolo_pan_x -= float(cx - px)
                    self._yolo_pan_y -= float(cy - py)
                    self._yolo_pan_last_pos = (cx, cy)
                    self._render_yolo_view()
                    return True
                if panel_index == 2 and self._yolo_panning_2 and self._yolo_pan_last_pos_2 is not None:
                    px, py = self._yolo_pan_last_pos_2
                    cx, cy = event.pos().x(), event.pos().y()
                    self._yolo_pan_x_2 -= float(cx - px)
                    self._yolo_pan_y_2 -= float(cy - py)
                    self._yolo_pan_last_pos_2 = (cx, cy)
                    self._render_yolo_view_2()
                    return True
                mapped = self._map_view_to_image_coords(event.pos().x(), event.pos().y(), panel_index=panel_index)
                if mapped is None:
                    if panel_index == 1:
                        self._last_mouse_xy = None
                    else:
                        self._last_mouse_xy_2 = None
                else:
                    x, y = mapped
                    if panel_index == 1:
                        self._last_mouse_xy = (x, y)
                    else:
                        self._last_mouse_xy_2 = (x, y)
                self._update_bottom_status()
            elif event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                mapped = self._map_view_to_image_coords(event.pos().x(), event.pos().y(), panel_index=panel_index)
                if mapped is not None:
                    x, y = mapped
                    if panel_index == 1:
                        self._last_mouse_xy = (x, y)
                    else:
                        self._last_mouse_xy_2 = (x, y)
                    z_m = self._depth_m_from_image_coord(x, y, panel_index=panel_index)
                    if z_m is None:
                        if not self._vision_panel_needs_depth(panel_index):
                            self.append_log(
                                f"{log_tag} 클릭 좌표: 깊이 구독 비활성화 상태입니다. "
                                "캘리브레이션 또는 비전클릭이동 ON에서 사용하세요.\n"
                            )
                        else:
                            self.append_log(f"{log_tag} 클릭 좌표: 깊이 미수신(Z 없음)으로 mm 환산 불가\n")
                        if panel_index == 2 and self._vision_coord_label_2 is not None:
                            self._vision_coord_label_2.hide()
                    else:
                        z_mm = z_m * 1000.0
                        cam_xyz = self._uvz_to_camera_xyz_mm(x, y, z_mm, require_intrinsics=True, panel_index=panel_index)
                        if cam_xyz is None or (not np.isfinite(np.asarray(cam_xyz, dtype=np.float64)).all()):
                            self.append_log(
                                f"{log_tag} 클릭 좌표: Z={z_mm:.1f}mm "
                                f"(camera_info 미수신: 비전 XYZ(mm) 환산 불가)\n"
                            )
                        else:
                            vx, vy, vz = [float(vv) for vv in cam_xyz]
                            robot_xyz = self._vision_uvz_to_robot_xyz_mm(x, y, z_mm, panel_index=panel_index)
                            if robot_xyz is None:
                                self.append_log(
                                    f"{log_tag} 클릭 비전 XYZ(mm): X={vx:.1f}, Y={vy:.1f}, Z={vz:.1f} "
                                    f"(로봇 변환행렬 미적용/변환 실패)\n"
                                )
                            else:
                                rx, ry, rz = robot_xyz
                                z_margin = self._get_vision_move_z_margin_mm()
                                rz_target = float(rz) + float(z_margin)
                                self._last_clicked_vision_xyz_mm = (float(vx), float(vy), float(vz))
                                self._last_clicked_robot_xyz_mm = (float(rx), float(ry), float(rz))
                                self._last_clicked_robot_target_xyz_mm = (float(rx), float(ry), float(rz_target))
                                self._last_clicked_source_vision = panel_index
                                self.append_log(
                                    f"{log_tag} 클릭 비전 XYZ(mm): X={vx:.1f}, Y={vy:.1f}, Z={vz:.1f} -> "
                                    f"로봇 XYZ(mm): X={rx:.2f}, Y={ry:.2f}, Z={rz:.2f}\n"
                                )
                                if panel_index == 2 and self._vision_coord_label_2 is not None:
                                    self._vision_coord_label_2.hide()
                                if self._vision_click_dialog_enabled:
                                    dialog_text = (
                                        f"비전 클릭 좌표\n"
                                        f"- 비전 XYZ(mm): X={vx:.1f}, Y={vy:.1f}, Z={vz:.1f}\n"
                                        f"- Robot XYZ(mm): X={rx:.2f}, Y={ry:.2f}, Z={rz:.2f}\n"
                                        f"- Z 마진: +{z_margin:.1f}mm\n"
                                        f"- 이동 목표 XYZ(mm): X={rx:.2f}, Y={ry:.2f}, Z={rz_target:.2f}\n\n"
                                        f"로봇을 이동하시겠습니까?"
                                    )
                                    answer = QMessageBox.question(
                                        self,
                                        "비전 좌표 이동 확인",
                                        dialog_text,
                                        QMessageBox.Yes | QMessageBox.No,
                                        QMessageBox.No,
                                    )
                                    if answer == QMessageBox.Yes:
                                        self._run_vision_target_move(
                                            self._last_clicked_vision_xyz_mm,
                                            self._last_clicked_robot_xyz_mm,
                                            source_panel=panel_index,
                                        )
                else:
                    self.append_log(f"{log_tag} 클릭 좌표: 영상 영역 밖\n")
                self._update_bottom_status()
            elif panel_index == 1 and event.type() == QEvent.MouseButtonPress and event.button() == Qt.MiddleButton:
                self._yolo_panning = True
                self._yolo_pan_last_pos = (event.pos().x(), event.pos().y())
                return True
            elif panel_index == 2 and event.type() == QEvent.MouseButtonPress and event.button() == Qt.MiddleButton:
                self._yolo_panning_2 = True
                self._yolo_pan_last_pos_2 = (event.pos().x(), event.pos().y())
                return True
            elif panel_index == 1 and event.type() == QEvent.Wheel:
                delta = event.angleDelta().y()
                if delta > 0:
                    self._yolo_zoom = min(self._yolo_zoom_max, self._yolo_zoom * 1.12)
                elif delta < 0:
                    self._yolo_zoom = max(self._yolo_zoom_min, self._yolo_zoom / 1.12)
                if self._yolo_zoom <= (self._yolo_zoom_min + 1e-6):
                    self._yolo_pan_x = 0.0
                    self._yolo_pan_y = 0.0
                self._render_yolo_view()
                return True
            elif panel_index == 2 and event.type() == QEvent.Wheel:
                delta = event.angleDelta().y()
                if delta > 0:
                    self._yolo_zoom_2 = min(self._yolo_zoom_max, self._yolo_zoom_2 * 1.12)
                elif delta < 0:
                    self._yolo_zoom_2 = max(self._yolo_zoom_min, self._yolo_zoom_2 / 1.12)
                if self._yolo_zoom_2 <= (self._yolo_zoom_min + 1e-6):
                    self._yolo_pan_x_2 = 0.0
                    self._yolo_pan_y_2 = 0.0
                self._render_yolo_view_2()
                return True
            elif panel_index == 1 and event.type() == QEvent.MouseButtonRelease and event.button() == Qt.MiddleButton:
                self._yolo_panning = False
                self._yolo_pan_last_pos = None
                return True
            elif panel_index == 2 and event.type() == QEvent.MouseButtonRelease and event.button() == Qt.MiddleButton:
                self._yolo_panning_2 = False
                self._yolo_pan_last_pos_2 = None
                return True
            elif event.type() == QEvent.Leave:
                if panel_index == 1:
                    self._last_mouse_xy = None
                else:
                    self._last_mouse_xy_2 = None
                self._update_bottom_status()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not UI_USE_DESIGN_GEOMETRY:
            self._layout_main_frames()
        self._reposition_top_status_row()
        self._reposition_cycle_labels()
        self._layout_control_buttons()
        if hasattr(self, "yolo_view"):
            self._render_yolo_view()
        if hasattr(self, "yolo_view_2"):
            self._render_yolo_view_2()

    def _stop_yolo_camera(self):
        if self._yolo_worker is not None:
            self._yolo_worker.stop()
        if self._yolo_thread is not None:
            self._yolo_thread.quit()
            if not self._yolo_thread.wait(1200):
                # Fallback: camera worker thread can hang on device read.
                self._yolo_thread.terminate()
                self._yolo_thread.wait(500)
        self._yolo_worker = None
        self._yolo_thread = None

    def on_pick_place(self):
        if self.backend is None:
            self.append_log("[작업] 백엔드 초기화 중입니다.\n")
            return
        value, ok = QInputDialog.getInt(self, "오브젝트 번호", "0~10 입력:", 0, 0, 10, 1)
        if not ok:
            return

        ok, msg = self.backend.send_pick_place(value)
        self.append_log(msg + "\n")

    def _get_current_pose_defaults(self):
        posj = [0.0] * 6
        posx = [0.0] * 6
        if self.backend is None:
            return posj, posx
        data = None
        if hasattr(self.backend, "get_position_snapshot"):
            data, _ = self.backend.get_position_snapshot()
        elif hasattr(self.backend, "get_current_positions"):
            data = self.backend.get_current_positions()
        if data and len(data) >= 2:
            pj, px = data
            if pj is not None and len(pj) >= 6:
                posj = [float(v) for v in list(pj)[:6]]
            if px is not None and len(px) >= 6:
                posx = [float(v) for v in list(px)[:6]]
        return posj, posx

    def _ask_six_values_form(self, title, labels, defaults, limits=None, guide_text=None):
        if labels is None or len(labels) < 6:
            return "라벨 설정 오류: 6개 라벨이 필요합니다."
        dialog = QDialog(self)
        dialog.setWindowTitle(str(title))
        dialog.setModal(True)
        dialog.resize(360, 320)

        layout = QVBoxLayout(dialog)
        if guide_text:
            guide_label = QLabel(str(guide_text), dialog)
            guide_label.setWordWrap(True)
            guide_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(guide_label)

        grid = QGridLayout()
        edits = []
        for i in range(6):
            lbl = QLabel(str(labels[i]), dialog)
            edit = QLineEdit(dialog)
            edit.setStyleSheet(
                "QLineEdit {"
                " selection-background-color: #1f6feb;"
                " selection-color: #ffffff;"
                "}"
            )
            edit_palette = edit.palette()
            for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
                edit_palette.setColor(group, QPalette.Highlight, QColor("#1f6feb"))
                edit_palette.setColor(group, QPalette.HighlightedText, QColor("#ffffff"))
            edit.setPalette(edit_palette)
            v = float(defaults[i]) if defaults is not None and len(defaults) > i else 0.0
            edit.setText(f"{v:.2f}")
            edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            edit.setMaxLength(16)
            min_v, max_v = (-999999.0, 999999.0)
            if limits is not None and len(limits) > i:
                min_v, max_v = limits[i]
            validator = QDoubleValidator(float(min_v), float(max_v), 3, edit)
            validator.setNotation(QDoubleValidator.StandardNotation)
            edit.setValidator(validator)
            grid.addWidget(lbl, i, 0)
            grid.addWidget(edit, i, 1)
            edits.append(edit)
        layout.addLayout(grid)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if edits:
            edits[0].setFocus()
            edits[0].selectAll()

        if dialog.exec_() != QDialog.Accepted:
            return None

        vals = []
        for i, edit in enumerate(edits):
            raw = edit.text().strip()
            if raw == "":
                return "입력 형식 오류: 빈 칸 없이 숫자를 입력하세요."
            try:
                v = float(raw)
            except Exception:
                return "입력 형식 오류: 숫자만 입력하세요."
            if limits is not None and len(limits) > i:
                min_v, max_v = limits[i]
                if v < float(min_v) or v > float(max_v):
                    label = str(labels[i]) if labels is not None and len(labels) > i else f"항목{i+1}"
                    return f"입력 범위 오류: {label}는 {float(min_v):.1f} ~ {float(max_v):.1f} 범위여야 합니다."
            vals.append(v)
        return tuple(vals)

    def _confirm_motion_with_values(self, title, intro, labels, values, log_prefix):
        if labels is None or values is None:
            return False
        lines = [str(intro), ""]
        n = min(len(labels), len(values), 6)
        for i in range(n):
            try:
                v = float(values[i])
                lines.append(f"{labels[i]}: {v:.2f}")
            except Exception:
                lines.append(f"{labels[i]}: {values[i]}")
        msg = QMessageBox(self)
        msg.setWindowTitle(str(title))
        msg.setIcon(QMessageBox.Warning)
        msg.setText("\n".join(lines))
        msg.setTextInteractionFlags(Qt.TextSelectableByMouse)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        msg.setStyleSheet(
            "QLabel {"
            " selection-background-color: #1f6feb;"
            " selection-color: #ffffff;"
            "}"
        )
        answer = msg.exec_()
        if answer != QMessageBox.Yes:
            self.append_log(f"[{log_prefix}] 이동 취소\n")
            return False
        return True

    def on_move_xyzabc_dialog(self):
        if self.backend is None:
            self.append_log("[좌표계 무브] 백엔드 초기화 중입니다.\n")
            return
        if not hasattr(self.backend, "send_move_cartesian"):
            self.append_log("[좌표계 무브] 백엔드가 좌표계 이동 기능을 지원하지 않습니다.\n")
            return
        _posj, posx = self._get_current_pose_defaults()
        vals = self._ask_six_values_form(
            "좌표계 무브",
            ["X", "Y", "Z", "A", "B", "C"],
            posx,
            guide_text="안내: 입력한 좌표로 로봇이 실제 이동합니다.",
        )
        if vals is None:
            return
        if isinstance(vals, str):
            self.append_log(f"[좌표계 무브] {vals}\n")
            return
        x, y, z, a, b, c = vals
        if not self._confirm_motion_with_values(
            "좌표계 무브 확인",
            "안내: 로봇이 실제로 이동합니다.\n좌표계로 이동할까요? 목표 위치는 아래와 같습니다.",
            ["X", "Y", "Z", "A", "B", "C"],
            (x, y, z, a, b, c),
            "좌표계 무브",
        ):
            return
        ok, msg = self.backend.send_move_cartesian(x, y, z, a, b, c)
        self.append_log(f"[좌표계 무브] {msg}\n")

    def on_move_joint_dialog(self):
        if self.backend is None:
            self.append_log("[조인트 무브] 백엔드 초기화 중입니다.\n")
            return
        if not hasattr(self.backend, "send_move_joint"):
            self.append_log("[조인트 무브] 백엔드가 조인트 이동 기능을 지원하지 않습니다.\n")
            return
        posj, _posx = self._get_current_pose_defaults()
        vals = self._ask_six_values_form(
            "조인트 무브",
            ["J1", "J2", "J3", "J4", "J5", "J6"],
            posj,
            limits=JOINT_INPUT_LIMITS_DEG,
            guide_text="안내: 입력한 조인트 각도로 로봇이 실제 이동합니다.",
        )
        if vals is None:
            return
        if isinstance(vals, str):
            self.append_log(f"[조인트 무브] {vals}\n")
            return
        j1, j2, j3, j4, j5, j6 = vals
        if not self._confirm_motion_with_values(
            "조인트 무브 확인",
            "안내: 로봇이 실제로 이동합니다.\n조인트로 이동할까요? 목표 위치는 아래와 같습니다.",
            ["J1", "J2", "J3", "J4", "J5", "J6"],
            (j1, j2, j3, j4, j5, j6),
            "조인트 무브",
        ):
            return
        ok, msg = self.backend.send_move_joint(j1, j2, j3, j4, j5, j6)
        self.append_log(f"[조인트 무브] {msg}\n")

    def on_print_positions(self):
        if self.backend is None:
            self.append_log("[좌표] 백엔드 초기화 중입니다.\n")
            return
        if hasattr(self.backend, "get_position_snapshot"):
            data, seen_at = self.backend.get_position_snapshot()
        else:
            data = self.backend.get_current_positions()
            seen_at = None

        if not data:
            self.append_log("[좌표] 수신된 좌표가 없습니다.\n")
            return
        if seen_at is not None and (time.monotonic() - seen_at) > POSITION_STALE_SEC:
            self.append_log("[좌표] 좌표 수신이 지연되어 출력하지 않습니다.\n")
            return

        posj, posx = data
        j_text = ", ".join(f"J{i+1}={self._fmt_ui_float(posj[i], 2)}" for i in range(6))
        p_text = ", ".join(
            (
                f"X={self._fmt_ui_float(posx[0], 2)}",
                f"Y={self._fmt_ui_float(posx[1], 2)}",
                f"Z={self._fmt_ui_float(posx[2], 2)}",
                f"A={self._fmt_ui_float(posx[3], 2)}",
                f"B={self._fmt_ui_float(posx[4], 2)}",
                f"C={self._fmt_ui_float(posx[5], 2)}",
            )
        )
        self.append_log(f"[좌표] {j_text}\n")
        self.append_log(f"[좌표] {p_text}\n")

        self.append_log("[좌표] 기준소스: RobotState snapshot(current_posx 우선)\n")

    def on_reset_robot(self):
        self._start_reset_async()

    def on_move_home(self):
        if self.backend is None:
            self.append_log("[홈] 백엔드 초기화 중입니다.\n")
            return
        if not hasattr(self.backend, "send_move_home"):
            self.append_log("[홈] 백엔드 홈 이동 기능을 지원하지 않습니다.\n")
            return
        if hasattr(self.backend, "get_home_posj"):
            target = tuple(float(v) for v in list(self.backend.get_home_posj())[:6])
        else:
            target = tuple(float(v) for v in list(HOME_POSJ)[:6])
        if not self._confirm_motion_with_values(
            "홈위치이동 확인",
            "홈위치로 이동할까요? 목표 위치는 아래와 같습니다.",
            ["J1", "J2", "J3", "J4", "J5", "J6"],
            target,
            "홈",
        ):
            return
        ok, msg = self.backend.send_move_home()
        self.append_log(f"[홈] {msg}\n")

    def on_save_home_position_dialog(self):
        if self.backend is None:
            self.append_log("[홈저장] 백엔드 초기화 중입니다.\n")
            return
        if not hasattr(self.backend, "set_home_posj"):
            self.append_log("[홈저장] 백엔드 홈 저장 기능을 지원하지 않습니다.\n")
            return
        posj, _posx = self._get_current_pose_defaults()
        vals = self._ask_six_values_form(
            "홈위치 저장",
            ["J1", "J2", "J3", "J4", "J5", "J6"],
            posj,
            limits=JOINT_INPUT_LIMITS_DEG,
            guide_text="안내: 현재/입력한 조인트 각도를 홈위치로 저장합니다.",
        )
        if vals is None:
            self.append_log("[홈저장] 저장 취소\n")
            return
        if isinstance(vals, str):
            self.append_log(f"[홈저장] {vals}\n")
            return
        ok, msg = self.backend.set_home_posj(vals)
        self.append_log(f"[홈저장] {msg}\n")

    def on_gripper_move(self):
        if self.backend is None:
            self.append_log("[그리퍼] 백엔드 초기화 중입니다.\n")
            return
        if not hasattr(self.backend, "send_gripper_move"):
            self.append_log("[그리퍼] 백엔드 그리퍼 수동 이동 기능을 지원하지 않습니다.\n")
            return
        if hasattr(self.backend, "get_robot_mode_snapshot"):
            mode_value, mode_seen_at = self.backend.get_robot_mode_snapshot()
            if mode_seen_at is None or mode_value is None:
                self.append_log("[그리퍼] 로봇모드 확인 중입니다. 잠시 후 다시 시도하세요.\n")
                return
            if int(mode_value) != 1:
                self.append_log("[그리퍼] 오토모드(1)에서만 실행할 수 있습니다.\n")
                return
        raw = ""
        if hasattr(self, "_gripper_stroke_input") and self._gripper_stroke_input is not None:
            raw = self._gripper_stroke_input.text().strip()
        if not raw:
            self.append_log("[그리퍼] 벌어진 거리(mm)를 입력하세요. (0~109)\n")
            return
        try:
            distance_mm = float(raw)
        except ValueError:
            self.append_log("[그리퍼] 거리 값은 숫자(mm)여야 합니다. (0~109)\n")
            return
        ok, msg = self.backend.send_gripper_move(distance_mm)
        self.append_log(f"[그리퍼] {msg}\n")

    def closeEvent(self, event):
        self._closing = True
        try:
            if hasattr(self, "_log_timer") and self._log_timer is not None:
                self._log_timer.stop()
            if hasattr(self, "_status_timer") and self._status_timer is not None:
                self._status_timer.stop()
            if hasattr(self, "_top_status_anim_timer") and self._top_status_anim_timer is not None:
                self._top_status_anim_timer.stop()
            if hasattr(self, "_pos_timer") and self._pos_timer is not None:
                self._pos_timer.stop()

            if self._reset_thread is not None:
                self._reset_thread.quit()
                if not self._reset_thread.wait(800):
                    self._reset_thread.terminate()
                    self._reset_thread.wait(300)
            self._stop_yolo_camera()
            self._stop_calibration_process(1)
            self._stop_calibration_process(2)
            self._teardown_external_vision_bridge()
            self._stop_external_vision_process()

            if self._backend_thread is not None:
                self._backend_thread.quit()
                if not self._backend_thread.wait(800):
                    self._backend_thread.terminate()
                    self._backend_thread.wait(300)
            if self.backend is not None:
                try:
                    self.backend.shutdown()
                except Exception:
                    pass
        finally:
            event.accept()
            super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont(UI_FONT_FAMILY, UI_FONT_SIZE))
    palette = app.palette()
    for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
        palette.setColor(group, QPalette.Highlight, QColor("#1f6feb"))
        palette.setColor(group, QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(
        f"""
        * {{
            font-family: '{UI_FONT_FAMILY}';
            font-size: {UI_FONT_SIZE}pt;
        }}
        QWidget, QDialog, QMessageBox {{
            selection-background-color: #1f6feb;
            selection-color: #ffffff;
        }}
        QLineEdit, QTextEdit, QPlainTextEdit, QTextBrowser, QLabel {{
            selection-background-color: #1f6feb;
            selection-color: #ffffff;
        }}
        """
    )

    window = App(backend=None, auto_start_backend=True)
    window.setFont(QFont(UI_FONT_FAMILY, UI_FONT_SIZE))
    window.show()
    window.raise_()
    window.activateWindow()
    window.setFocus(Qt.ActiveWindowFocusReason)
    sys.exit(app.exec_())
