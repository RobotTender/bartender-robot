from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .core import (
    DEFAULT_OBS_JOINT_DIM,
    ActionController,
    ActorPolicy,
    ObservationBuilder,
    PolicyConfig,
    compute_obs_extra_dim,
    parse_comma_floats,
    parse_int_csv,
    parse_vec3,
)
from .io import (
    DummyRobotIO,
    ModbusGripperConfig,
    ROS2RobotIO,
    RealRobotConfig,
    RealRobotIO,
    RobotIO,
    Ros2Config,
)


def parse_float_csv(raw: str) -> tuple[float, ...]:
    text = str(raw).strip()
    if not text:
        return ()
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def run_preflight_cycle_check(
    *,
    robot_io: RobotIO,
    obs_builder: ObservationBuilder,
    policy: ActorPolicy,
    controller: ActionController,
    device: torch.device,
    loop_dt: float,
    verbose: bool,
) -> None:
    """Run one dry cycle timing check before the main loop.

    It measures read->obs->policy->controller latency without sending commands.
    Controller state is restored after the check.
    """
    joint_targets_snapshot = None if controller.joint_targets is None else controller.joint_targets.copy()
    gripper_target_snapshot = float(controller.gripper_target)

    started = time.perf_counter()
    robot_state = robot_io.read_robot_state()
    object_state = robot_io.read_object_state()
    obs = obs_builder.build(robot_state, object_state)
    action = policy.act(obs, device=device)
    controller.step(action)
    elapsed = time.perf_counter() - started

    if joint_targets_snapshot is not None:
        controller.joint_targets = joint_targets_snapshot
    controller.gripper_target = gripper_target_snapshot

    budget_ms = loop_dt * 1000.0
    elapsed_ms = elapsed * 1000.0
    overrun_ms = (elapsed - loop_dt) * 1000.0
    if overrun_ms > 0.0:
        print(
            f"[warn] preflight cycle overrun: +{overrun_ms:.3f} ms "
            f"(elapsed={elapsed_ms:.3f} ms, budget={budget_ms:.3f} ms)"
        )
    elif verbose:
        margin_ms = budget_ms - elapsed_ms
        print(
            f"[preflight] cycle OK: elapsed={elapsed_ms:.3f} ms, "
            f"budget={budget_ms:.3f} ms, margin={margin_ms:.3f} ms"
        )


def run_preflight_motion_check(
    *,
    robot_io: RobotIO,
    controller: ActionController,
    loop_dt: float,
    mode: str,
    joint_number: int,
    delta_deg: float,
    command_sec: float,
    return_to_base: bool,
    return_sec: float,
) -> None:
    """Run one safe motion sanity check before entering policy loop."""
    robot_state = robot_io.read_robot_state()
    base_joint = np.asarray(robot_state.joint_pos_rad, dtype=np.float32).copy()
    gripper_ratio = float(np.clip(robot_state.gripper_close_ratio, 0.0, 1.0))
    if base_joint.shape[0] < 6:
        raise RuntimeError(f"preflight motion check 실패: joint 길이 부족({base_joint.shape[0]})")

    target_joint = base_joint.copy()
    mode_key = str(mode).lower().strip()
    if mode_key == "joint_delta":
        jn = int(np.clip(int(joint_number), 1, 6))
        idx = jn - 1
        target_joint[idx] += np.deg2rad(float(delta_deg))
        target_joint = np.clip(
            target_joint,
            controller.joint_lower[: target_joint.shape[0]],
            controller.joint_upper[: target_joint.shape[0]],
        )
        applied_delta_deg = float(np.rad2deg(target_joint[idx] - base_joint[idx]))
        print(
            f"[preflight-motion] mode=joint_delta joint={jn} "
            f"req_delta={float(delta_deg):.3f}deg applied_delta={applied_delta_deg:.3f}deg"
        )
    elif mode_key == "hold":
        print("[preflight-motion] mode=hold (현재 자세 유지 명령)")
    else:
        raise ValueError(f"지원하지 않는 preflight motion mode: {mode}")

    send_steps = max(1, int(round(float(command_sec) / max(loop_dt, 1.0e-6))))
    for _ in range(send_steps):
        robot_io.send_joint_targets(target_joint, gripper_ratio)
        time.sleep(loop_dt)

    if mode_key == "joint_delta" and bool(return_to_base):
        return_steps = max(1, int(round(float(return_sec) / max(loop_dt, 1.0e-6))))
        print("[preflight-motion] returning to base pose")
        for _ in range(return_steps):
            robot_io.send_joint_targets(base_joint, gripper_ratio)
            time.sleep(loop_dt)


def build_robot_io(mode: str, class_index: int, args: argparse.Namespace) -> RobotIO:
    mode_key = {"virtual": "dummy", "sim": "dummy"}.get(str(mode).lower(), str(mode).lower())
    object_fixed_custom_extra = parse_float_csv(args.object_fixed_custom_extra)

    if mode_key == "dummy":
        return DummyRobotIO(class_index=class_index, fixed_custom_extra=object_fixed_custom_extra)

    if mode_key == "ros2":
        ros2_cfg = Ros2Config(
            node_name=str(args.ros2_node_name),
            spin_timeout_s=float(args.ros2_spin_timeout_s),
            init_wait_s=float(args.ros2_init_wait_s),
            joint_state_topic=str(args.ros2_joint_state_topic),
            ee_pose_topic=str(args.ros2_ee_pose_topic),
            object_pose_topic=str(args.ros2_object_pose_topic),
            object_class_topic=str(args.ros2_object_class_topic),
            object_extra_topic=str(args.ros2_object_extra_topic),
            gripper_state_topic=str(args.ros2_gripper_state_topic),
            joint_cmd_topic=str(args.ros2_joint_cmd_topic),
            gripper_cmd_topic=str(args.ros2_gripper_cmd_topic),
            enable_gripper_output=bool(args.ros2_enable_gripper_output),
            gripper_command_mode=str(args.ros2_gripper_command_mode),
            drl_namespace=str(args.ros2_drl_namespace),
            drl_robot_system=int(args.ros2_drl_robot_system),
            drl_service_timeout_s=float(args.ros2_drl_service_timeout_s),
            drl_call_timeout_s=float(args.ros2_drl_call_timeout_s),
            drl_stroke_open=int(args.ros2_drl_stroke_open),
            drl_stroke_close=int(args.ros2_drl_stroke_close),
            drl_deadband_stroke=int(args.ros2_drl_deadband_stroke),
            drl_min_command_interval_s=float(args.ros2_drl_min_command_interval_s),
            drl_slave_id=int(args.ros2_drl_slave_id),
            drl_baudrate=int(args.ros2_drl_baudrate),
            drl_torque_enable_addr=int(args.ros2_drl_torque_enable_addr),
            drl_goal_current_addr=int(args.ros2_drl_goal_current_addr),
            drl_goal_position_addr=int(args.ros2_drl_goal_position_addr),
            drl_init_goal_current=int(args.ros2_drl_init_goal_current),
            fallback_object_xyz_m=parse_vec3(args.object_fixed_xyz_m),
            fallback_object_up_xyz=parse_vec3(args.object_fixed_up_xyz),
            fallback_object_custom_extra=object_fixed_custom_extra,
            fallback_object_class_index=int(class_index),
        )
        return ROS2RobotIO(cfg=ros2_cfg)

    if mode_key == "real":
        gripper_cfg = ModbusGripperConfig(
            protocol=str(args.gripper_protocol),
            enabled=bool(args.gripper_enabled),
            auto_register=bool(args.gripper_auto_register),
            ip=str(args.gripper_ip),
            port=int(args.gripper_port),
            slave_id=int(args.gripper_slave_id),
            cmd_signal_name=str(args.gripper_cmd_signal),
            cmd_reg_index=int(args.gripper_cmd_reg_index),
            pos_signal_name=str(args.gripper_pos_signal),
            pos_reg_index=int(args.gripper_pos_reg_index),
            open_raw=int(args.gripper_open_raw),
            close_raw=int(args.gripper_close_raw),
            command_deadband_raw=int(args.gripper_deadband_raw),
            min_write_interval_s=float(args.gripper_min_write_interval),
            rx_timeout_s=float(args.gripper_rx_timeout_s),
            ack_timeout_s=float(args.gripper_ack_timeout_s),
            serial_baudrate=int(args.gripper_serial_baudrate),
            serial_bytesize=int(args.gripper_serial_bytesize),
            serial_parity=str(args.gripper_serial_parity),
            serial_stopbits=int(args.gripper_serial_stopbits),
            serial_probe_retries=int(args.gripper_serial_probe_retries),
            serial_probe_wait_s=float(args.gripper_serial_probe_wait_s),
            reg_operating_mode=int(args.gripper_reg_operating_mode),
            reg_torque_enable=int(args.gripper_reg_torque_enable),
            reg_goal_current=int(args.gripper_reg_goal_current),
            reg_goal_velocity=int(args.gripper_reg_goal_velocity),
            reg_goal_position=int(args.gripper_reg_goal_position),
            reg_present_position=int(args.gripper_reg_present_position),
            operating_mode_value=int(args.gripper_operating_mode_value),
            init_goal_current=int(args.gripper_init_goal_current),
            init_goal_velocity=int(args.gripper_init_goal_velocity),
            position_word_order=str(args.gripper_position_word_order),
        )
        real_cfg = RealRobotConfig(
            robot_id=str(args.doosan_robot_id),
            robot_model=str(args.doosan_model),
            robot_host=str(args.doosan_host),
            robot_port=int(args.doosan_port),
            servo_time_s=float(args.servo_time_s),
            servo_vel_deg_s=float(args.servo_vel_deg_s),
            servo_acc_deg_s2=float(args.servo_acc_deg_s2),
            use_posx_orientation=bool(args.use_posx_orientation),
            object_source=str(args.object_source),
            object_json_path=str(args.object_json_path),
            object_fixed_xyz_m=parse_vec3(args.object_fixed_xyz_m),
            object_fixed_up_xyz=parse_vec3(args.object_fixed_up_xyz),
            object_fixed_custom_extra=object_fixed_custom_extra,
            object_class_index=int(class_index),
            gripper=gripper_cfg,
        )
        return RealRobotIO(cfg=real_cfg)

    raise ValueError(f"알 수 없는 모드: {mode}")


def run(args: argparse.Namespace) -> None:
    cfg = PolicyConfig()
    device = torch.device(args.device)
    hidden_override = parse_int_csv(args.policy_hidden_sizes) if args.policy_hidden_sizes else None
    policy = ActorPolicy.from_checkpoint(
        Path(args.checkpoint),
        device=device,
        activation=args.policy_activation,
        obs_dim_override=args.policy_obs_dim,
        act_dim_override=args.policy_act_dim,
        hidden_sizes_override=hidden_override,
    )

    if bool(args.print_policy_info):
        print(f"[policy] checkpoint={args.checkpoint}")
        print(f"[policy] {policy.summary()}")

    joint_lower = parse_comma_floats(args.joint_lower_rad, expected_len=6)
    joint_upper = parse_comma_floats(args.joint_upper_rad, expected_len=6)

    robot_io = build_robot_io(args.mode, class_index=args.object_class, args=args)
    obs_extra_dim = compute_obs_extra_dim(
        include_to_object=bool(args.obs_include_to_object),
        include_lift=bool(args.obs_include_lift),
        include_gripper_state=bool(args.obs_include_gripper_state),
        include_object_class=bool(args.obs_include_object_class),
        object_class_dim=int(args.obs_object_class_dim),
        include_object_up_z=bool(args.obs_include_object_up_z),
        custom_extra_dim=int(args.obs_custom_extra_dim),
    )
    if args.obs_joint_dim is None:
        remaining = int(policy.obs_dim) - int(obs_extra_dim)
        if remaining >= 2 and remaining % 2 == 0:
            obs_joint_dim = remaining // 2
        else:
            obs_joint_dim = DEFAULT_OBS_JOINT_DIM
            print(
                f"[warn] policy.obs_dim={policy.obs_dim}에서 obs_joint_dim 자동 추론 실패. "
                f"(obs_extra_dim={obs_extra_dim}) default={DEFAULT_OBS_JOINT_DIM} 사용"
            )
    else:
        obs_joint_dim = int(args.obs_joint_dim)

    obs_builder = ObservationBuilder(
        cfg=cfg,
        joint_lower_rad=joint_lower,
        joint_upper_rad=joint_upper,
        obs_joint_dim=obs_joint_dim,
        target_obs_dim=int(policy.obs_dim),
        include_to_object=bool(args.obs_include_to_object),
        include_lift=bool(args.obs_include_lift),
        include_gripper_state=bool(args.obs_include_gripper_state),
        include_object_class=bool(args.obs_include_object_class),
        object_class_dim=int(args.obs_object_class_dim),
        include_object_up_z=bool(args.obs_include_object_up_z),
        custom_extra_dim=int(args.obs_custom_extra_dim),
    )
    controller = ActionController(cfg=cfg, joint_lower_rad=joint_lower, joint_upper_rad=joint_upper)

    robot_io.connect()
    try:
        initial_robot = robot_io.read_robot_state()
        initial_object = robot_io.read_object_state()
        obs_builder.set_object_reference(initial_object)
        controller.reset(
            current_joint_pos_rad=initial_robot.joint_pos_rad,
            current_gripper_close_ratio=initial_robot.gripper_close_ratio,
        )

        loop_hz = float(args.hz)
        loop_dt = 1.0 / loop_hz
        step = 0
        if bool(args.preflight_cycle_check):
            run_preflight_cycle_check(
                robot_io=robot_io,
                obs_builder=obs_builder,
                policy=policy,
                controller=controller,
                device=device,
                loop_dt=loop_dt,
                verbose=bool(args.preflight_cycle_check_verbose),
            )

        while args.max_steps <= 0 or step < args.max_steps:
            started = time.perf_counter()

            robot_state = robot_io.read_robot_state()
            object_state = robot_io.read_object_state()
            obs = obs_builder.build(robot_state, object_state)
            action = policy.act(obs, device=device)
            joint_targets, gripper_target = controller.step(action)
            robot_io.send_joint_targets(joint_targets, gripper_target)

            if step % max(1, int(loop_hz)) == 0:
                print(
                    f"[step={step:06d}] action={np.array2string(action, precision=3)} "
                    f"gripper={gripper_target:.3f} obj_z={object_state.position_m[2]:.3f}"
                )

            elapsed = time.perf_counter() - started
            time.sleep(max(0.0, loop_dt - elapsed))
            step += 1
    finally:
        robot_io.shutdown()


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sim2real inference with model_1000.pt")
    parser.add_argument("--checkpoint", type=str, default="model_1000.pt", help="PyTorch checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dummy", "virtual", "sim", "ros2", "real"],
        default="dummy",
        help="I/O backend mode (dummy/virtual/sim are same)",
    )
    parser.add_argument("--hz", type=float, default=60.0, help="Control frequency")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run forever")
    parser.add_argument(
        "--preflight-cycle-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="메인 루프 시작 전 1회 주기 시간(preflight) 체크 실행",
    )
    parser.add_argument(
        "--preflight-cycle-check-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="preflight가 예산 이내여도 결과 출력",
    )
    parser.add_argument("--object-class", type=int, default=0, help="0=soju, 1=orange, 2=beer")
    parser.add_argument("--print-policy-info", action="store_true", help="로드된 policy 구조 정보 출력")
    parser.add_argument(
        "--policy-activation",
        type=str,
        choices=["elu", "relu", "tanh", "silu"],
        default="elu",
        help="policy 활성화 함수(체크포인트 구조와 동일해야 함)",
    )
    parser.add_argument("--policy-hidden-sizes", type=str, default="", help="예: 256,128,64 (미지정 시 체크포인트 자동 추론)")
    parser.add_argument("--policy-obs-dim", type=int, default=None, help="policy 입력 차원 강제 지정(미지정 시 자동)")
    parser.add_argument("--policy-act-dim", type=int, default=None, help="policy 출력 차원 강제 지정(미지정 시 자동)")
    parser.add_argument(
        "--obs-joint-dim",
        type=int,
        default=None,
        help=(
            "관측에서 조인트 pos/vel 각각의 차원. "
            "미지정(None) 시 policy.obs_dim과 관측 구성 옵션에서 자동 추론"
        ),
    )
    parser.add_argument(
        "--obs-include-to-object",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="관측에 to_object(dx,dy,dz) 포함",
    )
    parser.add_argument(
        "--obs-include-lift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="관측에 lift 포함",
    )
    parser.add_argument(
        "--obs-include-gripper-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="관측에 gripper state(0~1) 포함",
    )
    parser.add_argument(
        "--obs-include-object-class",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="관측에 object class one-hot 포함",
    )
    parser.add_argument(
        "--obs-object-class-dim",
        type=int,
        default=3,
        help="object class one-hot 차원 (obs-include-object-class=true일 때 사용)",
    )
    parser.add_argument(
        "--obs-include-object-up-z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="관측에 object up z 포함",
    )
    parser.add_argument(
        "--obs-custom-extra-dim",
        type=int,
        default=0,
        help="관측 끝에 추가할 custom extra scalar 개수",
    )

    parser.add_argument("--ros2-node-name", type=str, default="sim2real_policy_runner", help="ROS2 node name")
    parser.add_argument("--ros2-spin-timeout-s", type=float, default=0.01, help="ROS2 spin_once timeout")
    parser.add_argument("--ros2-init-wait-s", type=float, default=3.0, help="초기 토픽 수신 대기 시간")
    parser.add_argument("--ros2-joint-state-topic", type=str, default="/joint_states", help="sensor_msgs/JointState")
    parser.add_argument("--ros2-ee-pose-topic", type=str, default="/ee_pose", help="geometry_msgs/PoseStamped")
    parser.add_argument("--ros2-object-pose-topic", type=str, default="/object_pose", help="geometry_msgs/PoseStamped")
    parser.add_argument("--ros2-object-class-topic", type=str, default="/object_class", help="std_msgs/Int32")
    parser.add_argument(
        "--ros2-object-extra-topic",
        type=str,
        default="/object_extra",
        help="std_msgs/Float32MultiArray (custom extra vector)",
    )
    parser.add_argument("--ros2-gripper-state-topic", type=str, default="/gripper/state", help="std_msgs/Float32")
    parser.add_argument("--ros2-joint-cmd-topic", type=str, default="/arm/joint_position_cmd", help="sensor_msgs/JointState")
    parser.add_argument("--ros2-gripper-cmd-topic", type=str, default="/gripper/command", help="std_msgs/Float32")
    parser.add_argument(
        "--ros2-enable-gripper-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ROS2 gripper 명령 publish/DRL 호출 활성화 (외부 그리퍼 노드가 제어하면 끄기)",
    )
    parser.add_argument(
        "--ros2-gripper-command-mode",
        type=str,
        choices=["topic", "drl_service"],
        default="topic",
        help="ROS2에서 그리퍼 명령 전달 방식",
    )
    parser.add_argument("--ros2-drl-namespace", type=str, default="dsr01", help="DrlStart 서비스 namespace")
    parser.add_argument("--ros2-drl-robot-system", type=int, default=0, help="DrlStart robot_system 값")
    parser.add_argument("--ros2-drl-service-timeout-s", type=float, default=2.0, help="DrlStart 서비스 대기 timeout")
    parser.add_argument("--ros2-drl-call-timeout-s", type=float, default=5.0, help="DrlStart 호출 timeout")
    parser.add_argument("--ros2-drl-stroke-open", type=int, default=0, help="close_ratio=0일 때 stroke")
    parser.add_argument("--ros2-drl-stroke-close", type=int, default=750, help="close_ratio=1일 때 stroke")
    parser.add_argument("--ros2-drl-deadband-stroke", type=int, default=5, help="DRL 명령 deadband")
    parser.add_argument("--ros2-drl-min-command-interval-s", type=float, default=0.2, help="DRL 명령 최소 주기")
    parser.add_argument("--ros2-drl-slave-id", type=int, default=1, help="DRL modbus slave id")
    parser.add_argument("--ros2-drl-baudrate", type=int, default=57600, help="DRL flange serial baudrate")
    parser.add_argument("--ros2-drl-torque-enable-addr", type=int, default=256, help="DRL torque enable register")
    parser.add_argument("--ros2-drl-goal-current-addr", type=int, default=275, help="DRL goal current register")
    parser.add_argument("--ros2-drl-goal-position-addr", type=int, default=282, help="DRL goal position register")
    parser.add_argument("--ros2-drl-init-goal-current", type=int, default=400, help="DRL init goal current")

    parser.add_argument("--doosan-robot-id", type=str, default="dsr01", help="Doosan robot ID")
    parser.add_argument("--doosan-model", type=str, default="e0509", help="Doosan model name")
    parser.add_argument(
        "--doosan-host",
        type=str,
        default="",
        help="Doosan controller host/IP (빈값이면 SDK 기본 연결 설정 사용)",
    )
    parser.add_argument(
        "--doosan-port",
        type=int,
        default=0,
        help="Doosan controller port (0이면 SDK 기본값 사용)",
    )
    parser.add_argument("--servo-time-s", type=float, default=1.0 / 60.0, help="servoj 도달 시간(s)")
    parser.add_argument("--servo-vel-deg-s", type=float, default=90.0, help="servoj 속도 제한(deg/s)")
    parser.add_argument("--servo-acc-deg-s2", type=float, default=180.0, help="servoj 가속도 제한(deg/s^2)")
    parser.add_argument("--use-posx-orientation", action="store_true", help="get_current_posx 자세를 quaternion에 반영")
    parser.add_argument("--object-source", type=str, choices=["fixed", "json"], default="fixed", help="실물 오브젝트 상태 소스")
    parser.add_argument("--object-json-path", type=str, default="/tmp/object_state.json", help="object-source=json일 때 입력 파일")
    parser.add_argument(
        "--object-fixed-xyz-m",
        type=str,
        default="0.0,0.65,1.30",
        help="object-source=fixed일 때 오브젝트 중심[m], x,y,z",
    )
    parser.add_argument(
        "--object-fixed-up-xyz",
        type=str,
        default="0.0,0.0,1.0",
        help="object-source=fixed일 때 오브젝트 up 벡터, x,y,z",
    )
    parser.add_argument(
        "--object-fixed-custom-extra",
        type=str,
        default="",
        help="object-source=fixed fallback custom extra, comma-separated (예: 0.11,0.02)",
    )
    parser.add_argument(
        "--gripper-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Modbus gripper 제어 활성화",
    )
    parser.add_argument(
        "--gripper-protocol",
        type=str,
        choices=["flange_serial_fc", "modbus_tcp_signal"],
        default="flange_serial_fc",
        help="그리퍼 통신 백엔드",
    )
    parser.add_argument(
        "--gripper-auto-register",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="protocol=modbus_tcp_signal일 때 add_modbus_signal 자동 등록",
    )
    parser.add_argument("--gripper-ip", type=str, default="192.168.137.2", help="ModbusTCP IP")
    parser.add_argument("--gripper-port", type=int, default=502, help="ModbusTCP port")
    parser.add_argument("--gripper-slave-id", type=int, default=1, help="Modbus slave id")
    parser.add_argument("--gripper-cmd-signal", type=str, default="rh_cmd", help="출력 명령 signal name")
    parser.add_argument("--gripper-cmd-reg-index", type=int, default=0, help="출력 register index")
    parser.add_argument("--gripper-pos-signal", type=str, default="rh_pos", help="입력 위치 signal name")
    parser.add_argument("--gripper-pos-reg-index", type=int, default=1, help="입력 register index")
    parser.add_argument("--gripper-open-raw", type=int, default=0, help="완전 오픈 raw 값")
    parser.add_argument("--gripper-close-raw", type=int, default=750, help="완전 클로즈 raw 값")
    parser.add_argument("--gripper-deadband-raw", type=int, default=2, help="명령 재전송 deadband(raw)")
    parser.add_argument("--gripper-min-write-interval", type=float, default=0.05, help="최소 쓰기 주기(s)")
    parser.add_argument("--gripper-rx-timeout-s", type=float, default=0.5, help="FC03 read timeout")
    parser.add_argument("--gripper-ack-timeout-s", type=float, default=0.1, help="FC06/FC16 ack timeout")
    parser.add_argument("--gripper-serial-baudrate", type=int, default=57600, help="flange serial baudrate")
    parser.add_argument("--gripper-serial-bytesize", type=int, default=8, help="flange serial bytesize")
    parser.add_argument("--gripper-serial-parity", type=str, default="N", help="flange serial parity (N/E/O)")
    parser.add_argument("--gripper-serial-stopbits", type=int, default=1, help="flange serial stopbits")
    parser.add_argument("--gripper-serial-probe-retries", type=int, default=5, help="serial open/probe 재시도 횟수")
    parser.add_argument("--gripper-serial-probe-wait-s", type=float, default=0.1, help="probe 재시도 간 대기")
    parser.add_argument("--gripper-reg-operating-mode", type=int, default=5, help="FC06 operating mode register")
    parser.add_argument("--gripper-reg-torque-enable", type=int, default=256, help="FC06 torque enable register")
    parser.add_argument("--gripper-reg-goal-current", type=int, default=275, help="FC06 goal current register")
    parser.add_argument("--gripper-reg-goal-velocity", type=int, default=276, help="FC06 goal velocity register")
    parser.add_argument("--gripper-reg-goal-position", type=int, default=282, help="FC16 goal position start register")
    parser.add_argument("--gripper-reg-present-position", type=int, default=290, help="FC03 present position register")
    parser.add_argument("--gripper-operating-mode-value", type=int, default=(5 << 8), help="operating mode write value")
    parser.add_argument("--gripper-init-goal-current", type=int, default=200, help="init 시 goal current 값")
    parser.add_argument("--gripper-init-goal-velocity", type=int, default=-1, help="init 시 goal velocity 값 (-1=skip)")
    parser.add_argument(
        "--gripper-position-word-order",
        type=str,
        choices=["LO_HI", "HI_LO"],
        default="LO_HI",
        help="present position 32-bit 워드 순서",
    )
    parser.add_argument(
        "--joint-lower-rad",
        type=str,
        default="-2.0071287,-3.1415927,-3.1415927,-3.1415927,-3.1415927,-3.1415927",
        help="6개 조인트 하한(rad), comma-separated",
    )
    parser.add_argument(
        "--joint-upper-rad",
        type=str,
        default="2.0071287,3.1415927,3.1415927,3.1415927,3.1415927,3.1415927",
        help="6개 조인트 상한(rad), comma-separated",
    )
    return parser


def main() -> None:
    parser = create_arg_parser()
    args = parser.parse_args()
    run(args)
