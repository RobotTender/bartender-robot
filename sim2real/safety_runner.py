from __future__ import annotations

import argparse

from .core import ActionController, PolicyConfig, parse_comma_floats
from .runner import build_robot_io, create_arg_parser, run_preflight_motion_check


def create_safety_arg_parser() -> argparse.ArgumentParser:
    parser = create_arg_parser()
    parser.description = "Run standalone safety motion check for sim2real integration"
    parser.add_argument(
        "--safety-motion-mode",
        type=str,
        choices=["hold", "joint_delta"],
        default="hold",
        help="안전 점검 모션 방식",
    )
    parser.add_argument(
        "--safety-motion-joint-number",
        type=int,
        default=6,
        help="joint_delta 모드에서 이동할 조인트 번호(1~6)",
    )
    parser.add_argument(
        "--safety-motion-delta-deg",
        type=float,
        default=10.0,
        help="joint_delta 모드에서 적용할 목표 증분(deg)",
    )
    parser.add_argument(
        "--safety-motion-command-sec",
        type=float,
        default=5.0,
        help="안전 점검 모션 명령 유지 시간(s)",
    )
    parser.add_argument(
        "--safety-motion-return-to-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="joint_delta 모드 후 시작 자세로 복귀할지 여부",
    )
    parser.add_argument(
        "--safety-motion-return-sec",
        type=float,
        default=0.5,
        help="복귀 명령 유지 시간(s)",
    )
    return parser


def run_safety(args: argparse.Namespace) -> None:
    cfg = PolicyConfig()
    loop_hz = float(args.hz)
    if loop_hz <= 0.0:
        raise ValueError(f"--hz는 0보다 커야 합니다. 입력={args.hz}")
    cfg.set_control_hz(loop_hz)

    joint_lower = parse_comma_floats(args.joint_lower_rad, expected_len=6)
    joint_upper = parse_comma_floats(args.joint_upper_rad, expected_len=6)
    controller = ActionController(cfg=cfg, joint_lower_rad=joint_lower, joint_upper_rad=joint_upper)
    robot_io = build_robot_io(args.mode, class_index=args.object_class, args=args)

    robot_io.connect()
    try:
        initial_robot = robot_io.read_robot_state()
        controller.reset(
            current_joint_pos_rad=initial_robot.joint_pos_rad,
            current_gripper_close_ratio=initial_robot.gripper_close_ratio,
        )
        loop_dt = 1.0 / loop_hz
        run_preflight_motion_check(
            robot_io=robot_io,
            controller=controller,
            loop_dt=loop_dt,
            mode=str(args.safety_motion_mode),
            joint_number=int(args.safety_motion_joint_number),
            delta_deg=float(args.safety_motion_delta_deg),
            command_sec=float(args.safety_motion_command_sec),
            return_to_base=bool(args.safety_motion_return_to_base),
            return_sec=float(args.safety_motion_return_sec),
        )
        print("[safety] done")
    finally:
        robot_io.shutdown()


def main() -> None:
    parser = create_safety_arg_parser()
    args = parser.parse_args()
    run_safety(args)
