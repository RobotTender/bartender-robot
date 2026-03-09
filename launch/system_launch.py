#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchService
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetLaunchConfiguration,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _parse_cli_overrides(argv):
    overrides = {}
    for token in argv or []:
        if ':=' not in token:
            continue
        key, value = token.split(':=', 1)
        key = key.strip()
        if not key:
            continue
        if key == 'run_ui':
            key = 'run_frontend'
        overrides[key] = value
    run_robot = str(overrides.get('run_robot', 'true')).strip().lower()
    if run_robot != 'true' and 'robot_mode' not in overrides:
        overrides['robot_mode'] = 'virtual'
    mode = str(overrides.get('robot_mode', '')).strip().lower()
    if mode == 'virtual' and 'robot_host' not in overrides:
        overrides['robot_host'] = '127.0.0.1'
    return overrides


def _is_true_text(value, default=False):
    text = str(value if value is not None else '').strip().lower()
    if text in ('1', 'true', 'yes', 'on'):
        return True
    if text in ('0', 'false', 'no', 'off'):
        return False
    return bool(default)


def _collect_processes_by_pattern(pattern):
    try:
        output = subprocess.check_output(['pgrep', '-af', pattern], text=True)
    except Exception:
        return {}
    found = {}
    for raw in str(output).splitlines():
        line = str(raw).strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except Exception:
            continue
        if pid == os.getpid():
            continue
        found[pid] = parts[1] if len(parts) > 1 else ''
    return found


def _wait_for_exit(pids, timeout_sec):
    remaining = set(int(pid) for pid in pids)
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while remaining and time.monotonic() < deadline:
        for pid in list(remaining):
            try:
                os.kill(pid, 0)
            except OSError:
                remaining.discard(pid)
        if remaining:
            time.sleep(0.05)
    return remaining


def _cleanup_lingering_gazebo_processes(enabled=True):
    if not enabled:
        return
    patterns = (
        r'ruby .*/gz sim',
        r'ruby .*/ign gazebo',
        r'gzserver',
        r'gzclient',
        r'gazebo( |$)',
    )
    found = {}
    for pattern in patterns:
        found.update(_collect_processes_by_pattern(pattern))
    if not found:
        return
    remaining = set(found.keys())
    for sig, wait_sec in (
        (signal.SIGINT, 1.0),
        (signal.SIGTERM, 1.0),
        (signal.SIGKILL, 0.2),
    ):
        if not remaining:
            break
        for pid in list(remaining):
            try:
                os.kill(pid, sig)
            except OSError:
                remaining.discard(pid)
        remaining = _wait_for_exit(remaining, wait_sec)
    cleaned = sorted(set(found.keys()) - set(remaining))
    if cleaned:
        print(f"[system_launch] Gazebo 잔여 프로세스 정리: {', '.join(str(pid) for pid in cleaned)}")
    if remaining:
        print(f"[system_launch] Gazebo 잔여 프로세스 정리 실패: {', '.join(str(pid) for pid in sorted(remaining))}")


def generate_launch_description(overrides=None):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, '..'))
    doosan_share = get_package_share_directory('dsr_bringup2')

    args = [
        DeclareLaunchArgument('run_robot', default_value='true'),
        DeclareLaunchArgument('robot_mode', default_value='real'),
        DeclareLaunchArgument('robot_host', default_value='110.120.1.68'),
        DeclareLaunchArgument('robot_model', default_value='e0509'),
        DeclareLaunchArgument('robot_gz', default_value='true'),
        DeclareLaunchArgument('run_sensors', default_value='true'),
        DeclareLaunchArgument('run_frontend', default_value='true'),
    ]

    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(doosan_share, 'launch', 'dsr_bringup2_gazebo.launch.py')
        ),
        launch_arguments={
            'mode': LaunchConfiguration('robot_mode'),
            'host': LaunchConfiguration('robot_host'),
            'model': LaunchConfiguration('robot_model'),
            'gz': LaunchConfiguration('robot_gz'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('run_robot')),
    )

    sensor_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(this_dir, 'realsense_launch.py')),
        condition=IfCondition(LaunchConfiguration('run_sensors')),
    )

    frontend_proc = ExecuteProcess(
        cmd=[sys.executable, os.path.join(project_root, 'src', 'frontend', 'developer_frontend.py')],
        output='screen',
        emulate_tty=True,
        additional_env={
            'BARTENDER_ROBOT_MODE_HINT': LaunchConfiguration('robot_mode'),
            'BARTENDER_ROBOT_MODEL_HINT': LaunchConfiguration('robot_model'),
            'BARTENDER_ROBOT_HOST_HINT': LaunchConfiguration('robot_host'),
        },
        condition=IfCondition(LaunchConfiguration('run_frontend')),
    )

    shutdown_on_frontend_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=frontend_proc,
            on_exit=[EmitEvent(event=Shutdown(reason='developer_frontend exited'))],
        )
    )

    override_actions = [SetLaunchConfiguration(name, value) for name, value in (overrides or {}).items()]

    return LaunchDescription(
        args + override_actions + [robot_launch, sensor_launch, frontend_proc, shutdown_on_frontend_exit]
    )


def main(argv=None):
    argv = argv or []
    overrides = _parse_cli_overrides(argv)
    ls = LaunchService(argv=argv)
    ls.include_launch_description(generate_launch_description(overrides))
    cleanup_gazebo = bool(
        _is_true_text((overrides or {}).get('run_robot', 'true'), default=True)
        and _is_true_text((overrides or {}).get('robot_gz', 'true'), default=True)
    )
    try:
        return ls.run()
    finally:
        _cleanup_lingering_gazebo_processes(enabled=cleanup_gazebo)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
