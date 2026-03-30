#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = ROOT / "llm"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
WORKSPACE_DIR = ROOT.parent.parent


def _workspace_setup_candidates() -> list[Path]:
    return [
        ROOT / "robot" / "install" / "setup.bash",
        WORKSPACE_DIR / "install" / "setup.bash",
    ]


def _build_workspace_source_prefix() -> str:
    parts = ["source /opt/ros/jazzy/setup.bash"]
    for candidate in _workspace_setup_candidates():
        if candidate.exists():
            parts.append(f"source {sh_quote(str(candidate))}")
            break
    return " && ".join(parts)


@dataclass
class ManagedProcess:
    name: str
    cmd: str
    cwd: Path
    process: subprocess.Popen[str] | None = None


def _build_startup_command() -> str:
    python_bin = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    return (
        f"{_build_workspace_source_prefix()} && "
        f"export PYTHONPATH={sh_quote(str(ROOT / 'robot' / 'src' / 'bartender_test'))}:$PYTHONPATH && "
        f"exec {sh_quote(python_bin)} -m bartender_test.startup"
    )

def _build_gripper_command() -> str:
    python_bin = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    return (
        f"{_build_workspace_source_prefix()} && "
        f"export PYTHONPATH={sh_quote(str(ROOT / 'robot' / 'src' / 'bartender_test'))}:$PYTHONPATH && "
        f"exec {sh_quote(python_bin)} -m bartender_test.gripper"
    )

def _build_pick_command() -> str:
    python_bin = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    return (
        f"{_build_workspace_source_prefix()} && "
        f"export PYTHONPATH={sh_quote(str(ROOT / 'robot' / 'src' / 'bartender_test'))}:$PYTHONPATH && "
        f"exec {sh_quote(python_bin)} -m bartender_test.pick"
    )

def _build_pour_command() -> str:
    python_bin = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    return (
        f"{_build_workspace_source_prefix()} && "
        f"export PYTHONPATH={sh_quote(str(ROOT / 'robot' / 'src' / 'bartender_test'))}:$PYTHONPATH && "
        f"exec {sh_quote(python_bin)} -m bartender_test.pour"
    )

def _build_place_command() -> str:
    python_bin = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    return (
        f"{_build_workspace_source_prefix()} && "
        f"export PYTHONPATH={sh_quote(str(ROOT / 'robot' / 'src' / 'bartender_test'))}:$PYTHONPATH && "
        f"exec {sh_quote(python_bin)} -m bartender_test.place"
    )

def _build_snap_command() -> str:
    python_bin = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    return (
        f"{_build_workspace_source_prefix()} && "
        f"export PYTHONPATH={sh_quote(str(ROOT / 'robot' / 'src' / 'bartender_test'))}:$PYTHONPATH && "
        f"exec {sh_quote(python_bin)} -m bartender_test.snap"
    )

def _build_volume_detection_command() -> str:
    python_bin = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    return (
        f"{_build_workspace_source_prefix()} && "
        f"exec {sh_quote(python_bin)} detection/realsense_cam2.py"
    )

def _build_web_command(host: str, port: int) -> str:
    python_bin = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    return f"exec {sh_quote(python_bin)} manage.py runserver {host}:{port}"

def sh_quote(value: str) -> str:
    return shlex.quote(value)

def start_process(spec: ManagedProcess) -> None:
    stdout = None
    stderr = None

    if spec.name in ("camera", "camera2"):
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL

    # Use exec to replace the bash shell with the intended command where possible
    cmd = spec.cmd
    if " && " not in cmd and ";" not in cmd and not cmd.strip().startswith("exec "):
        cmd = f"exec {cmd}"

    spec.process = subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=str(spec.cwd),
        stdout=stdout,
        stderr=stderr,
        text=True,
        start_new_session=True,
        env=os.environ.copy(),
    )
    print(f"[start] {spec.name}: {cmd}")

def stop_process(spec: ManagedProcess, sig: signal.Signals = signal.SIGINT) -> None:
    if spec.process is None or spec.process.poll() is not None:
        return

    try:
        pgid = os.getpgid(spec.process.pid)
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        return

def main() -> int:
    parser = argparse.ArgumentParser(description="Start bartender web + pick stack together.")
    parser.add_argument("--with-bringup", action="store_true", help="Also run robot bringup command.")
    parser.add_argument(
        "--bringup-cmd",
        default="ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=real host:=110.120.1.68 model:=e0509",
        help="Command used when --with-bringup is set.",
    )
    parser.add_argument(
        "--camera-cmd",
        default="ros2 launch realsense2_camera rs_launch.py camera_name:=camera_1 serial_no:=_311322302867 enable_pointcloud:=true align_depth.enable:=true enable_rgbd:=true",
        help="Optional camera command to run alongside the stack.",
    )
    parser.add_argument(
        "--camera2-cmd",
        default="ros2 launch realsense2_camera rs_launch.py camera_name:=camera_2 serial_no:=_313522301601 enable_pointcloud:=true align_depth.enable:=true",
        help="Camera 2 (Volume Detection) hardware command.",
    )
    parser.add_argument("--skip-web", action="store_true", help="Do not start Django server.")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8000)
    args = parser.parse_args()

    base_processes: list[ManagedProcess] = []

    if args.with_bringup:
        base_processes.append(ManagedProcess("bringup", args.bringup_cmd, ROOT))

    if args.camera_cmd:
        base_processes.append(ManagedProcess("camera", args.camera_cmd, ROOT))

    if args.camera2_cmd:
        base_processes.append(ManagedProcess("camera2", args.camera2_cmd, ROOT))

    # Always start volume detection unless we want to remove it too, 
    # but the user only mentioned pick/pour.
    base_processes.append(ManagedProcess("volume_detection", _build_volume_detection_command(), ROOT))

    if not base_processes and args.skip_web:
        print("Nothing to start.")
        return 1

    def handle_signal(signum, frame):
        del signum, frame
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    started_processes: list[ManagedProcess] = []
    return_code = 0

    try:
        # 1. Start base processes (bringup, cameras)
        for spec in base_processes:
            start_process(spec)
            started_processes.append(spec)
            time.sleep(0.3)

        # 2. Run startup synchronously (This starts logic nodes)
        startup_cmd = _build_startup_command()
        print(f"\n[start] startup (blocking): {startup_cmd}")

        if args.with_bringup:
            print("Waiting 3s for bringup to settle...")
            time.sleep(3.0)

        startup_proc = subprocess.run(
            ["bash", "-lc", startup_cmd],
            cwd=str(ROOT),
            env=os.environ.copy()
        )
        if startup_proc.returncode != 0:
            if startup_proc.returncode < 0: raise KeyboardInterrupt
            raise RuntimeError(f"startup exited with code {startup_proc.returncode}")
        print("[ok] startup completed. Logic nodes (gripper, pick, pour, place) are now running.")

        # 3. Start web server
        if not args.skip_web:
            web_spec = ManagedProcess("web", _build_web_command(args.web_host, args.web_port), LLM_DIR)
            start_process(web_spec)
            started_processes.append(web_spec)

        print("\n[ready] stack is running. Press Ctrl+C to stop all processes.")
        while True:
            time.sleep(1.0)
            for spec in started_processes:
                if spec.process is not None and spec.process.poll() is not None:
                    raise RuntimeError(f"{spec.name} exited with code {spec.process.returncode}")
    except KeyboardInterrupt:
        print("\n[stop] received interrupt, stopping stack...")
    except Exception as exc:
        print(f"\n[error] {exc}")
        return_code = 1
    else:
        return_code = 0
    finally:
        print("[stop] cleaning up persistent logic nodes...")
        subprocess.run(["pkill", "-9", "-f", r"python3 -m bartender_test\.(gripper)"], stderr=subprocess.DEVNULL)

        # Graceful shutdown
        for spec in reversed(started_processes):
            stop_process(spec, signal.SIGINT)
        
        # Wait for processes to exit
        for spec in reversed(started_processes):
            if spec.process is not None:
                try:
                    spec.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(f"[warn] {spec.name} (pid={spec.process.pid}) did not stop gracefully, sending SIGTERM...")
                    stop_process(spec, signal.SIGTERM)
                    try:
                        spec.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"[warn] {spec.name} (pid={spec.process.pid}) still alive, sending SIGKILL...")
                        stop_process(spec, signal.SIGKILL)
                        spec.process.wait(timeout=5)
    
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
