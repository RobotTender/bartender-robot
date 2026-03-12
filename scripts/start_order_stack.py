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


def _build_pick_command() -> str:
    return (
        f"{_build_workspace_source_prefix()} && "
        f"export PYTHONPATH={sh_quote(str(ROOT / 'robot' / 'src' / 'bartender_test'))}:$PYTHONPATH && "
        "python3 -m bartender_test.pick"
    )


def _build_web_command(host: str, port: int) -> str:
    python_bin = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    return f"{sh_quote(python_bin)} manage.py runserver {host}:{port}"


def sh_quote(value: str) -> str:
    return shlex.quote(value)


def start_process(spec: ManagedProcess) -> None:
    spec.process = subprocess.Popen(
        ["bash", "-lc", spec.cmd],
        cwd=str(spec.cwd),
        stdout=None,
        stderr=None,
        text=True,
        start_new_session=True,
        env=os.environ.copy(),
    )
    print(f"[start] {spec.name}: {spec.cmd}")


def stop_process(spec: ManagedProcess) -> None:
    if spec.process is None or spec.process.poll() is not None:
        return

    try:
        os.killpg(os.getpgid(spec.process.pid), signal.SIGINT)
    except ProcessLookupError:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Start bartender web + pick stack together.")
    parser.add_argument("--with-bringup", action="store_true", help="Also run robot bringup command.")
    parser.add_argument(
        "--bringup-cmd",
        default="ros2 launch dsr_bringup2 dsr_bringup2.launch.py mode:=real host:=110.120.1.59 model:=e0509",
        help="Command used when --with-bringup is set.",
    )
    parser.add_argument(
        "--camera-cmd",
        default="ros2 launch realsense2_camera rs_launch.py camera_name:=camera_1 serial_no:=_311322302867 enable_pointcloud:=true align_depth.enable:=true enable_rgbd:=true",
        help="Optional camera command to run alongside the stack.",
    )
    parser.add_argument("--skip-web", action="store_true", help="Do not start Django server.")
    parser.add_argument("--skip-pick", action="store_true", help="Do not start pick.py node.")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8000)
    args = parser.parse_args()

    processes: list[ManagedProcess] = []

    if args.with_bringup:
        processes.append(ManagedProcess("bringup", args.bringup_cmd, ROOT))

    if args.camera_cmd:
        processes.append(ManagedProcess("camera", args.camera_cmd, ROOT))

    if not args.skip_pick:
        processes.append(ManagedProcess("pick", _build_pick_command(), ROOT))

    if not args.skip_web:
        processes.append(ManagedProcess("web", _build_web_command(args.web_host, args.web_port), LLM_DIR))

    if not processes:
        print("Nothing to start.")
        return 1

    def handle_signal(signum, frame):
        del signum, frame
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        for spec in processes:
            start_process(spec)
            time.sleep(0.3)

        print("[ready] stack is running. Press Ctrl+C to stop all processes.")
        while True:
            time.sleep(1.0)
            for spec in processes:
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
        for spec in reversed(processes):
            stop_process(spec)
        for spec in reversed(processes):
            if spec.process is not None:
                try:
                    spec.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(spec.process.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
