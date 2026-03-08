#!/usr/bin/env python3
import importlib.util
import sys
from pathlib import Path

SYSTEM_LAUNCH_PATH = Path(__file__).resolve().parent / "launch" / "system_launch.py"
spec = importlib.util.spec_from_file_location("bartender_system_launch", SYSTEM_LAUNCH_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)
main = module.main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
