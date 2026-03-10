#!/usr/bin/env python3
import os
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Optional alias envs for future user-frontend naming.
if os.environ.get("VOICE_ORDER_WEBUI_HOST") is None and os.environ.get("USER_FRONTEND_HOST") is not None:
    os.environ["VOICE_ORDER_WEBUI_HOST"] = str(os.environ.get("USER_FRONTEND_HOST"))
if os.environ.get("VOICE_ORDER_WEBUI_PORT") is None and os.environ.get("USER_FRONTEND_PORT") is not None:
    os.environ["VOICE_ORDER_WEBUI_PORT"] = str(os.environ.get("USER_FRONTEND_PORT"))
if os.environ.get("VOICE_ORDER_WEBUI_ENABLED") is None and os.environ.get("USER_FRONTEND_ENABLED") is not None:
    os.environ["VOICE_ORDER_WEBUI_ENABLED"] = str(os.environ.get("USER_FRONTEND_ENABLED"))

from order_integration.voice_order_webui import main as web_main


if __name__ == "__main__":
    raise SystemExit(web_main(sys.argv[1:]))
