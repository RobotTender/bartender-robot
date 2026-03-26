from .core import ObjectState, RobotState

_API_IMPORT_ERROR: Exception | None = None
try:
    from .api import Sim2RealRuntime, Sim2RealRuntimeConfig
except Exception as exc:  # pragma: no cover
    _API_IMPORT_ERROR = exc


__all__ = [
    "Sim2RealRuntime",
    "Sim2RealRuntimeConfig",
    "RobotState",
    "ObjectState",
]


def __getattr__(name: str):
    if name in {"Sim2RealRuntime", "Sim2RealRuntimeConfig"} and _API_IMPORT_ERROR is not None:
        raise RuntimeError(
            "sim2real.api import 실패: torch가 필요합니다. "
            "runtime API를 쓰지 않고 io/core만 사용할 때는 torch 없이도 import 가능합니다."
        ) from _API_IMPORT_ERROR
    raise AttributeError(f"module 'sim2real' has no attribute {name!r}")
