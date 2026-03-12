"""Bartender robot action planner.

목적:
- 주문 결과(레시피) + vision1 객체 메타를 바탕으로 로봇 시퀀스를 구성한다.
- 플래너 코드에서는 "데이터 처리"와 "모션 시퀀스"를 분리한다.
- 모션 호출은 DSR 원본 감각과 맞추기 위해 movel(posx(...)), movej(posj(...)) 명시를 따른다.

주의:
- run_robot_action(context) : 기본 진입점 (step 계획 생성)
- run_robot_action(context, api=...) : 외부 API 객체 주입 가능
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


# ---------------------------------------------------------------------------
# 1) 메뉴/레시피 규칙
# ---------------------------------------------------------------------------

MENU_ORDER = {
    "soju": ["soju"],
    "beer": ["beer"],
    "somaek": ["soju", "beer"],
}

MENU_DEFAULT_RECIPE = {
    "soju": {"soju": 50.0},
    "beer": {"beer": 200.0},
    "somaek": {"soju": 60.0, "beer": 140.0},
}

INGREDIENT_CLASS_ALIASES = {
    "soju": ("soju", "소주"),
    "beer": ("beer", "맥주"),
}


# ---------------------------------------------------------------------------
# 2) 모션 파라미터(여기를 수정하며 튜닝)
# ---------------------------------------------------------------------------

# 사용자 요청에 맞춰 고정값은 아래 시퀀스 함수에서 숫자 리터럴로 직접 표기한다.


class PlannerSequenceApi:
    """플래너 호출형 API -> backend step 포맷 변환기.

    모션 메서드는 원본 DSR API 의미를 이름에 반영한다.
    - movel_posx(...): movel(posx(...), vel=..., acc=..., radius=..., ra=...)
    - movej_posj(...): movej(posj(...), v=..., a=...)
    """

    def __init__(self):
        self._steps = []

    @property
    def sequence_steps(self):
        return list(self._steps)

    def _append(self, step: dict):
        self._steps.append(dict(step))

    @staticmethod
    def _is_resolved_target_pose(pose):
        return isinstance(pose, dict) and ("__resolved_target_key__" in pose)

    def set_robot_mode(self, mode: int, label: str = "로봇 오토모드 전환", enabled: bool = True):
        self._append(
            {
                "op": "set_robot_mode",
                "label": str(label),
                "mode": int(mode),
                "enabled": bool(enabled),
            }
        )

    def move_home(self, label: str = "홈 이동", enabled: bool = True):
        self._append(
            {
                "op": "backend_call",
                "label": str(label),
                "method": "send_move_home",
                "args": [],
                "kwargs": {},
                "enabled": bool(enabled),
            }
        )

    def movel_posx(self, pose6, label: str = "카테시안 이동", enabled: bool = True, timeout_sec: float | None = None):
        if self._is_resolved_target_pose(pose6):
            payload = dict(pose6)
            step = {
                "op": "movel_resolved_target",
                "label": str(label),
                "target_key": str(payload.get("__resolved_target_key__", "")),
                "approach_up_mm": float(payload.get("approach_up_mm", 0.0)),
                "enabled": bool(enabled),
            }
            if payload.get("xyzabc") is not None:
                step["xyzabc"] = list(payload.get("xyzabc"))
            if payload.get("offset_xyz_mm") is not None:
                step["offset_xyz_mm"] = list(payload.get("offset_xyz_mm"))
            if payload.get("abc") is not None:
                step["abc"] = list(payload.get("abc"))
            if timeout_sec is not None:
                step["timeout_sec"] = float(timeout_sec)
            self._append(step)
            return

        vals = [float(v) for v in list(pose6)[:6]]
        step = {
            "op": "backend_call",
            "label": str(label),
            "method": "send_move_cartesian",
            "args": vals,
            "kwargs": {},
            "enabled": bool(enabled),
        }
        if timeout_sec is not None:
            step["timeout_sec"] = float(timeout_sec)
        self._append(step)

    def movej_posj(self, joints6, label: str = "조인트 이동", timeout_sec: float | None = None):
        vals = [float(v) for v in list(joints6)[:6]]
        if len(vals) != 6:
            raise ValueError("movej_posj는 6개 조인트 값이 필요합니다.")
        step = {
            "op": "backend_call",
            "label": str(label),
            "method": "send_move_joint",
            "args": vals,
            "kwargs": {},
            "enabled": True,
        }
        if timeout_sec is not None:
            step["timeout_sec"] = float(timeout_sec)
        self._append(step)

    # backward compatibility aliases
    def movel(self, pose6, label: str = "카테시안 이동", enabled: bool = True, timeout_sec: float | None = None):
        self.movel_posx(pose6=pose6, label=label, enabled=enabled, timeout_sec=timeout_sec)

    def movej(self, joints6, label: str = "조인트 이동", timeout_sec: float | None = None):
        self.movej_posj(joints6=joints6, label=label, timeout_sec=timeout_sec)

    def gripper(self, distance_mm: float, label: str = "그리퍼 이동", enabled: bool = True):
        self._append(
            {
                "op": "backend_call",
                "label": str(label),
                "method": "send_gripper_move",
                "args": [float(distance_mm)],
                "kwargs": {},
                "enabled": bool(enabled),
            }
        )

    def resolve_detection_target(
        self,
        *,
        target_key: str,
        ingredient_code: str,
        center_uv,
        depth_m: float,
        label: str = "비전 타겟 계산",
        extra_offset_xyz_mm=None,
        apply_menu_offset: bool = True,
        enabled: bool = True,
    ):
        self._append(
            {
                "op": "resolve_detection_target",
                "label": str(label),
                "target_key": str(target_key),
                "ingredient_code": str(ingredient_code),
                "target_uv": [float(center_uv[0]), float(center_uv[1])],
                "target_depth_m": float(depth_m),
                "apply_menu_offset": bool(apply_menu_offset),
                "extra_offset_xyz_mm": list(extra_offset_xyz_mm or [0.0, 0.0, 0.0]),
                "enabled": bool(enabled),
            }
        )

    def get_resolved_target_pose(self, *, target_key: str, approach_up_mm: float = 0.0, xyzabc=None, offset_xyz_mm=None, abc=None):
        payload = {
            "__resolved_target_key__": str(target_key),
            "approach_up_mm": float(approach_up_mm),
        }
        if xyzabc is not None:
            payload["xyzabc"] = list(xyzabc)
        if offset_xyz_mm is not None:
            payload["offset_xyz_mm"] = list(offset_xyz_mm)
        if abc is not None:
            payload["abc"] = list(abc)
        return payload

    def add_pose_offset_xyz(self, pose_payload, *, dx_mm: float = 0.0, dy_mm: float = 0.0, dz_mm: float = 0.0):
        if not isinstance(pose_payload, dict):
            raise ValueError("pose_payload는 get_resolved_target_pose() 반환 dict여야 합니다.")
        base = pose_payload.get("offset_xyz_mm", [0.0, 0.0, 0.0])
        try:
            bx = float(base[0]) if isinstance(base, (list, tuple)) and len(base) >= 1 else 0.0
            by = float(base[1]) if isinstance(base, (list, tuple)) and len(base) >= 2 else 0.0
            bz = float(base[2]) if isinstance(base, (list, tuple)) and len(base) >= 3 else 0.0
        except Exception:
            bx, by, bz = 0.0, 0.0, 0.0
        pose_payload["offset_xyz_mm"] = [
            float(bx) + float(dx_mm),
            float(by) + float(dy_mm),
            float(bz) + float(dz_mm),
        ]
        return pose_payload

    def movel_resolved_target(
        self,
        *,
        target_key: str,
        label: str = "비전 타겟 이동",
        approach_up_mm: float = 0.0,
        xyzabc=None,
        offset_xyz_mm=None,
        abc=None,
        enabled: bool = True,
        timeout_sec: float | None = None,
    ):
        step = {
            "op": "movel_resolved_target",
            "label": str(label),
            "target_key": str(target_key),
            "approach_up_mm": float(approach_up_mm),
            "enabled": bool(enabled),
        }
        if xyzabc is not None:
            step["xyzabc"] = list(xyzabc)
        if offset_xyz_mm is not None:
            step["offset_xyz_mm"] = list(offset_xyz_mm)
        if abc is not None:
            step["abc"] = list(abc)
        if timeout_sec is not None:
            step["timeout_sec"] = float(timeout_sec)
        self._append(step)

    def move_to_detection(
        self,
        *,
        ingredient_code: str,
        center_uv,
        depth_m: float,
        label: str,
        extra_offset_xyz_mm=None,
        approach_up_mm: float = 40.0,
        apply_menu_offset: bool = True,
        enabled: bool = True,
    ):
        # backward compatibility helper:
        # 기존 move_to_detection 호출은 "계산 + 이동" 2단계로 풀어서 기록한다.
        key = f"legacy_{str(label)}_{str(ingredient_code)}"
        self.resolve_detection_target(
            target_key=key,
            ingredient_code=ingredient_code,
            center_uv=center_uv,
            depth_m=depth_m,
            label=f"{label} (타겟계산)",
            extra_offset_xyz_mm=extra_offset_xyz_mm,
            apply_menu_offset=apply_menu_offset,
            enabled=enabled,
        )
        pose = self.get_resolved_target_pose(target_key=key, approach_up_mm=approach_up_mm)
        self.movel_posx(pose, label=label, enabled=enabled)

    def wait_volume_target(
        self,
        target_volume_ml: float,
        label: str,
        timeout_sec: float = 20.0,
        poll_sec: float = 0.1,
        enabled: bool = True,
    ):
        self._append(
            {
                "op": "wait_volume_target",
                "label": str(label),
                "target_volume_ml": float(target_volume_ml),
                "timeout_sec": float(timeout_sec),
                "poll_sec": float(poll_sec),
                "enabled": bool(enabled),
            }
        )


def _safe_float(value: Any):
    try:
        return float(value)
    except Exception:
        return None


def _norm_code(value: Any):
    return str(value or "").strip().lower()


# ---------------------------------------------------------------------------
# 3) 데이터 준비(주문/비전)
# ---------------------------------------------------------------------------


def _extract_order_result(context: dict):
    ctx = dict(context or {})
    order_result = dict(ctx.get("order_result", {}) or {})
    status = str(order_result.get("status", "")).strip().lower()
    if status != "success":
        return None, {
            "ok": False,
            "status": "order_not_success",
            "message": "주문 결과 status가 success가 아닙니다.",
        }
    return order_result, None


def _extract_vision1_detections(context: dict):
    ctx = dict(context or {})
    vision1_meta = dict(ctx.get("vision1_meta", {}) or {})
    detections = vision1_meta.get("detections", [])
    if not isinstance(detections, list):
        detections = []
    return detections


def _pick_detection_for_ingredient(detections, ingredient_code: str):
    if not isinstance(detections, list):
        return None
    aliases = tuple(_norm_code(v) for v in INGREDIENT_CLASS_ALIASES.get(_norm_code(ingredient_code), (ingredient_code,)))
    best = None
    best_score = -1.0
    for det in detections:
        if not isinstance(det, dict):
            continue
        class_name = _norm_code(det.get("class_name"))
        if aliases and class_name not in aliases:
            continue

        center_uv = det.get("center_uv")
        if not isinstance(center_uv, (list, tuple)) or len(center_uv) < 2:
            bbox = det.get("bbox_xyxy")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                try:
                    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
                    center_uv = [(x1 + x2) * 0.5, (y1 + y2) * 0.5]
                except Exception:
                    center_uv = None
        if not isinstance(center_uv, (list, tuple)) or len(center_uv) < 2:
            continue

        depth_m = _safe_float(det.get("depth_m"))
        if depth_m is None or depth_m <= 0.0:
            continue
        conf = _safe_float(det.get("confidence"))
        score = float(conf if conf is not None else 0.0)
        if score >= best_score:
            best = {
                "class_name": class_name,
                "confidence": score,
                "center_uv": [float(center_uv[0]), float(center_uv[1])],
                "depth_m": float(depth_m),
            }
            best_score = score
    return best


def _resolve_recipe_in_menu_order(order_result: dict):
    selected_menu = _norm_code(order_result.get("selected_menu"))
    raw_recipe = order_result.get("recipe", {})
    recipe = dict(raw_recipe) if isinstance(raw_recipe, dict) else {}

    ordered_codes = list(MENU_ORDER.get(selected_menu, []))
    if not ordered_codes and recipe:
        ordered_codes = [_norm_code(k) for k in recipe.keys()]
    if not ordered_codes:
        return [], selected_menu, "주문에서 메뉴/레시피를 확인할 수 없습니다."

    resolved = []
    seen = set()
    for code in ordered_codes:
        code = _norm_code(code)
        if not code:
            continue
        amount = _safe_float(recipe.get(code))
        if amount is None:
            amount = _safe_float(MENU_DEFAULT_RECIPE.get(selected_menu, {}).get(code))
        if amount is None or amount <= 0.0:
            return [], selected_menu, f"레시피 누락: 메뉴={selected_menu}, 재료={code}"
        resolved.append((code, float(amount)))
        seen.add(code)

    for key, value in recipe.items():
        code = _norm_code(key)
        if (not code) or code in seen:
            continue
        amount = _safe_float(value)
        if amount is None or amount <= 0.0:
            continue
        resolved.append((code, float(amount)))
        seen.add(code)
    return resolved, selected_menu, ""


def _resolve_targets_from_recipe(recipe_items, detections):
    picked_targets = []
    missing_ingredients = []
    resolved_targets = []

    cumulative_target_ml = 0.0
    for ingredient_code, amount_ml in recipe_items:
        det = _pick_detection_for_ingredient(detections, ingredient_code)
        if det is None:
            missing_ingredients.append(ingredient_code)
            continue

        cumulative_target_ml += float(amount_ml)
        row = {
            "ingredient_code": str(ingredient_code),
            "amount_ml": float(amount_ml),
            "target_volume_ml": float(cumulative_target_ml),
            "detection": dict(det),
        }
        resolved_targets.append(row)
        picked_targets.append(dict(row))

    return picked_targets, missing_ingredients, resolved_targets


# ---------------------------------------------------------------------------
# 4) 모션 시퀀스 구성(movel(posx), movej(posj) 의미를 직접 사용)
# ---------------------------------------------------------------------------


def _read_current_posj(api: PlannerSequenceApi):
    """현재 조인트값을 읽어 반환한다.

    반환: (ok: bool, msg: str, posj6: list|None)
    """
    backend = getattr(api, "backend", None)
    if backend is None or (not hasattr(backend, "get_current_positions")):
        return False, "현재 조인트 조회 실패: backend.get_current_positions 미지원", None

    data = backend.get_current_positions()
    if not isinstance(data, (list, tuple)) or len(data) < 2:
        return False, "현재 조인트 조회 실패: 위치 캐시 없음", None

    posj = data[0]
    if not isinstance(posj, (list, tuple)) or len(posj) < 6:
        return False, "현재 조인트 조회 실패: current_posj 길이 부족", None

    try:
        out = [float(v) for v in list(posj)[:6]]
    except Exception as exc:
        return False, f"현재 조인트 조회 실패: 파싱 오류({exc})", None

    return True, "ok", out


def _read_current_posx(api: PlannerSequenceApi):
    """현재 TCP posx값을 읽어 반환한다.

    우선순위:
    1) backend.get_current_posx_live()
    2) backend.get_current_positions()의 current_posx 캐시

    반환: (ok: bool, msg: str, posx6: list|None)
    """
    backend = getattr(api, "backend", None)
    if backend is None:
        return False, "현재 posx 조회 실패: backend 미지원", None

    if hasattr(backend, "get_current_posx_live"):
        posx_live, _sol_live, err_live = backend.get_current_posx_live()
        if err_live is None and isinstance(posx_live, (list, tuple)) and len(posx_live) >= 6:
            try:
                out = [float(v) for v in list(posx_live)[:6]]
            except Exception as exc:
                return False, f"현재 posx 조회 실패: 실시간 파싱 오류({exc})", None
            return True, "ok(live)", out

    if hasattr(backend, "get_current_positions"):
        data = backend.get_current_positions()
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            posx = data[1]
            if isinstance(posx, (list, tuple)) and len(posx) >= 6:
                try:
                    out = [float(v) for v in list(posx)[:6]]
                except Exception as exc:
                    return False, f"현재 posx 조회 실패: 캐시 파싱 오류({exc})", None
                return True, "ok(cache)", out

    return False, "현재 posx 조회 실패: 실시간/캐시 모두 없음", None


def _read_current_pose_to_buffer(api: PlannerSequenceApi, runtime_buf: dict | None = None, key: str = "current_pose"):
    """현재 로봇 pose를 읽어 버퍼에 저장한다.

    저장 형태:
    runtime_buf[key] = {
        "posx": [x, y, z, a, b, c],
        "posj": [j1, j2, j3, j4, j5, j6]
    }

    반환:
    (ok: bool, msg: str, pose: dict|None)
    """
    ok_x, msg_x, cur_posx = _read_current_posx(api)
    if (not ok_x) or (not isinstance(cur_posx, (list, tuple))) or len(cur_posx) < 6:
        return False, str(msg_x), None

    ok_j, msg_j, cur_posj = _read_current_posj(api)
    if (not ok_j) or (not isinstance(cur_posj, (list, tuple))) or len(cur_posj) < 6:
        return False, str(msg_j), None

    pose = {
        "posx": [float(v) for v in list(cur_posx)[:6]],
        "posj": [float(v) for v in list(cur_posj)[:6]],
    }

    if isinstance(runtime_buf, dict):
        runtime_buf[str(key)] = dict(pose)
    return True, "ok", pose


def _append_start_sequence(api: PlannerSequenceApi):
    api.set_robot_mode(mode=1, label="로봇 오토모드 전환")
    # 원본 의미: movej(posj(HOME_POSJ), v=..., a=...)
    api.move_home(label="초기 홈 이동")
    api.gripper(90.0, label="그리퍼 초기 오픈")


def _append_ingredient_sequence(api: PlannerSequenceApi, row: dict, seq_index: int):
  
    ingredient_code = str(row["ingredient_code"])
    center_uv = row["detection"]["center_uv"]
    depth_m = float(row["detection"]["depth_m"])
    target_volume_ml = float(row["target_volume_ml"])

    # 객체 기준 로봇좌표에서 각 단계별 XYZ 오프셋(mm)을 명시적으로 더해 사용한다.
    pick_base_offset = [0.0, 0.0, 0.0]
    pick_lift_offset = [0.0, 0.0, 110.0]
    place_top_offset = [0.0, 0.0, 110.0]
    place_offset = [0.0, 0.0, 0.0]
    retreat_offset = [0.0, 0.0, 110.0]
    pour_ready_pose = [430.0, -110.0, 360.0, 180.0, 0.0, 180.0]
    pour_pose_by_ingredient = {
        "soju": [430.0, -95.0, 270.0, 180.0, 0.0, 180.0],
        "beer": [430.0, -130.0, 270.0, 180.0, 0.0, 180.0],
    }
    pour_pose = list(pour_pose_by_ingredient.get(ingredient_code, pour_ready_pose))
    target_key = f"{seq_index}_{ingredient_code}_target"

    # A) 비전1 대상 병 접근 -> 파지 -> 리프트
    # 역할: 비전 검출값(center_uv, depth_m)을 로봇 좌표계 목표로 변환해 `target_key`로 1회 저장한다.
    api.resolve_detection_target(
        target_key=target_key,
        ingredient_code=ingredient_code,
        center_uv=center_uv,
        depth_m=depth_m,
        label=f"[{ingredient_code}] 병 기준 타겟 계산",
        extra_offset_xyz_mm=pick_base_offset,
        apply_menu_offset=True,
    )
    # 비전 목표위치 획득.
    base_xyz = getattr(api, "_resolved_targets", {}).get(target_key)
    if base_xyz is None:
        raise RuntimeError(f"target_key={target_key} 미해결")
    x, y, z = [float(v) for v in base_xyz[:3]]

    # 현재위치 (posx/posj)를 버퍼로 읽는다.    # 필요할 때마다 같은 패턴으로 재호출하면 된다.
    ok_pose, msg_pose, pose_now = _read_current_pose_to_buffer(api)
    if not ok_pose:
        # posx/posj는 모두 필수 값이다. 하나라도 없으면 시퀀스를 중단한다.
        raise RuntimeError(f"[{ingredient_code}] 시퀀스 중지: {msg_pose}")
    cur_posx = list(pose_now["posx"])
    cur_posj = list(pose_now["posj"])

    a = float(cur_posx[3])
    b = float(cur_posx[4])
    c = float(cur_posx[5])

    # 픽 접근 위치정의(절대좌표 XYZABC)
    pick_approach_pose = [
        x,
        y-200,
        z,
        a,
        b,
        c,
    ]
    # 비전 목표위치 이동
    api.movel_posx(pick_approach_pose, label=f"[{ingredient_code}] 병 접근(절대좌표)")

    # 병 파지 위치(절대좌표 XYZABC)
    pick_pose = [
        x,
        y,
        z,
        a,
        b,
        c,
    ]
    api.movel_posx(pick_pose, label=f"[{ingredient_code}] 병 파지 위치 이동")
    # 역할: 그리퍼를 닫아 병을 파지한다.
    api.gripper(80.0, label=f"[{ingredient_code}] 병 파지")

    # 역할: 기준 타겟 + 리프트 오프셋(절대좌표 XYZABC)으로 이동한다.
    lift_pose = [
        x ,
        y ,
        z + 70.0,
        a,
        b,
        c,
    ]
    # 역할: 병을 든 상태로 리프트 포즈까지 이동한다.
    api.movel_posx(lift_pose, label=f"[{ingredient_code}] 병 리프트 위치 이동")
    

    # B) 따르기 위치 이동 -> 비전2 목표용량까지 대기
    # 원본 의미: movel(posx([430,-110,360,180,0,180]), vel=velx, acc=accx, radius=0.0, ra=DR_MV_RA_DUPLICATE)
    # 역할: 공통 따르기 준비 포즈로 이동한다.

       # 현재위치 (posx/posj)를 버퍼로 읽는다.    # 필요할 때마다 같은 패턴으로 재호출하면 된다.
    ok_pose, msg_pose, pose_now = _read_current_pose_to_buffer(api)
    if not ok_pose:
        # posx/posj는 모두 필수 값이다. 하나라도 없으면 시퀀스를 중단한다.
        raise RuntimeError(f"[{ingredient_code}] 시퀀스 중지: {msg_pose}")
    cur_posx = list(pose_now["posx"])
    cur_posj = list(pose_now["posj"])

    j_target = list(cur_posj)
    j_target[0] = 0.0

    api.movej_posj(j_target, label=f"[{ingredient_code}] 따르기 준비 위치 이동")
    '''
    api.movel_posx(pour_ready_pose, label=f"[{ingredient_code}] 따르기 준비 위치 이동")
    

    
    # 역할: 재료별 실제 따르기 포즈로 이동한다.
    api.movel_posx(pour_pose, label=f"[{ingredient_code}] 따르기 위치 이동")
    # 역할: vision2 용량 인식값이 누적 목표(target_volume_ml)에 도달할 때까지 대기한다.
    api.wait_volume_target(
        target_volume_ml=target_volume_ml,
        label=f"[{ingredient_code}] 용량 목표 대기",
        timeout_sec=28.0,
        poll_sec=0.1,
    )
    # 역할: 따르기 완료 후 안전한 준비 포즈로 복귀한다.
    api.movel_posx(pour_ready_pose, label=f"[{ingredient_code}] 따르기 준비 위치 복귀")

    # C) 병 원위치 복귀 -> 놓기 -> 이격
    # 역할: 기준 타겟 + 상부 복귀 오프셋(절대좌표 XYZABC)으로 이동한다.
    place_top_pose = [
        x + float(place_top_offset[0]),
        y + float(place_top_offset[1]),
        z + float(place_top_offset[2]),
        a,
        b,
        c,
    ]
    # 역할: 병 원위치 상부까지 이동한다.
    api.movel_posx(place_top_pose, label=f"[{ingredient_code}] 병 원위치 상부 복귀")

    # 역할: 기준 타겟 + 배치 오프셋(절대좌표 XYZABC)으로 이동한다.
    place_pose = [
        x + float(place_offset[0]),
        y + float(place_offset[1]),
        z + float(place_offset[2]),
        a,
        b,
        c,
    ]
    # 역할: 병을 원위치 배치 포즈로 이동한다.
    api.movel_posx(place_pose, label=f"[{ingredient_code}] 병 원위치 배치")
    # 역할: 그리퍼를 열어 병을 놓는다.
    api.gripper(90.0, label=f"[{ingredient_code}] 병 놓기")

    # 역할: 기준 타겟 + 이격 오프셋(절대좌표 XYZABC)으로 이동한다.
    retreat_pose = [
        x + float(retreat_offset[0]),
        y + float(retreat_offset[1]),
        z + float(retreat_offset[2]),
        a,
        b,
        c,
    ]
    # 역할: 병과 주변 물체로부터 안전하게 떨어지는 위치로 이동한다.
    api.movel_posx(retreat_pose, label=f"[{ingredient_code}] 병에서 이격")
    '''

def _append_finish_sequence(api: PlannerSequenceApi):
    # 원본 의미: movel(posx([520,-20,300,180,0,180]), ...)
    api.movel_posx([520.0, -20.0, 300.0, 180.0, 0.0, 180.0], label="완성잔 전달 위치 이동")
    # 원본 의미: movej(posj(HOME_POSJ), ...)
    api.move_home(label="시퀀스 종료 홈 복귀")


def run_robot_action(context: dict, api: PlannerSequenceApi | None = None):
    order_result, err_out = _extract_order_result(context)
    if err_out is not None:
        return dict(err_out)

    recipe_items, selected_menu, recipe_err = _resolve_recipe_in_menu_order(order_result)
    if recipe_err:
        return {"ok": False, "status": "recipe_invalid", "message": recipe_err}

    detections = _extract_vision1_detections(context)
    sequence_api = api if api is not None else PlannerSequenceApi()

    picked_targets, missing_ingredients, resolved_targets = _resolve_targets_from_recipe(recipe_items, detections)
    if missing_ingredients:
        return {
            "ok": False,
            "status": "vision_target_missing",
            "message": f"vision1에서 레시피 재료를 찾지 못했습니다: {', '.join(missing_ingredients)}",
            "missing_ingredients": list(missing_ingredients),
            "plan": {
                "selected_menu": selected_menu,
                "recipe_items": [{"code": c, "amount_ml": float(a)} for c, a in recipe_items],
                "picked_targets": picked_targets,
                "sequence_steps": sequence_api.sequence_steps,
            },
        }

    _append_start_sequence(sequence_api)
    for idx, row in enumerate(resolved_targets, start=1):
        _append_ingredient_sequence(sequence_api, row, seq_index=idx)
    _append_finish_sequence(sequence_api)

    return {
        "ok": True,
        "status": "planned_sequence",
        "message": "레시피/비전 기반 전체 로봇 시퀀스를 생성했습니다.",
        "plan": {
            "selected_menu": selected_menu,
            "recipe_items": [{"code": c, "amount_ml": float(a)} for c, a in recipe_items],
            "picked_targets": picked_targets,
            "sequence_steps": sequence_api.sequence_steps,
        },
    }


def _load_input(args):
    if args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as fp:
            return json.load(fp)
    raw = sys.stdin.read()
    if not str(raw).strip():
        return {}
    return json.loads(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bartender robot action planner")
    parser.add_argument("--json-file", default="", help="input JSON file path (optional)")
    args = parser.parse_args(argv)

    try:
        context = _load_input(args)
    except Exception as exc:
        out = {"ok": False, "status": "invalid_input", "message": f"입력 JSON 오류: {exc}"}
        print(json.dumps(out, ensure_ascii=False))
        return 1

    try:
        out = run_robot_action(context)
    except Exception as exc:
        out = {"ok": False, "status": "exception", "message": f"planner 예외: {exc}"}
        print(json.dumps(out, ensure_ascii=False))
        return 1

    print(json.dumps(out, ensure_ascii=False))
    return 0 if bool(out.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
