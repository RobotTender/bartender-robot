# Bartender Robot Sim2Real 가이드

이 문서는 **강화학습 정책(`.pt`)을 실제 로봇에 이식하는 sim2real 실행 방법**을 설명합니다.
카메라/비전 처리 자체는 이 저장소의 책임 범위가 아니며, 외부 노드/클래스에서 계산된 상태를 입력으로 받는 구조를 기준으로 합니다.
초보자도 바로 따라할 수 있게 순서대로 정리했습니다.

문서 역할은 아래처럼 나눠서 보는 것을 권장합니다.

- 루트 `README.md`(현재 문서): sim2real 실행법, ROS2/실기 연결법, 체크포인트 교체법, 입출력 연결 규약
- `test_e0509/README.md`: 학습 태스크 자체의 관측/액션 의미, stage 차이, 정규화 기준, 환경 맥락
- `test_e0509/TRAINING.md`: IsaacLab/rsl_rl 학습 실행, 체크포인트 저장 경로, 재생/평가 스크립트 사용법
- `test_e0509/USD/README.md`: 실제로 남겨둔 USD 자산들의 역할, 워크스페이스/로봇/오브젝트 구성 메모

즉, **이 문서는 “어떻게 연결해서 쓰는지” 중심**, `test_e0509` 문서는 **“그 모델이 무엇을 배우고 어떤 의미로 입출력하는지” 중심**으로 읽으면 덜 헷갈립니다.

---

## 1. 이 프로젝트가 하는 일

`run_sim2real_e0509.py`를 실행하면 내부에서 다음을 반복합니다.

1. 현재 로봇/오브젝트 상태를 입력으로 받음
2. 강화학습 정책이 행동(action) 계산
3. 행동을 로봇 조인트 목표값 + 그리퍼 명령으로 변환
4. 변환된 명령을 출력(토픽 발행 또는 실기 API 호출)

핵심 파일:

- `run_sim2real_e0509.py`: 실행 엔트리
- `sim2real/runner.py`: 실행 루프 + CLI 옵션
- `sim2real/core.py`: 정책 로딩, 관측 생성, 액션 출력 가공
- `sim2real/io.py`: I/O 백엔드(ROS2/실기/더미)
- `sim2real/api.py`: 통합 파이썬 코드에 직접 붙이는 API

참고:

- 체크포인트의 관측/액션 의미가 궁금하면 `test_e0509/README.md`를 먼저 확인하세요.
- 이 문서는 그 의미를 새로 정의하기보다, **이미 정의된 관측/액션을 실제 입력/출력 경로에 맞게 연결하는 방법**을 설명합니다.

---

## 2. 가장 빠른 테스트 (더미 모드)

실장비나 ROS2 없이 동작 확인:

```bash
python run_sim2real_e0509.py --mode dummy --checkpoint test_model.pt --print-policy-info
```

이 저장소의 `test_model.pt`는 빠른 연동 점검용 샘플 체크포인트입니다.

- 정책 로딩/관측 생성/액션 변환 파이프라인이 정상 실행되는지 확인하는 용도
- 실제 작업 성능을 보장하는 실전 모델은 아님
- 실전에서는 원하는 학습 파일로 `--checkpoint`만 바꿔 실행
- sim2real 실행기는 이름과 상관없이 원하는 `.pt` 체크포인트를 읽을 수 있음
- `rsl_rl`로 학습하면 체크포인트는 보통 `model_XXXX.pt` 형식으로 저장되며, 내부 문서/스크립트도 그 형식을 기준으로 설명함

`--mode virtual` 또는 `--mode sim`도 동일하게 동작합니다.

정상 동작하면 step 로그가 출력됩니다.

제어주기 설정:

- `--hz`로 루프 주기를 지정합니다. (기본 60)
- 현재 코드는 `--hz`를 바꾸면 다음이 함께 바뀝니다.
  - 루프 sleep 주기
  - 액션 증분 적분 시간(`control_dt`)
  - `mode=real`에서 `--servo-time-s` 미지정 시 기본 `servoj t = 1/--hz`
  - `mode=real`에서 `--speedj-time-s` 미지정 시 기본 `speedj t = 1/--hz`

시작 전 1회 주기 점검(preflight):

- 기본으로 ON (`--preflight-cycle-check`)
- 한 주기 예산을 넘기면 초과 ms를 경고 로그로 출력합니다.
- 예산 이내일 때도 보고 싶으면 `--preflight-cycle-check-verbose`

시작 전 안전 모션 점검(preflight motion):

- 메인 루프(`run_sim2real_e0509.py`)와 분리되어 있습니다.
- 별도 실행기: `sim2real_safety_test.py`
- `hold` 모드: 현재 자세 유지 명령만 송신
- `joint_delta` 모드: 단일 조인트에 소각도 증분(기본 joint6, +10deg) 테스트 후 복귀
- 기본 동작: `joint_delta`는 목표각에서 약 5초 유지 후 자동 복귀
- 옵션:
  - `--safety-motion-mode {hold,joint_delta}`
  - `--safety-motion-joint-number 6`
  - `--safety-motion-delta-deg 10`
  - `--safety-motion-command-sec 5.0`
  - `--safety-motion-return-to-base` / `--no-safety-motion-return-to-base`

예시(안전 점검만 빠르게):

```bash
python sim2real_safety_test.py \
  --mode real \
  --checkpoint /path/to/model_XXXX.pt \
  --safety-motion-mode hold
```

예시(joint6 +10deg 테스트 후 복귀):

```bash
python sim2real_safety_test.py \
  --mode real \
  --checkpoint /path/to/model_XXXX.pt \
  --safety-motion-mode joint_delta \
  --safety-motion-joint-number 6 \
  --safety-motion-delta-deg 10
```

---

## 3. ROS2로 사용하는 방법

## 3-0. 로봇 IP 입력이 안 보이는 이유 (중요)

이 프로젝트는 모드에 따라 연결 방식이 다릅니다.

- `--mode ros2`
- 이 모드는 로봇에 직접 TCP 연결하지 않습니다.
- ROS2 토픽/서비스 브릿지 역할만 합니다.
- 즉, 로봇 컨트롤러 IP 연결은 별도 Doosan ROS2/드라이버 쪽에서 이미 끝나 있어야 합니다.

- `--mode real`
- 이 모드는 `DR_init/DSR_ROBOT2`로 직접 제어합니다.
- 기본은 SDK 환경의 기본 연결 설정을 사용합니다.
- 필요하면 이제 아래 옵션으로 host/port를 직접 넘길 수 있습니다.
- `--doosan-host <controller_ip>`
- `--doosan-port <controller_port>`

- `--gripper-ip`는 암(arm) IP가 아니라 ModbusTCP 그리퍼용 IP입니다.
- `--gripper-protocol flange_serial_fc`를 쓰면 그리퍼는 플랜지 시리얼 경로라서 `--gripper-ip`를 사용하지 않습니다.

## 3-1. 준비

ROS2 노드들이 아래 토픽을 제공해야 합니다.

중요:

- 아래는 "토픽 인터페이스" 목록입니다. 관측/출력 차원(`obs_dim`, `act_dim`)은 체크포인트 모델마다 달라질 수 있습니다.
- 모델 차원 호환 방법은 [5장](#5-입력출력입출력-연결-규약), [6장](#6-모델pt-교체-방법)을 따릅니다.

입력(정책이 구독):

- `/joint_states` (`sensor_msgs/JointState`)
- `/ee_pose` (`geometry_msgs/PoseStamped`)
- `/object_pose` (`geometry_msgs/PoseStamped`)
- `/object_class` (`std_msgs/Int32`)
- `/object_extra` (`std_msgs/Float32MultiArray`, `--obs-custom-extra-dim > 0`일 때 사용)
- `/gripper/state` (`std_msgs/Float32`, 0~1, `--obs-include-gripper-state`일 때 사용)

출력(정책이 발행):

- `/arm/joint_position_cmd` (`sensor_msgs/JointState`, rad)
- `/gripper/command` (`std_msgs/Float32`, 0~1, 모델 `act_dim >= 7` + gripper output 활성일 때 사용)

주의:

- `/object_pose`, `/object_class`는 외부 비전 노드/클래스가 publish하면 됩니다.
- 즉, 이 sim2real 코드가 RealSense raw 토픽을 직접 처리할 필요는 없습니다.

## 3-2. 실행

```bash
python run_sim2real_e0509.py \
  --mode ros2 \
  --checkpoint /path/to/model_XXXX.pt \
  --print-policy-info
```

토픽 이름이 다르면 옵션으로 바꿉니다.

예:

```bash
python run_sim2real_e0509.py \
  --mode ros2 \
  --checkpoint /path/to/model_XXXX.pt \
  --ros2-joint-state-topic /my_robot/joint_states \
  --ros2-joint-cmd-topic /my_robot/joint_cmd
```

## 3-3. Doosan DrlStart 기반 그리퍼 사용(기존 프로젝트 방식)

`/dsr01/drl/drl_start` 서비스로 DRL 코드를 보내는 모드입니다.

```bash
python run_sim2real_e0509.py \
  --mode ros2 \
  --checkpoint /path/to/model_XXXX.pt \
  --ros2-gripper-command-mode drl_service \
  --ros2-drl-namespace dsr01
```

주요 옵션:

- `--ros2-drl-slave-id` (기본 1)
- `--ros2-drl-baudrate` (기본 57600)
- `--ros2-drl-torque-enable-addr` (기본 256)
- `--ros2-drl-goal-current-addr` (기본 275)
- `--ros2-drl-goal-position-addr` (기본 282)
- `--ros2-drl-init-goal-current` (기본 400)

그리퍼를 외부 노드가 제어할 때:

- ROS2 모드: `--no-ros2-enable-gripper-output`
- 실기 모드: `--no-gripper-enabled`

---

## 4. 통합 파이썬 코드에 직접 넣는 방법 (ROS2 없이 가능)

이미 네가 가진 메인 제어 루프 안에서 정책만 호출하고 싶을 때 사용합니다.

```python
from sim2real import Sim2RealRuntime, Sim2RealRuntimeConfig, RobotState, ObjectState
import numpy as np

runtime = Sim2RealRuntime(
    Sim2RealRuntimeConfig(
        checkpoint="/path/to/model_XXXX.pt",
        device="cpu",  # 또는 "cuda:0"
        control_hz=60.0,
    )
)
print(runtime.summary())

# 초기 상태 1회
robot_state = RobotState(
    joint_pos_rad=np.zeros(6, dtype=np.float32),
    joint_vel_rad_s=np.zeros(6, dtype=np.float32),
    gripper_close_ratio=0.0,
    ee_pos_m=np.zeros(3, dtype=np.float32),
    ee_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
)
object_state = ObjectState(
    position_m=np.array([0.0, 0.65, 1.30], dtype=np.float32),
    up_vector_w=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    class_index=0,
    custom_extra=np.array([0.11, 0.02], dtype=np.float32),  # 필요 없으면 생략 가능
)
runtime.reset(robot_state, object_state)

# 주기 루프
while True:
    # 1) 센서/비전에서 최신 상태 채우기
    # robot_state = ...
    # object_state = ...

    # 2) 정책 1스텝
    out = runtime.step(robot_state, object_state)

    # 3) 결과 사용
    joint_targets_rad = out["joint_targets"]
    joint_velocity_cmd_rad_s = out["joint_velocity_cmd_rad_s"]
    gripper_target = out["gripper_target"]

    # 4) 로봇 명령으로 변환해서 송신
    # ex) servoj(np.rad2deg(joint_targets_rad).tolist(), ...)
    # ex) speedj(np.rad2deg(joint_velocity_cmd_rad_s).tolist(), ...)
    # ex) gripper stroke = int(round(gripper_target * 750))
```

---

## 5. 입력/출력(입출력) 연결 규약

이 장은 sim2real 실행기가 **정책 입력/출력을 어떤 형식으로 주고받는지** 설명합니다.

중요:

- 여기서는 "입출력 형식과 연결 규약"을 설명합니다.
- 특정 학습 태스크의 정확한 의미(예: GripBottle의 액션이 step 증분인지, MoveBottle의 액션이 속도 비율인지)는 `test_e0509/README.md`에서 설명합니다.
- 즉, 루트 README는 **형식/연결**, 태스크 README는 **의미/물리 해석** 담당입니다.

## 5-1. 정책 입력(Observation)

`ObservationBuilder`가 다음 순서로 만듭니다.

1. 조인트 위치(정규화)
2. 조인트 속도(스케일)
3. TCP -> object 그립 포인트 벡터 (dx, dy, dz)
4. lift amount
5. gripper state (0~1)
6. object class one-hot (3)
7. object up z
8. custom extra vector (N, 선택)

기본식:

- `obs_dim = 2 * obs_joint_dim + extra_dim`
- `extra_dim = to_object(3) + lift(1) + gripper_state(1) + object_class_dim + object_up_z(1) + custom_extra_dim`
- 각 항목은 CLI/API 옵션으로 on/off 가능

예:

- 21차원 모델이면 자동으로 `obs_joint_dim=6`
- 29차원 모델이면 자동으로 `obs_joint_dim=10`

관측 구성 토글 옵션:

- `--obs-include-to-object` / `--no-obs-include-to-object`
- `--obs-include-lift` / `--no-obs-include-lift`
- `--obs-include-gripper-state` / `--no-obs-include-gripper-state`
- `--obs-include-object-class` / `--no-obs-include-object-class`
- `--obs-object-class-dim`
- `--obs-include-object-up-z` / `--no-obs-include-object-up-z`
- `--obs-custom-extra-dim` (관측 끝에 붙일 custom scalar 개수)

custom extra 값 입력 경로:

- `--mode ros2`: `/object_extra` 토픽 (`std_msgs/Float32MultiArray`)
- `--mode real|dummy`: `--object-fixed-custom-extra` 또는 `object-source=json` 파일의 `custom_extra` 필드

예시(20관측, 6액션, gripper state 제외):

```bash
python run_sim2real_e0509.py \
  --mode dummy \
  --checkpoint /path/to/model_XXXX.pt \
  --obs-joint-dim 6 \
  --no-obs-include-gripper-state
```

예시(22관측 모델, custom extra 2개 사용):

```bash
python run_sim2real_e0509.py \
  --mode dummy \
  --checkpoint model_1500.pt \
  --obs-joint-dim 6 \
  --obs-custom-extra-dim 2 \
  --object-fixed-custom-extra 0.11,0.02 \
  --print-policy-info
```

## 5-2. 정책 출력(Action)

- `action[0:6]`: 6축 조인트 제어 입력
- `action[6]`: 그리퍼 close ratio 제어 입력 (모델 `act_dim >= 7`일 때만)
- 모델이 `act_dim = 6`이면 그리퍼 액션은 없고, 그리퍼는 외부 통합 코드에서 별도 제어하면 됩니다.

`ActionController`가 이를 실제 명령으로 바꿉니다.

- 조인트: 증분 적분 후 절대 목표각(`joint_targets`, rad)
- 조인트 속도 명령: 적분 직전 증분을 시간으로 나눈 값(`joint_velocity_cmd_rad_s`, rad/s)
- 그리퍼: 0~1 비율(`gripper_target`)
- 조인트 증분 스케일은 `control_dt`를 사용하며, CLI 실행 시 `control_dt = 1/--hz`로 자동 동기화됩니다.

실무 해석 포인트:

- 정책 raw 출력값 자체는 단위가 없는 연속 실수값이며, 보통 각 원소가 `[-1, 1]` 범위입니다.
- 따라서 출력값을 보정/클램프/스무딩할 때는 raw 값만 보기보다, 이 값이 실제로 어떤 물리량으로 해석된 뒤 로봇에 들어가는지 같이 봐야 합니다.
- 이 sim2real 실행기 기본 동작은 `servoj`용 절대 목표각(`joint_targets`, rad)을 만드는 방식입니다.
- `speedj`를 쓰고 싶다면 raw action을 바로 넣지 말고, 먼저 조인트 목표 속도(`deg/s` 또는 `rad/s`)로 변환한 뒤 보내야 합니다.
- 체크포인트의 정확한 액션 의미(예: step 증분인지, 속도 비율인지)는 학습 태스크 문서를 따라야 합니다. `test_e0509` 계열 모델 기준 설명은 `test_e0509/README.md`를 참고하세요.

Doosan 명령으로 옮길 때:

- `servoj`를 쓸 때: `runtime.step(...)` 결과의 `joint_targets`를 `rad -> deg`로 바꿔서 넣습니다.
- `speedj`를 쓸 때: `runtime.step(...)` 결과의 `joint_velocity_cmd_rad_s`를 `rad/s -> deg/s`로 바꿔서 넣거나, 별도 속도 명령 변환층을 둡니다.
- 급격한 튐 방지를 위해서는 `speedj`에 넣기 전에 속도 스무딩, 가속도 제한, 이상치 차단을 추가하는 편이 안전합니다.

---

## 6. 모델(.pt) 교체 방법

가장 기본:

```bash
python run_sim2real_e0509.py --mode ros2 --checkpoint /path/to/new_model.pt --print-policy-info
```

로더는 기본적으로 `actor.*`를 읽어서 구조를 자동 추론합니다.

필요 시 수동 강제:

- `--policy-obs-dim`
- `--policy-act-dim`
- `--policy-hidden-sizes` (예: `256,128,64`)
- `--policy-activation` (`elu|relu|tanh|silu`)

관측 조인트 차원 강제:

- `--obs-joint-dim`
- 생략하면 `policy.obs_dim`에서 자동 추론

추가 관측 슬롯 사용:

- `--obs-custom-extra-dim N`
- `--object-fixed-custom-extra v1,v2,...` (fixed/dummy fallback)
- `object-source=json`이면 JSON에 `"custom_extra": [v1, v2, ...]` 추가

특정 모델 호환 점검(로봇 안 움직임, 추천):

```bash
python run_sim2real_e0509.py \
  --mode dummy \
  --checkpoint /path/to/model_30000.pt \
  --max-steps 1 \
  --print-policy-info
```

이 명령은 다음을 확인합니다.

- 체크포인트 로드 가능 여부
- 추론된 `obs_dim / act_dim / hidden` 구조
- 현재 sim2real 관측 구성과 최소 실행 호환 여부

주의:

- 여기서 통과하는 것은 **형식 호환성** 확인입니다.
- 실제 관측 의미까지 맞는지는 학습 태스크 문서(예: `test_e0509/README.md`)와 함께 확인해야 합니다.

---

## 7. 자주 막히는 포인트

1. 토픽은 오는데 로봇이 안 움직임
- `/arm/joint_position_cmd`를 실제 드라이버가 subscribe하는지 확인
- 단위가 rad인지 확인

2. 그리퍼가 안 움직임
- topic 모드인지 drl_service 모드인지 먼저 확인
- drl_service 모드면 `/dsr01/drl/drl_start` 서비스가 살아있는지 확인
- 외부 그리퍼 제어를 쓰는 구성이라면 `--no-ros2-enable-gripper-output` 또는 `--no-gripper-enabled`를 확인

3. 모델 로드 에러
- 체크포인트가 `actor.*` 키를 포함하는지 확인
- `--print-policy-info`로 감지된 구조 확인

4. 21차원 모델인데 동작이 이상함
- `--print-policy-info`로 obs_dim 확인
- 필요하면 `--obs-joint-dim 6` 명시

---

## 8. 실전 권장 시작 명령

ROS2 + DrlStart 그리퍼(두산 환경):

```bash
python run_sim2real_e0509.py \
  --mode ros2 \
  --checkpoint /path/to/model_XXXX.pt \
  --hz 60 \
  --print-policy-info \
  --ros2-gripper-command-mode drl_service \
  --ros2-drl-namespace dsr01
```

ROS2 + 외부 인지/외부 그리퍼 제어(정책 이식 최소 구성):

```bash
python run_sim2real_e0509.py \
  --mode ros2 \
  --checkpoint /path/to/model_XXXX.pt \
  --hz 60 \
  --print-policy-info \
  --no-ros2-enable-gripper-output
```

실기 직접 제어(`mode=real`, host 지정 예시, 기본 `servoj`):

```bash
python run_sim2real_e0509.py \
  --mode real \
  --checkpoint /path/to/model_XXXX.pt \
  --hz 60 \
  --doosan-robot-id dsr01 \
  --doosan-model e0509 \
  --doosan-host 192.168.137.100 \
  --doosan-port 12345
```

실기 직접 제어(`mode=real`, `speedj` 사용 예시):

```bash
python run_sim2real_e0509.py \
  --mode real \
  --checkpoint /path/to/model_XXXX.pt \
  --hz 60 \
  --joint-command-mode speedj \
  --speedj-vel-deg-s 30 \
  --speedj-acc-deg-s2 100 \
  --doosan-robot-id dsr01 \
  --doosan-model e0509 \
  --doosan-host 192.168.137.100 \
  --doosan-port 12345
```

참고:

- `mode=real`에서 `--servo-time-s`를 생략하면 `1/--hz`가 자동 적용됩니다.
- `mode=real`에서 `--speedj-time-s`를 생략하면 `1/--hz`가 자동 적용됩니다.
- `speedj`는 기본적으로 실측 조인트 기준 재앵커링(`--speedj-reanchor-to-measured`)이 켜져 있습니다.
- 내부 누적 속도(`joint_velocity_cmd_rad_s`)를 그대로 쓰고 싶으면 `--no-speedj-reanchor-to-measured`를 사용합니다.
- 필요하면 `--servo-time-s`, `--speedj-time-s`를 각각 수동 고정할 수 있습니다.

통합 파이썬 코드 임베드:

- `from sim2real import Sim2RealRuntime`
- `runtime.step(...)` 결과의 `joint_targets`는 `servoj`용, `joint_velocity_cmd_rad_s`는 `speedj`용으로 사용할 수 있습니다.
