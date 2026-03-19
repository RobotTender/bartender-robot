# test_e0509 사용 메모

이 폴더는 IsaacLab task 폴더 내부에 배치해서 사용합니다.
이 문서의 내용은 Isaac Sim / Isaac Lab `5.1` 버전 기준으로 정리했습니다.

- 배치 경로:
  - `/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/test_e0509`
- 즉, `direct` 아래에 `test_e0509` 폴더 전체(현재 폴더)를 그대로 넣어야 합니다.
- 학습/체크포인트 운영 메모는 [TRAINING.md](./TRAINING.md)를 참고하세요.

## 환경 시나리오 요약 (Grip Bottle 기준)

이전 브랜치 README에서 쓰던 "환경 맥락"을 유지해서 정리하면 아래와 같습니다.

- 로봇 에셋:
  - `USD/e0509/e0509_model.usd`: Doosan E0509 암 + RH-P12-RN(A) 그리퍼 결합 모델(암 `joint_1~joint_6`, 그리퍼 `rh_r1_joint` 기준 제어)
- 워크스페이스 에셋:
  - `USD/table_hole.usd`: 로봇이 얹히는 작업대(워크스페이스) 모델
  - `USD/tables_3.usd`: 실제 환경을 반영한 선반/테이블 배경 모델(오브젝트 위치 맥락/충돌 공간 반영)
    - 모델 내부 고정 오프셋: `(-55, 1120, 728) mm` (`(-0.055, 1.120, 0.728) m`)
- 오브젝트 에셋:
  - `USD/soju.usd`, `USD/orange.usd`, `USD/beer.usd`: 랜덤 active 병 후보 3종

배치/해석 포인트:

- 코드 변수명은 `table_*`지만, 프로젝트 해석은 "선반 상면/선반 중심" 기준으로 봐도 됩니다.
- 로봇 기준 설치 높이: `z = 0.73 m` (마운트 상면 기준)
- 상면 기준값(현재 설정): `(x, y, z) = (0.00, 0.67, 1.30) m`
  - `x = 0.00` (`table_top_center_xy[0]`)
  - `y = 0.67` (`table_top_center_xy[1]`, `tcp_y_offset_obs` 기준)
  - `z = 1.30` (`table_top_z`, `tcp_height_obs` 기준)
- `table_hole.usd`, `tables_3.usd`는 둘 다 scene에 로드되며, 로봇은 같은 env 안에서 별도 articulation으로 배치됩니다.

오브젝트 스폰 규칙(Grip):

- 매 reset마다 병 3개 중 1개가 active로 랜덤 선택됩니다.
- active 병(soju/orange/beer)만 선반 상면 스폰 영역 안에서 랜덤 `(x, y, yaw)`로 배치됩니다.
- 나머지 2개는 파킹 좌표(작업영역 밖)로 이동됩니다.
- one-hot도 active 병에 맞춰 같이 갱신됩니다.

## 환경 시나리오 요약 (Move Bottle 기준)

MoveBottle은 GripBottleEnv를 상속하므로, 기본 scene 에셋(로봇/선반 USD 경로)은 동일하게 사용합니다.

핵심 차이:

- `use_virtual_bottle = True`가 기본값입니다.
- 물리 병 3개는 reset 시 파킹 위치로 이동(+가시성 off)하고, 학습 대상 병은 TCP 기준 가상 오브젝트로 계산됩니다.
- reset 시작 자세는 랜덤 IK가 아니라 고정 시작 자세(`move_start_joint_pos`)를 사용합니다.
- 시작 시 그리퍼는 닫힘 상태(`start_gripper_close_ratio=1.0`, `lock_gripper_closed=True`) 기준입니다.

목표 자세(단계별):

- Stage1 (`MoveBottleStage1EnvCfg`):
  - 목표 조인트: `(90, -45, 90, 0, 45, -90)` deg
  - 목적: 고정 시작자세에서 standby 근방으로 안정 이동
- Stage2/Stage3 (`MoveBottleStage2EnvCfg`, `MoveBottleStage3EnvCfg`):
  - 목표 조인트: `(45, 0, 135, 90, -90, -135)` deg
  - 목적: 운반 중 자세/충돌/기울기 제약을 포함한 목표 자세 도달

Move 관측에서 추가된 항목의 의미:

- `tcp_height_obs`: 선반 상면(`z=1.30`) 대비 TCP 높이
- `tcp_y_offset_obs`: 선반 중심선(`y=0.67`) 대비 TCP y 편차
- 둘 다 매 step 현재 EE/TCP 상태에서 실시간 계산됩니다.

## 관측/액션 정의

### 1. Grip Bottle (`Isaac-E0509-Grip-Bottle-Direct-v0`)

- `action_space = 6`
- `observation_space = 20`
- 코드 위치:
  - `grip_bottle_env.py`의 `GripBottleEnvCfg`
  - `grip_bottle_env.py`의 `_get_observations`

관측 벡터 순서(총 20):

1. `0:6`   arm joint pos (normalized)
2. `6:12`  arm joint vel (scaled)
3. `12`    gripper_state (0~1)
4. `13:16` to_object `(dx, dy, dz)`
5. `16`    object_up_z
6. `17:20` active_bottle_one_hot `(soju, orange, beer)`

### 2. Move Bottle (`Isaac-E0509-Move-Bottle-Direct-v0`, Stage1/2/3 공통 Env)

- `action_space = 6`
- `observation_space = 22`
- 코드 위치:
  - `move_bottle_env.py`의 `MoveBottleEnvCfg`
  - `move_bottle_env.py`의 `_get_observations`

관측 벡터 순서(총 22):

1. `0:6`    arm joint pos (normalized)
2. `6:12`   arm joint vel (scaled)
3. `12:18`  goal_err_obs (goal joint error, normalized)
4. `18`     y_parallel_obs
5. `19`     ee_speed_obs
6. `20`     tcp_height_obs
7. `21`     tcp_y_offset_obs

## 핵심 정리

- 두 환경 모두 액션 차원은 동일하게 `6`입니다.
- 관측 차원은 동일하지 않습니다.
  - Grip: `20`
  - Move: `22`
- Move Stage1/Stage2/Stage3는 보상/게이트 설정이 다르고, 관측/액션 인터페이스는 같은 `MoveBottleEnv`를 사용합니다.

## 조인트 제한각 표 (`q_low`, `q_high`)

`q_low`, `q_high`는 고정 `±180°`가 아니라 아래 순서로 결정됩니다.

1. 로봇 USD의 `soft_joint_pos_limits`를 기본값으로 가져옴
2. 수동 제한이 켜져 있으면(`use_*_abs_limit=True`) 해당 조인트를 추가로 클램프
3. 최종 `q_low`, `q_high`는 "기본 soft limit"과 "수동 제한"의 교집합

추가 수동 제한(기본 설정):

| 조인트 | GripBottle | MoveBottle(Stage1/2/3) |
|---|---:|---:|
| Joint 1 | ±115° (±2.0071 rad) | ±115° (±2.0071 rad) |
| Joint 2 | soft limit 사용 | ±95° (±1.6581 rad) |
| Joint 3 | soft limit 사용 | soft limit 사용 |
| Joint 4 | soft limit 사용 | ±120° (±2.0944 rad) |
| Joint 5 | soft limit 사용 | soft limit 사용 |
| Joint 6 | soft limit 사용 | soft limit 사용 |

참고:

- E0509 설정에서 soft limit이 사실상 ±180°로 잡힌 조인트가 많아 "대부분 ±180°처럼 보일" 수는 있습니다.
- 하지만 MoveBottle은 Joint 2/4가 추가 제한되므로, 외부 런타임에서 동일 정규화를 쓰려면 이 제한도 같이 맞춰야 합니다.

## 입력값 넣는 방법 (중요)

이 폴더의 Env는 IsaacLab 시뮬레이터 내부 상태로 관측을 자동 생성합니다.  
즉, 학습 중에는 사용자가 관측 벡터를 직접 넣지 않습니다.

다만 sim2real/외부 런타임에서 같은 정책을 쓰려면, 아래와 같은 **원시 상태(raw state)**를 같은 단위로 넣고
동일 수식으로 관측을 만들어야 합니다.

### 공통 단위 규칙

- 조인트 위치: `rad`
- 조인트 속도: `rad/s`
- 위치 좌표: `m`
- 방향 벡터/쿼터니언: 정규화된 값 사용 권장
- 최종 관측은 Env에서 `[-5, 5]`로 클립

### Grip(20) 입력 구성 상세

원시 입력(필수):

- arm 6축 현재 조인트 위치 `q[6]` (rad)
- arm 6축 현재 조인트 속도 `qd[6]` (rad/s)
- 그리퍼 상태(조인트 기반 close ratio, 0~1)
- EE body 위치 `ee_pos[3]` (m)
- 오브젝트 위치 `obj_pos[3]` (m)
- 오브젝트 up의 z성분 `obj_up_z` (또는 up 벡터)
- 오브젝트 클래스 one-hot(soju/orange/beer)
  - `[1, 0, 0]` = soju
  - `[0, 1, 0]` = orange
  - `[0, 0, 1]` = beer
  - 클래스 인덱스 기준도 동일(`0=soju, 1=orange, 2=beer`)

관측 생성 수식:

1. `q_norm = 2 * (q - q_low) / (q_high - q_low) - 1`
2. `qd_norm = qd * 0.1` (`dof_velocity_scale`)
3. `to_object = obj_pos - ee_pos`
4. `obs = [q_norm(6), qd_norm(6), gripper_state(1), to_object(3), obj_up_z(1), class_one_hot(3)]`
5. `obs = clip(obs, -5, 5)`

총 차원: `6 + 6 + 1 + 3 + 1 + 3 = 20`

### Move(22) 입력 구성 상세

원시 입력(필수):

- arm 6축 현재 조인트 위치 `q[6]` (rad)
- arm 6축 현재 조인트 속도 `qd[6]` (rad/s)
- 목표 조인트 `q_goal[6]` (Env 설정값, rad)
- TCP Y축의 world Z 정렬값 계산에 필요한 EE 자세
- EE 선속도 크기 `||v_ee||` (m/s)
- TCP 높이(`tcp_height_obs`): 선반 상면 기준 높이
  - 코드 변수명은 `table_top_z`지만, 현재 프로젝트에서는 선반 상면으로 사용
  - 현재 기준값: 상면 `z = 1.30 m`
  - 매 step 실시간 계산: `tcp_height_from_shelf = tcp_z - shelf_top_z`
- TCP y 오프셋(`tcp_y_offset_obs`): 선반 중심선 대비 y 편차
  - 현재 기준값: 선반 중심 `y = 0.67 m`
  - 매 step 실시간 계산: `tcp_y_offset = tcp_y - shelf_center_y`
  - 의미: TCP가 선반 중심선에서 y축으로 얼마나 벗어났는지(부호 포함)

관측 생성 수식:

1. `q_norm = 2 * (q - q_low) / (q_high - q_low) - 1`
2. `qd_norm = qd * 0.1`
3. `goal_err = clip((q - q_goal) / (q_high - q_low), -1, 1)` (6차원)
4. `y_parallel_obs = clip(sign * tcp_y_world_z, -1, 1)` (`preferred_tcp_y_world_z_sign` 반영)
5. `ee_speed_obs = clip(||v_ee|| / success_max_ee_speed, 0, 5)`
6. `tcp_height_obs = clip((tcp_height_from_shelf) / obs_tcp_height_scale_m, -5, 5)`
7. `tcp_y_offset_obs = clip((tcp_y - shelf_center_y) / obs_tcp_y_offset_scale_m, -5, 5)`
8. `obs = [q_norm(6), qd_norm(6), goal_err(6), y_parallel(1), ee_speed(1), tcp_height(1), tcp_y_offset(1)]`
9. `obs = clip(obs, -5, 5)`

총 차원: `6 + 6 + 6 + 1 + 1 + 1 + 1 = 22`

## 실전 이식 시 체크리스트

- 정책의 `obs_dim`이 Env 관측 차원과 정확히 같은지 확인
- 입력 단위(rad, rad/s, m)가 맞는지 확인
- 클래스 인덱스/one-hot 순서(soju, orange, beer) 유지
- Move 계열은 `goal_err`, `y_parallel`, `tcp_height`, `tcp_y_offset` 항목까지 동일 수식으로 생성해야 함
