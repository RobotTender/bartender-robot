# test_e0509 사용 메모

이 폴더는 IsaacLab task 폴더 내부에 배치해서 사용합니다.

- 배치 경로:
  - `/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/test_e0509`
- 즉, `direct` 아래에 `test_e0509` 폴더 전체(현재 폴더)를 그대로 넣어야 합니다.

## 관측/액션 정의

### 1) Grip Bottle (`Isaac-E0509-Grip-Bottle-Direct-v0`)

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

### 2) Move Bottle (`Isaac-E0509-Move-Bottle-Direct-v0`, Stage1/2/3 공통 Env)

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
- TCP 높이(테이블 상면 기준), TCP y 오프셋(테이블 중심 y 기준)

관측 생성 수식:

1. `q_norm = 2 * (q - q_low) / (q_high - q_low) - 1`
2. `qd_norm = qd * 0.1`
3. `goal_err = clip((q - q_goal) / (q_high - q_low), -1, 1)` (6차원)
4. `y_parallel_obs = clip(sign * tcp_y_world_z, -1, 1)` (`preferred_tcp_y_world_z_sign` 반영)
5. `ee_speed_obs = clip(||v_ee|| / success_max_ee_speed, 0, 5)`
6. `tcp_height_obs = clip((tcp_height_from_table) / obs_tcp_height_scale_m, -5, 5)`
7. `tcp_y_offset_obs = clip((tcp_y - table_center_y) / obs_tcp_y_offset_scale_m, -5, 5)`
8. `obs = [q_norm(6), qd_norm(6), goal_err(6), y_parallel(1), ee_speed(1), tcp_height(1), tcp_y_offset(1)]`
9. `obs = clip(obs, -5, 5)`

총 차원: `6 + 6 + 6 + 1 + 1 + 1 + 1 = 22`

## 실전 이식 시 체크리스트

- 정책의 `obs_dim`이 Env 관측 차원과 정확히 같은지 확인
- 입력 단위(rad, rad/s, m)가 맞는지 확인
- 클래스 인덱스/one-hot 순서(soju, orange, beer) 유지
- Move 계열은 `goal_err`, `y_parallel`, `tcp_height`, `tcp_y_offset` 항목까지 동일 수식으로 생성해야 함
