# test_e0509 TRAINING 메모

이 문서는 `test_e0509` 태스크의 rsl_rl 학습/체크포인트 운영 규칙을 간단히 정리한 문서입니다.
이 문서의 내용은 Isaac Sim / Isaac Lab `5.1` 버전 기준으로 정리했습니다.

## 1. 체크포인트 파일명/저장 간격

rsl_rl 체크포인트 파일명 기본 형식은 아래와 같습니다.

- `model_XXXX.pt` (예: `model_500.pt`, `model_1000.pt`)

저장 간격은 각 RunnerCfg의 `save_interval` 값으로 결정됩니다.

- `GripBottlePPORunnerCfg`: `save_interval = 500`
- `MoveBottlePPORunnerCfg`(Stage1/2/3 기반): `save_interval = 250`

코드 위치:

- `test_e0509/agents/rsl_rl_ppo_cfg.py`

## 2. 로그/체크포인트 기본 경로

rsl_rl 기본 경로 형식:

- `/IsaacLab/logs/rsl_rl/<experiment_name>/<run_name>/model_XXXX.pt`

예시(`rsl_rl`):

- `/IsaacLab/logs/rsl_rl/move_bottle_stage1/2026-03-17_10-00-00_move_bottle_stage1_v5/model_2000.pt`

`<experiment_name>`은 RunnerCfg의 `experiment_name` 값으로 결정됩니다.

## 3. run_name 지정

- `run_name`은 로그 경로의 실행 단위 폴더 이름입니다.
- 미지정 시 보통 timestamp 기반으로 자동 생성됩니다.
- 필요하면 `--run_name`으로 직접 지정할 수 있습니다.

예시:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-E0509-Move-Bottle-Stage1-Direct-v0 \
  --run_name stage1_v7_num2500 \
  --headless
```

## 4. 학습 실행 예시 (rsl_rl)

IsaacLab 루트(`/IsaacLab`)에서 실행:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-E0509-Grip-Bottle-Direct-v0 \
  --headless
```

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-E0509-Move-Bottle-Stage1-Direct-v0 \
  --headless
```

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-E0509-Move-Bottle-Stage2-Direct-v0 \
  --headless
```

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-E0509-Move-Bottle-Stage3-Direct-v0 \
  --headless
```

## 5. 학습 시 자주 쓰는 옵션

- `--num_envs N`
  - 병렬 환경 수를 CLI에서 강제 지정합니다.
  - 이 태스크 코드 기본값은 `64` (`grip_bottle_env.py`의 `scene.num_envs=64`)이며, 옵션으로 덮어쓸 수 있습니다.
  - 예: `--num_envs 2500`, `--num_envs 4096`
- `--max_iterations K`
  - 학습 반복 수를 CLI에서 덮어씁니다.
  - 미지정 시 `rsl_rl_ppo_cfg.py`의 각 RunnerCfg `max_iterations` 값을 사용합니다.
- `--headless`
  - GUI 없이 실행합니다(학습 속도/리소스 측면에서 일반적으로 권장).
  - 이 옵션을 빼면 Isaac Sim GUI 창이 열려서 상태를 보면서 디버깅할 수 있습니다.

예시(Stage1, 대규모 병렬 + 반복 수 지정):

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-E0509-Move-Bottle-Stage1-Direct-v0 \
  --num_envs 2500 \
  --max_iterations 12000 \
  --headless
```

## 6. 재생/평가 실행 예시

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-E0509-Move-Bottle-Stage1-Direct-v0 \
  --checkpoint /IsaacLab/logs/rsl_rl/move_bottle_stage1/<run_name>/model_2000.pt \
  --num_envs 16
```

스테이지 체크포인트 스윕 평가는 아래 스크립트를 사용:

- `test_e0509/eval_stage_checkpoints.sh`

동작 요약:

- `--from ~ --to` 구간을 `--step` 간격으로 순회하며 `model_XXXX.pt`를 자동 탐색합니다.
- 각 체크포인트마다 `play.py --video`를 1회 실행해 영상을 생성합니다.
- 결과는 "체크포인트별 개별 영상"으로 생성되며, 하나의 영상으로 자동 병합되지는 않습니다.
- `--dry_run`을 쓰면 실제 실행 없이 생성될 명령어만 확인할 수 있습니다.

경로 주의:

- 현재 스크립트는 `run_dir/model_XXXX.pt` 구조를 전제로 합니다.
- 즉, `--run_dir`는 체크포인트 파일(`model_XXXX.pt`)이 직접 들어있는 run 디렉터리여야 합니다.

MoveBottle 로그 출력 정책:

- 터미널 로그는 기본적으로 compact 모드입니다(`log_compact = True`).
- 상세 로그 파일 저장이 기본 활성화되어 있습니다(`save_full_log_file = True`).
- 파일명은 기본 `metrics_full.jsonl`이며, 기본적으로 run 폴더 안에 생성됩니다.
- 기본 설정에서 env step마다 1개 JSONL 레코드를 기록합니다(`full_log_every_env_steps = 1`).
- 파일 flush 주기는 기본 50회 기록마다 1번입니다(`full_log_flush_every_writes = 50`).
- 각 레코드에는 full 지표와 compact 스냅샷이 함께 들어가므로, 오프라인 분석/코덱스 분석에 바로 사용하기 좋습니다.

## 7. 팀 운영 팁

- 체크포인트 파일명 기본 형식은 `model_XXXX.pt`입니다.
- 문서와 스크립트 호환을 위해 체크포인트 파일명은 기본 형식을 그대로 유지하는 것을 권장합니다.
- sim2real로 가져갈 때는 해당 모델의 `obs_dim/act_dim`과 관측 순서를 반드시 같이 기록해두는 것이 안전합니다.
