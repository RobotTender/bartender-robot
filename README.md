# bartender-robot

바텐더 로봇 런타임 저장소입니다. 이 저장소는 `Doosan bringup + RealSense + frontend/backend/vision` 앱 코드를 한곳에 정리한 독립 repo입니다.

## 빠른 실행

### 환경 로드

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

### 기본 실행

```bash
cd ~/ros2_ws/src/bartender-robot
python3 run_bartender.py
```

기본값:

```bash
run_robot:=true
robot_mode:=real
robot_host:=110.120.1.68
robot_model:=e0509
run_sensors:=true
run_frontend:=true
```

### 자주 쓰는 실행 예시

실제 로봇:

```bash
python3 run_bartender.py
python3 run_bartender.py robot_mode:=real robot_host:=110.120.1.68 robot_model:=e0509
```

가상 로봇:

```bash
python3 run_bartender.py robot_mode:=virtual robot_model:=e0509
python3 run_bartender.py robot_mode:=virtual robot_model:=e0509 robot_gz:=false
```

로봇 bringup 없이:

```bash
python3 run_bartender.py run_robot:=false
```

센서 없이:

```bash
python3 run_bartender.py run_sensors:=false
```

프론트엔드 없이:

```bash
python3 run_bartender.py run_frontend:=false
```

## 주요 폴더

```text
bartender-robot/
  assets/      UI 파일, 모델 파일
  config/      parameter.csv, calibration 결과
  docs/        구조/배포/Git 문서
  launch/      시스템/센서/비전 launch
  scripts/     배포 보조 스크립트
  src/         frontend/backend/vision 코드
  vendor/      Doosan vendor patch
  run_bartender.py
```

상세 설명:

- [docs/PROJECT_LAYOUT.md](/home/fc/ros2_ws/src/robot_bartender/docs/PROJECT_LAYOUT.md)
- [docs/ARCHITECTURE.md](/home/fc/ros2_ws/src/robot_bartender/docs/ARCHITECTURE.md)

## 실행 entrypoint

- [run_bartender.py](/home/fc/ros2_ws/src/robot_bartender/run_bartender.py)
  - 사용자용 단일 실행 파일
- [launch/system_launch.py](/home/fc/ros2_ws/src/robot_bartender/launch/system_launch.py)
  - Doosan bringup, RealSense, frontend 실행
- [launch/realsense_launch.py](/home/fc/ros2_ws/src/robot_bartender/launch/realsense_launch.py)
  - RealSense 2대 실행
- [launch/object_detection_launch.py](/home/fc/ros2_ws/src/robot_bartender/launch/object_detection_launch.py)
  - 객체 인식 프로세스 실행
- [launch/calibration_launch.py](/home/fc/ros2_ws/src/robot_bartender/launch/calibration_launch.py)
  - 캘리브레이션 프로세스 실행

## 실행/배포/Git 문서

- 구조 설명: [docs/PROJECT_LAYOUT.md](/home/fc/ros2_ws/src/robot_bartender/docs/PROJECT_LAYOUT.md)
- 런타임 구조: [docs/ARCHITECTURE.md](/home/fc/ros2_ws/src/robot_bartender/docs/ARCHITECTURE.md)
- 배포 절차: [docs/DEPLOYMENT.md](/home/fc/ros2_ws/src/robot_bartender/docs/DEPLOYMENT.md)
- Doosan patch: [docs/VENDOR_PATCHES.md](/home/fc/ros2_ws/src/robot_bartender/docs/VENDOR_PATCHES.md)
- Git 업로드 절차: [docs/GIT_WORKFLOW.md](/home/fc/ros2_ws/src/robot_bartender/docs/GIT_WORKFLOW.md)
- 협업 규칙: [CONTRIBUTING.md](/home/fc/ros2_ws/src/robot_bartender/CONTRIBUTING.md)

## 현재 주의사항

- Doosan 원본은 vendor patch 적용이 필요합니다.
- `config/parameter.csv`의 `vision1_serial`, `vision2_serial`이 실제 카메라와 맞아야 합니다.
- 객체 인식 실행 파일 [drink_detection.py](/home/fc/ros2_ws/src/robot_bartender/src/vision/drink_detection.py), [glass_fill_level.py](/home/fc/ros2_ws/src/robot_bartender/src/vision/glass_fill_level.py)는 현재 placeholder입니다.
- 모델 파일은 [assets/models/best.pt](/home/fc/ros2_ws/src/robot_bartender/assets/models/best.pt)에 있습니다.
