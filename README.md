# 바텐더 로봇 프로젝트

두산 로봇, RealSense 2대, 개발자 UI, 사용자 Web UI, 음성주문(STT->LLM->레시피), 비전 메타 파이프라인을 통합 실행하는 프로젝트입니다.

## 핵심 구성

- 로봇 제어 백엔드
  - 로봇 상태/모드/좌표 수집, 이동 명령, 그리퍼 초기화/제어를 담당합니다.
- 개발자 UI
  - 로봇/비전/음성주문 상태 확인, 테스트 실행, 로그 디버깅을 담당합니다.
- 사용자 Web UI
  - 최종 사용자 주문 진입 화면을 제공합니다.
  - 현재는 `USER_FRONTEND_ENABLED`(하위호환: `VOICE_ORDER_WEBUI_ENABLED`) 플래그로 진입 허용/차단을 제어합니다.
- 음성주문 통합
  - 백엔드 요청 -> 음성처리 워커 -> Gemini STT -> 주문 분류(LLM 보조) -> 레시피 결과를 반환합니다.
- 바텐더 시퀀스 매니저(백엔드)
  - `BartenderSequenceManager`가 오토/메뉴얼 공용 시퀀스를 실행합니다.
  - 안전 판정/정지 권한은 백엔드만 보유하고, 프론트엔드는 시작/중지 요청과 상태 표시만 수행합니다.
- 비전 메타 파이프라인
  - 비전1 객체 메타, 비전2 용량 메타를 발행하고 UI에서 오버레이합니다.
- 캘리브레이션
  - Eye-to-Hand 행렬 계산/저장/적용으로 비전 좌표를 로봇 좌표로 변환합니다.

## 실행

### 1) 환경 로드

```bash
source /opt/ros/humble/setup.bash
source <ros2_ws>/install/setup.bash
```

### 2) 기본 실행

```bash
cd <project-root>
python3 run_bartender.py
```

기본 launch 인자:

```bash
run_robot:=true
robot_mode:=real
robot_host:=110.120.1.68
robot_model:=e0509
run_sensors:=true
run_frontend:=true
run_user_frontend:=true
webui_host:=0.0.0.0
webui_port:=8000
sequence_api_host:=127.0.0.1
sequence_api_port:=8765
```

### 3) 자주 쓰는 실행 예시

실제 로봇:

```bash
python3 run_bartender.py robot_mode:=real robot_host:=110.120.1.68 robot_model:=e0509
```

가상 로봇:

```bash
python3 run_bartender.py robot_mode:=virtual robot_model:=e0509
```

로봇 bringup 제외:

```bash
python3 run_bartender.py run_robot:=false
```

사용자 Web UI 제외:

```bash
python3 run_bartender.py run_user_frontend:=false
```

## 주문 기능 환경변수(.env)

`.env` 파일은 프로젝트 루트(`<project-root>/.env`)에 두고 사용합니다.

- `GOOGLE_API_KEY` 또는 `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `VOICE_ORDER_MODEL`
- `VOICE_ORDER_STT_MODEL`
- `VOICE_ORDER_STT_RETRIES`
- `USER_FRONTEND_ENABLED` (또는 `VOICE_ORDER_WEBUI_ENABLED`)
- `USER_FRONTEND_HOST` (또는 `VOICE_ORDER_WEBUI_HOST`)
- `USER_FRONTEND_PORT` (또는 `VOICE_ORDER_WEBUI_PORT`)
- `BARTENDER_SEQUENCE_API_HOST`
- `BARTENDER_SEQUENCE_API_PORT`
- `VOICE_ORDER_CYCLE_INTERVAL_MS`

## 프로그램 구조(요약)

```text
run_bartender.py
  -> launch/system_launch.py
     -> doosan bringup
     -> realsense launch
     -> developer_frontend.py
     -> user_frontend.py (최종 사용자 Web UI)

developer_frontend.py
  -> backend(task_backend_node.py).run_voice_order_runtime(...)
     -> subprocess: voice_order_worker.py
        -> gemini_stt_pipeline.py
        -> voice_order_pipeline.py
        -> recipe result
  -> backend(task_backend_node.py).BartenderSequenceManager
     -> 시퀀스 상태/로그/안전정지 권한 단일화

백엔드 시퀀스 제어 API:
- `GET /api/sequence/state`
- `POST /api/sequence/start`
- `POST /api/sequence/stop`
```

상세 구조/상태:

- 구조와 폴더 역할: [docs/PROJECT_LAYOUT.md](docs/PROJECT_LAYOUT.md)
- 런타임/프로세스 구조: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 주문기능 머지 1차 상태: [docs/ORDER_FEATURE_MERGE_PHASE1.md](docs/ORDER_FEATURE_MERGE_PHASE1.md)
- 배포 절차: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- 벤더 패치: [docs/VENDOR_PATCHES.md](docs/VENDOR_PATCHES.md)

## 현재 주의사항

- Doosan 동작에는 vendor patch 적용이 필요합니다.
- Web UI는 기본값이 비활성화(`USER_FRONTEND_ENABLED=0`, 하위호환 `VOICE_ORDER_WEBUI_ENABLED=0`)라서 접근해도 주문은 차단될 수 있습니다.
- 사용자 Web UI의 `/api/control/start`는 현재 백엔드 로봇 실행과 직접 연결되지 않은 상태입니다.
- 음성주문 결과는 현재 백엔드 메모리 스냅샷과 UI 이벤트 기반이며, ROS 토픽/서비스 표준 인터페이스는 아직 확정 전입니다.
