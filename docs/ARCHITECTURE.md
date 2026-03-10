# Runtime Structure

기준일: 2026-03-11

## 1) 프로세스 구조

```text
run_bartender.py
  -> launch/system_launch.py
     -> (선택) doosan bringup
     -> (선택) realsense_launch.py
     -> (선택) src/frontend/developer_frontend.py
     -> (선택) src/frontend/user_frontend.py
                -> src/order_integration/voice_order_webui.py
```

`system_launch.py` 주요 플래그:

- `run_robot`
- `run_sensors`
- `run_frontend`
- `run_user_frontend`
- `webui_host`
- `webui_port`

구형 인자 호환:

- `run_web`, `run_webui` -> `run_user_frontend`로 매핑

## 2) 역할 분리

### 개발자 UI (`src/frontend/developer_frontend.py`)

- 로봇 상태, 비전 상태, 음성주문 디버그 패널 표시
- 백엔드 API 직접 호출로 음성주문 테스트 실행
- UI 타이머/워커로 업데이트 주기 표시

### 사용자 Web UI (`src/order_integration/voice_order_webui.py`)

- 최종 사용자 주문 진입용 HTTP 서버
- STT/LLM 기반 주문 처리 endpoint 제공
- `VOICE_ORDER_WEBUI_ENABLED`가 꺼져 있으면 주문 진입 차단

### 백엔드 (`src/backend/task_backend_node.py`)

- 로봇 상태 구독 및 로봇 명령 처리
- 음성주문 요청을 워커 subprocess로 위임
- 마지막 음성주문 payload 스냅샷 저장/제공

### 음성처리 워커 (`src/order_integration/voice_order_test_worker.py`)

- 입력 요청 수신
- 필요 시 마이크 캡처 + Gemini STT
- LLM 보조 분류/레시피 도출
- stage/result/done 이벤트를 JSON lines로 출력

### 비전 처리 (`src/vision/*.py`)

- 비전1: 객체 인식 메타(`/vision1/object/meta`)
- 비전2: 용량 인식 메타(`/vision2/volume/meta`)
- 캘리브레이션: Eye-to-Hand 행렬 계산/저장

## 3) 음성주문 데이터 흐름

### A. 개발자 UI 경로

```text
Developer UI 버튼
  -> backend.run_voice_order_runtime(...)
     -> subprocess(voice_order_test_worker.py)
        -> (옵션) microphone capture
        -> gemini_stt_pipeline.transcribe_audio_bytes()
        -> voice_order_pipeline.classify_voice_order()
     -> payload(events/result/ok) 반환
  -> UI 패널 로그/결과 반영
```

특징:

- 현재 ROS 토픽/서비스를 통하지 않고 프로세스 호출+메모리 스냅샷 방식
- 로봇 모션 트리거는 음성 워커에서 수행하지 않음

### B. 사용자 Web UI 경로

```text
Browser audio upload
  -> /stt/transcribe/
     -> gemini_stt_pipeline
     -> voice_order_runtime
     -> JSON response(status/menu/recipe/tts_text)
```

`/api/control/start`는 현재 진행 이벤트 표시에 가깝고, 백엔드 로봇 실행과 직접 연동되어 있지 않습니다.

## 4) 상태/주기 업데이트

개발자 UI 기본 주기:

- 로그 flush: `100ms`
- 로봇 상태 갱신: `250ms`
- 비전1/2 상태 갱신: `300ms`
- 음성주문 업데이트 워커: `VOICE_ORDER_CYCLE_INTERVAL_MS` (기본 `100ms`)

사용자 Web UI:

- 서버 loop poll interval: `0.2s`
- 브라우저 오디오 청크 업로드 간격: 프런트 스크립트 기준 `250ms`

## 5) 종료 시퀀스

- `developer_frontend` 종료 시 launch 전체 셧다운 이벤트 발생
- 종료 과정에서:
  - UI 타이머/스레드 정지
  - 음성 워커 중지
  - 비전 compose 워커 정지
  - `system_launch.py`가 Gazebo 잔여 프로세스 정리 시도

## 6) 현재 미완성/확정 전 항목

- 음성주문 결과의 ROS 표준 인터페이스(토픽/서비스) 미확정
- 사용자 Web UI의 `/api/control/start`와 백엔드 로봇 실행 파이프라인 미연동
- Web UI TTS는 현재 톤(wav) 응답 기반으로 임시 구현
- UI 레이아웃 일부는 `.ui` 정적 배치가 아니라 런타임 코드 배치 우선
