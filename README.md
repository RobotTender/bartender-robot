# RobotTender: 자율 바텐더 로봇 프로젝트

Doosan Robotics E0509 로봇팔, Robotis RH-P12-RN 그리퍼, RealSense 깊이 카메라를 활용한 자율 바텐더 시스템입니다.
음성 주문을 받아 병을 집고, 원하는 양만큼 따르고, 제자리에 돌려놓는 전 과정을 자동화합니다.

---

## TL;DR

- **음성 → LLM → 로봇 제조**까지 end-to-end 파이프라인 구현 (Google Gemini STT + LangGraph + ROS 2)
- **비전 기반 실시간 부피 추정**: YOLOv8 Segmentation + 비율 기반 룩업 테이블 + EMA 평활화로 정량 제어
- **스냅 복구(Snap Recovery)**: 목표량 도달 시 자동 인터럽트 → 녹화된 역궤적으로 고속 복귀, 과주 방지
- **다회 발화 주문 처리**: LangGraph 상태 머신으로 애매한 발화·메뉴 추천·비율 결정까지 대화형으로 처리
- **한국어 발음 보정**: "소주 vs 주스" 혼동을 키워드 점수 기반 disambiguation으로 해결

---

## 배경 및 해결 과제

기존 바텐더 로봇 시스템의 한계:

- **정량 제어 불안정**: 고정된 따르기 시간에 의존 → 병 종류·기울기·탄산 유무에 따라 오차 발생
- **환경 의존성**: 컵·병 위치가 사전에 고정돼야 동작 → 실제 사용 환경에서 재현성 낮음
- **인터페이스 제약**: 터치/버튼 기반 주문 → 자연스러운 주문 경험 불가, 운영 인력 필요

이 프로젝트의 접근:

- **비전 기반 실시간 피드백**: 카메라로 액체량을 측정하며 목표 부피 도달 시 자동 중단 → 시간 기반 제어 탈피
- **위치 독립적 측정**: Auto-Tare(자동 영점 조정)와 비율 기반 부피 계산으로 컵 위치·크기 무관하게 동작
- **완전 자동화 주문 파이프라인**: 음성 인식 → 메뉴 결정 → 레시피 생성 → ROS 토픽 발행까지 사람 개입 없이 처리

---

## 팀 구성
- 오재환: [Github](https://github.com/jaehwan-AI)
- 안준성: [Github](https://github.com/JunsungAhn)
- 손주영: [Github](https://github.com/sonjuyeong-00)
- 서미지: [Github](https://github.com/prograsshopper)
- 손영석: [Github](https://github.com/YOUNGSUKSON)
- 김기영: [Github](https://github.com/friday043)
- 김상전: [Github](https://github.com/ca3545)

---

## 목차

1. [시스템 구성 요약](#1-시스템-구성-요약)
2. [전체 실행 흐름](#2-전체-실행-흐름)
3. [프로젝트 구조](#3-프로젝트-구조)
4. [LLM (주문 엔진)](#4-llm-주문-엔진)
5. [Detection (비전 시스템)](#5-detection-비전-시스템)
6. [Robot (로봇 제어 시스템)](#6-robot-로봇-제어-시스템)
7. [하드웨어 및 소프트웨어 요구사항](#7-하드웨어-및-소프트웨어-요구사항)
8. [설치 및 환경 설정](#8-설치-및-환경-설정)
9. [실행 명령어](#9-실행-명령어)
10. [향후 개선 사항](#10-향후-개선-사항)

---

## 1. 시스템 구성 요약

프로젝트는 세 개의 독립적인 컴포넌트로 구성됩니다:

| 컴포넌트 | 역할 | 주요 기술 |
| :--- | :--- | :--- |
| **LLM (주문 엔진)** | 음성 주문 인식 → 메뉴 결정 → 로봇 명령 발행 | Google Gemini STT, OpenAI GPT-4o, LangGraph, Django |
| **Detection (비전)** | 병 위치 탐지 + 액체 부피 측정 | YOLOv8, RealSense D400, OpenCV |
| **Robot (로봇 제어)** | 픽업/따르기/복귀 모션 실행 | ROS 2 Jazzy, Doosan DSR, RH-P12-RN Gripper |

---

## 2. 전체 실행 흐름

### 자동 모드 전체 흐름

```
[사용자] 브라우저에서 http://127.0.0.1:8000/order/ 접속
  → 마이크 녹음 → POST /stt_transcribe/
  → STT (Gemini) → 텍스트 변환
  → LangGraph 처리
      → 직접 주문이면: menu_detail → recipe 확정
      → 추천 필요 시: TTS 응답 후 다음 발화 대기
  → recipe 확정 → /bartender/order_detail 토픽 발행

[Manager Node] 주문 수신
  → ingredient_queue 구성 (병 이름, ml)
  → 루프: 재료가 남아있는 동안
      1. [Pick Node] 카메라 1으로 병 탐지 → 집기 → last_pose 저장
      2. [Pour Node] CHEERS 자세로 이동 → 따르기 시작
             ↕ (병렬)
          [Camera 2] prepare_pouring → 컵 잠금
                   → 액체 감지 → 목표 부피 도달 → 스냅 트리거
      3. [Pour Node] 스냅 수신 → MoveStop → 스냅 복구 (고속 역추적)
      4. [Place Node] last_pose로 병 반납
  → 모든 재료 완료 → order_status = "completed"

[웹 서버] /robot_status/ 폴링으로 완료 감지 → 브라우저에 결과 표시
```

---

## 3. 프로젝트 구조

```
bartender-robot/
├── detection/
│   ├── realsense_cam1.py        # 카메라 1: 병 위치 탐지 (YOLOv8 + 깊이)
│   ├── realsense_cam2.py        # 카메라 2: 액체 부피 측정 (YOLOv8 Seg + EMA)
│   └── weights/
│       ├── cam_1.pt             # 병 탐지 모델
│       └── cam_2.pt             # 액체 분할 모델
├── llm/
│   ├── web/
│   │   ├── order_engine/
│   │   │   ├── graph.py         # LangGraph 상태 머신
│   │   │   ├── state.py         # GraphState 타입 정의
│   │   │   ├── stt_pipeline.py  # STT (Google Gemini)
│   │   │   ├── tts_pipeline.py  # TTS (OpenAI gpt-4o-mini-tts)
│   │   │   ├── robot_topic.py   # ROS 토픽 발행
│   │   │   └── node/
│   │   │       ├── classify_order.py  # 주문 분류 노드
│   │   │       ├── make_order.py      # 주문 확인 노드
│   │   │       └── menu_detail.py     # 레시피 생성 노드
│   │   └── views.py             # Django HTTP 뷰
│   └── manage.py
├── robot/
│   └── src/bartender_test/bartender_test/
│       ├── defines.py           # 전역 설정 (자세 각도, 병별 파라미터)
│       ├── startup.py           # 부팅 시퀀스 및 노드 스폰
│       ├── manager.py           # 오케스트레이션 노드 (Pick→Pour→Place)
│       ├── gripper.py           # 그리퍼 제어 노드
│       ├── pick.py              # 병 집기 노드
│       ├── pour.py              # 따르기 + 스냅 복구 노드
│       ├── place.py             # 병 되돌리기 노드
│       └── snap.py              # 스냅 트리거 노드 (스페이스바 / 비전)
└── scripts/
    └── start_order_stack.py     # 전체 스택 통합 실행 스크립트
```

---

## 4. LLM (주문 엔진)

### 4.1 전체 데이터 흐름

```
브라우저 마이크 녹음 (WebM/MP4)
  → POST /stt_transcribe/
  → Google Gemini STT → STTResult {text, emotion, recommend_menu}
  → LangGraph 상태 머신
      ├─ classify_order: 직접 주문? / 추천 필요?
      ├─ (필요 시) make_order: 애매한 응답 처리
      └─ menu_detail: 레시피 생성 {병 이름: ml}
  → recipe 확정 시 → ROS 토픽 /bartender/order_detail 발행
  → TTS 응답 (WAV) 브라우저 재생
```

### 4.2 웹 인터페이스

Django 웹 서버를 통해 주문을 받습니다.

| URL | 메서드 | 역할 |
| :--- | :--- | :--- |
| `/` | GET | 홈 화면 |
| `/order/` | GET | 주문 시작 (세션 초기화) |
| `/stt_transcribe/` | POST | 음성 파일 → STT → 주문 처리 → TTS |
| `/robot_status/` | GET | 로봇 작업 상태 폴링 (`/tmp/bartender_order_status.json`) |
| `/tts/` | POST | 텍스트 → TTS WAV 반환 |

기본 접속 주소: `http://127.0.0.1:8000/order/`

### 4.3 LangGraph 상태 머신

`graph.py`는 3개 노드로 구성된 LangGraph 상태 머신입니다:

```
진입점 (route_entry)
  ├─ status == "init"       → classify_order
  ├─ status == "processing" → makeorder
  └─ status == "success"    → menu_detail → END

classify_order: 직접 주문 탐지
  ├─ 성공 (status=success)  → menu_detail
  └─ 추천 필요 (status=processing) → END (다음 턴 대기)

makeorder: 이전 추천 확인 또는 재시도
  ├─ 확인 (status=success)  → menu_detail
  └─ 거절/실패 (status=end) → END
```

**세션 상태 유지**: 여러 번의 발화를 거쳐 주문이 확정되므로, `retry`, `recommend_menu`, `ratio` 등이 세션을 통해 턴 간 유지됩니다.

### 4.4 STT 파이프라인 (Google Gemini)

- 모델: `gemini-flash-latest` (`.env`의 `STT_MODEL`로 변경 가능)
- 입력: WebM / MP4 오디오 바이트
- 출력: `{text, emotion, recommend_menu, reason}`
- 재시도: 429 / 500 / 503 응답 시 최대 2회 지수 백오프

**소주 vs. 주스 발음 혼동 처리**
한국어 특성상 "소주"와 "주스"가 비슷하게 들릴 수 있어 점수 기반 보정 로직을 사용합니다:

| 조건 | 점수 |
| :--- | :--- |
| 정확한 키워드 ("소주" / "주스") 포함 | +3 |
| 유사 발음 ("쏘주", "쥬스" 등) 포함 | +2 |
| Gemini 추천 메뉴 일치 | +1 |

두 점수가 모두 2점 이상이면 높은 쪽을 채택하며, 텍스트 내 메뉴어도 일치하도록 보정합니다.

### 4.5 TTS 파이프라인 (OpenAI)

- 모델: `gpt-4o-mini-tts`
- 음성: `nova` (기본값) / `alloy` / `onyx`
- 출력 형식: WAV (스트리밍, 4KB 청크)
- API 키: Django settings → `.env` 순서로 탐색

### 4.6 지원 메뉴

| 메뉴 코드 | 설명 |
| :--- | :--- |
| `soju` | 소주 단품 |
| `beer` | 맥주 단품 |
| `juice` | 주스 단품 |
| `somaek` | 소맥 (소주 + 맥주 비율 LLM 결정) |
| `koktail` | 칵테일 (비율 LLM 결정) |

---

## 5. Detection (비전 시스템)

### 5.1 카메라 1 — 병 탐지 (`realsense_cam1.py`)

카메라 1은 병 선반을 바라보며 픽업 대상 병의 위치를 탐지합니다.

- **마운트**: 카메라가 뒤집혀 장착 → 이미지 상하 반전 처리
- **모델**: YOLOv8 (`weights/cam_1.pt`), confidence ≥ 0.5
- **클래스**: 0=juice, 1=beer, 2=soju
- **출력**: 바운딩 박스 중심의 깊이(m) + 레이블 시각화 (OpenCV)

실제 픽업 좌표 계산은 `pick.py`가 담당하며, 카메라 1은 탐지 결과를 ROS 토픽으로 발행합니다.

### 5.2 카메라 2 — 액체 부피 측정 (`realsense_cam2.py`)

카메라 2는 따르기 동작 중 컵 안의 액체 양을 실시간으로 측정합니다.

**부피 측정 파이프라인:**

```
컵 안정 감지 (≥45 프레임)
  → 컵 전체 높이(px) 및 바닥 Y좌표 잠금 (자동 영점 조정, Auto-Tare)
  → 따르기 시작 후 (prepare_pouring 서비스 호출)
  → 액체 높이 / 컵 높이 = Ratio
  → Ratio → ml 변환 (룩업 테이블: 0.0~0.8434 → 0~500ml)
  → EMA 평활화 (alpha=0.2) → /dsr01/robotender/liquid_volume 발행
```

**노이즈 필터링:**
- 컵 면적의 5% 미만 액체는 반사/기포로 간주하여 무시 (`MIN_LIQUID_RATIO`)
- 컵 상단 5% 여유 패딩 적용 (`PADDING_RATIO`, 물결 흔들림 허용)
- 5 프레임 연속 액체 미감지 시 EMA 히스토리 초기화

**Detection ↔ Robot 핸드쉐이크 프로토콜:**

| 토픽 / 서비스 | 방향 | 역할 |
| :--- | :--- | :--- |
| `/dsr01/robotender/prepare_pouring` (Service) | Robot → Camera2 | 따르기 전 컵 영역 잠금 요청 |
| `/dsr01/robotender/pouring_status` (Topic) | Robot → Camera2 | 따르기 완료 후 EMA 상태 리셋 신호 |
| `/dsr01/robotender/flow_started` (Topic) | Camera2 → Robot | 액체 흐름 감지 시작 알림 |
| `/dsr01/robotender/liquid_volume` (Topic, Float32) | Camera2 → Robot | 현재 액체 부피 (ml) 실시간 발행 |

목표 부피 도달 시 카메라 2가 자동으로 스냅 트리거를 발행하여 로봇이 따르기를 중단합니다.

### 5.3 이전 버전 대비 개선 사항

| 항목 | 이전 (`od-realsense` 브랜치) | 현재 |
| :--- | :--- | :--- |
| 부피 계산 방식 | 절댓값 픽셀 높이 → 고정 룩업 테이블 | 비율(Ratio) 기반 룩업 테이블 |
| 노이즈 처리 | 없음 | EMA 필터 (alpha=0.2) |
| 컵 위치 | 고정 가정 | Auto-Tare (안정 감지 후 자동 잠금) |
| 따르기 종료 | 수동 | 목표 부피 도달 시 자동 트리거 |

---

## 6. Robot (로봇 제어 시스템)

### 6.1 노드 구성

| 노드 | 소스 파일 | 역할 | 주요 서비스 / 토픽 |
| :--- | :--- | :--- | :--- |
| `robotender_manager` | `manager.py` | 전체 시퀀스 오케스트레이션 | Sub: `/bartender/order_detail`, Pub: `/bartender/order_status` |
| `robotender_gripper` | `gripper.py` | 그리퍼 열기/닫기 (One-Shot DRL) | `/open` (Trigger), `/close` (Trigger) |
| `robotender_pick` | `pick.py` | 비전 기반 병 집기 | Action: `robotender_pick/execute`, Pub: `last_pose` |
| `robotender_pour` | `pour.py` | 따르기 + 스냅 복구 | Action: `robotender_pour/execute` |
| `robotender_place` | `place.py` | 병 원위치 복귀 | Action: `robotender_place/execute` |
| `robotender_snap` | `snap.py` | 스냅 트리거 수신 (스페이스바 / 비전) | Pub: `robotender_snap/trigger` |
| `robotender_monitor` | `monitor.py` | 통합 텔레메트리 표시 | Sub: `/joint_states` 등 |

### 6.2 시작 시퀀스 (`startup.py`)

전체 스택 실행 시 아래 순서로 로봇 상태를 검증한 후 로직 노드를 스폰합니다:

1. DSR 드라이버 안정화 대기 (5초)
2. 안전 상태 초기화 (`SetRobotControl(robot_control=2)`)
3. DRL 상태 확인 → idle이 아니면 `DrlStop` 호출
4. STANDBY 상태 확인 (최대 10회 폴링, `robot_state == 1`)
5. MoveJoint 서비스 준비 확인 (최대 15초)
6. 로직 노드 순차 스폰 (각 노드 준비 확인 후 다음 진행):
   - `gripper` → `pick` → `pour` → `place` → `manager`

> **주의**: 어느 단계에서든 실패하면 전체 스타트업이 중단됩니다. 실패 시 티치 펜던트에서 로봇 상태를 확인하세요.

### 6.3 Manager 노드 (`manager.py`)

주문 수신부터 병 반납까지 전체 흐름을 조율하는 핵심 오케스트레이션 노드입니다.

- **멀티스레드**: 8 스레드 executor (비동기 액션 체인 데드락 방지)
- **ingredient_queue**: 주문의 각 재료를 `(병 이름, ml)` 튜플 큐로 관리
- **상태 파일**: `/tmp/bartender_order_status.json`에 실시간 기록 → 웹 서버에서 폴링 가능

**운전 모드:**

| 모드 | 동작 |
| :--- | :--- |
| **Auto** | Pick → Pour → Place 자동 연속 실행 |
| **Manual** | 각 단계별로 서비스 호출 대기 |

### 6.4 그리퍼 노드 (`gripper.py`)

- **One-Shot DRL 인젝션**: 매 명령마다 `/dsr01/drl/drl_start` 서비스를 통해 DRL 코드를 직접 주입합니다. 그리퍼 스크립트가 로봇 컨트롤러를 점유하지 않도록 하기 위한 방식입니다.
- **Modbus 통신**: 플랜지 시리얼 포트(57600 baud)를 통해 FC06/FC16 커맨드로 RH-P12-RN을 제어합니다.

**병별 그리퍼 설정** (`defines.py`):

| 병 종류 | `gripper_pos` | `gripper_force` | 비고 |
| :--- | :--- | :--- | :--- |
| Juice | 800 | 400 | 원통형, 강한 파지 필요 |
| Beer | 570 | 90 | 얇은 유리병, 낮은 힘으로 손상 방지 |
| Soju | 425 | 200 | 중간 크기 |

### 6.5 픽업 노드 (`pick.py`)

- **비전-로봇 좌표 변환**: RGB + Depth 프레임 동기화 후 YOLOv8로 병 탐지, 사전 보정된 회전 행렬(R)과 이동 벡터(t)로 픽셀 → 로봇 베이스 좌표 변환
- **좌표 메모리**: 픽업 시 병의 정확한 XYZ를 `robotender_pick/last_pose` 토픽에 발행 → 이후 `place.py`가 원위치 복귀에 활용

### 6.6 따르기 노드 (`pour.py`)

따르기 중단 시 안전하게 복귀하는 **스냅 복구(Snap Recovery)** 로직이 핵심입니다.

**병렬 인터럽트 흐름:**

1. **스레드 A (모션)**: `movesx` 명령 전송 후 완료까지 블로킹 대기
2. **스레드 B (모니터)**: 비전 시스템 또는 스페이스바로부터 스냅 신호 독립 수신
3. **스냅 이벤트**: 스레드 B가 즉시 `MoveStop(Stop Mode 2)` 전송
4. **컨트롤러 반응**: 진행 중인 궤적 즉시 종료
5. **모션 해제**: 스레드 A가 깨어나 인터럽트 감지
6. **복구 실행**: 녹화된 경로를 역방향으로 고속 이동

**복구 실행 세부:**

1. **실시간 경로 녹화**: 따르는 동안 3Hz로 관절 자세(`posj`) 버퍼 기록
2. **동적 역추적**: 스냅 후 버퍼를 역순으로 다운샘플링 (1~3 포인트) → `CHEERS_POSE` 추가
3. **고속 복귀**: `movesj` 150 deg/s로 역경로 실행 → 빠른 복귀로 액체 넘침 방지

**병별 따르기 대기 시간** (`pour_wait_time`, 6개 볼륨 구간):

| 병 종류 | 최대 대기 시간 | `pour_velocity` |
| :--- | :--- | :--- |
| Juice | 6.0초 | 6 |
| Beer | 11.0초 | 6 |
| Soju | 7.8초 | 4 |

### 6.7 복귀 노드 (`place.py`)

- `robotender_pick/last_pose` 토픽을 구독하여 병의 원래 위치 기억
- `/dsr01/robotender_place/start` 서비스 호출 시 저장된 좌표로 병을 원위치에 반납

### 6.8 공통 설정 (`defines.py`)

모든 노드가 공유하는 중앙 설정 파일:

**주요 관절 자세 (단위: 도):**

| 자세 이름 | 설명 |
| :--- | :--- |
| `POSJ_HOME` | 기본 대기 자세 |
| `POSJ_PICK_READY` | 병 선반 접근 준비 자세 |
| `POSJ_CHEERS` | 건배 자세 (스냅 복구 복귀 목적지) |
| `POSJ_SNAP` | 병 확인용 사진 촬영 자세 |

**모션 상수:**

| 상수 | 값 | 설명 |
| :--- | :--- | :--- |
| `PICK_PLACE_Z` | 650 | 픽업/반납 Z 높이 (mm) |
| `SNAP_VELOCITY` | 150 deg/s | 스냅 복구 속도 |
| `VEL_APPROACH` | 100 | 접근/후퇴 속도 |
| `VEL_LIFT` | 75 | 병 들어올리기 속도 (느리게) |

---

## 7. 하드웨어 및 소프트웨어 요구사항

### 하드웨어

| 장비 | 용도 | 비고 |
| :--- | :--- | :--- |
| Doosan Robotics E0509 | 로봇팔 | IP: `110.120.1.68` (기본값) |
| Robotis RH-P12-RN | 그리퍼 | Modbus RTU over 로봇 플랜지 시리얼 (57600 baud) |
| Intel RealSense (카메라 1) | 병 위치 탐지 | 시리얼: `_311322302867` |
| Intel RealSense (카메라 2) | 액체 부피 측정 | 시리얼: `_313522301601` |
| 마이크 | 음성 주문 입력 | 웹 브라우저 마이크 사용 |

### 소프트웨어

- **OS**: Ubuntu 24.04
- **ROS 2**: Jazzy
- **Python**: 3.12 이상
- **Doosan DSR ROS 2 패키지** (별도 설치 필요 — [THIRD_PARTY.md](THIRD_PARTY.md) 참고)
- **Python 패키지** (자세한 내용은 `requirements.txt` 참고):
  ```
  Django>=5.0
  openai>=1.0.0
  python-dotenv>=1.0.0
  google-genai==1.65.0
  langgraph==1.0.10
  numpy<2
  opencv-python>=4.10.0
  ultralytics>=8.3.0
  pyrealsense2>=2.56.0
  ```

---

## 8. 설치 및 환경 설정

### 8.1 패키지 설치

```bash
# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 8.2 ROS 2 워크스페이스 빌드

```bash
cd robot
colcon build --symlink-install
```

### 8.3 환경 변수 설정 (`.env`)

프로젝트 루트에 `.env` 파일을 생성하고 아래 키를 설정합니다:

```dotenv
# STT (Google Gemini)
GOOGLE_API_KEY=your_google_api_key_here
# 또는
GEMINI_API_KEY=your_gemini_api_key_here

# LLM 주문 처리 + TTS (OpenAI)
OPENAI_API_KEY=your_openai_api_key_here

# STT 모델 (선택 사항, 기본값: gemini-flash-latest)
STT_MODEL=gemini-flash-latest
```

### 8.4 하드코딩 경로 확인 (중요)

`robot/src/bartender_test/bartender_test/startup.py`의 46번째 줄에 가상환경 Python 경로가 하드코딩되어 있습니다:

```python
VENV_PYTHON = "/home/fastcampus/bartender-robot/.venv/bin/python3"
```

배포 환경이 다를 경우 이 경로를 실제 환경에 맞게 수정해야 합니다. (예: `/home/{사용자명}/bartender-robot/.venv/bin/python3`)

### 8.5 YOLOv8 가중치 파일

아래 경로에 학습된 가중치 파일이 있어야 합니다:

```
detection/weights/cam_1.pt   # 병 탐지 (카메라 1용)
detection/weights/cam_2.pt   # 액체 분할 (카메라 2용)
```

---

## 9. 실행 명령어

### 9.1 전체 스택 시작

```bash
python3 scripts/start_order_stack.py --with-bringup
```

**모든 CLI 옵션:**

| 옵션 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `--with-bringup` | False | DSR 로봇 bringup 포함 실행 |
| `--bringup-cmd` | `dsr_bringup2 e0509 110.120.1.68` | bringup 명령어 오버라이드 |
| `--camera-cmd` | 시리얼 `_311322302867` | 카메라 1 실행 명령어 오버라이드 |
| `--camera2-cmd` | 시리얼 `_313522301601` | 카메라 2 실행 명령어 오버라이드 |
| `--skip-web` | False | Django 웹서버 실행 생략 |
| `--web-host` | `127.0.0.1` | Django 서버 호스트 |
| `--web-port` | `8000` | Django 서버 포트 |

**프로세스 시작 순서:**
1. bringup / 카메라 1, 2 / 부피 측정 (병렬, 0.3초 간격)
2. `startup.py` (블로킹 — 완료까지 대기)
3. Django 웹서버 (startup 완료 후 시작)

### 9.2 모드 전환 (실시간)

```bash
# 수동 모드 전환 (주문 수신 후 자동 실행 안 함)
ros2 topic pub --once /dsr01/robotender_manager/mode std_msgs/msg/String "{data: 'manual'}"

# 자동 모드 전환 (Pick → Pour → Place 자동 연속 실행)
ros2 topic pub --once /dsr01/robotender_manager/mode std_msgs/msg/String "{data: 'auto'}"
```

### 9.3 CLI 주문 주입 (웹 UI 없이)

```bash
ros2 topic pub --once /bartender/order_detail std_msgs/msg/String "{data: '{\"recipe\": {\"soju\": 1}}'}"
```

### 9.4 수동 모드 단계별 실행

수동 모드에서 각 단계를 개별 호출합니다:

```bash
# 픽업
ros2 service call /dsr01/robotender_manager/pick_bottle std_srvs/srv/Trigger {}

# 따르기
ros2 service call /dsr01/robotender_manager/pour_bottle std_srvs/srv/Trigger {}

# 반납
ros2 service call /dsr01/robotender_manager/place_bottle std_srvs/srv/Trigger {}
```

### 9.5 스냅 수동 트리거

```bash
# snap 노드를 별도 터미널에서 실행 후 스페이스바로 트리거
python3 -m bartender_test.snap
```

---

## 10. 향후 개선 사항

### Detection

- **경계 EMA 안정화**: 컵 경계(`bottle_area_ema`)에도 EMA 필터 적용하여 보정 중 기준 스케일 안정화
- **동적 목표 부피**: `target_volume_ml`을 고정값 대신 LLM 주문 엔진에서 동적으로 설정
- **투명/반사 용기 지원**: 유리컵 등 반사 재질에 대한 깊이 인식 기반 분할 모델 개선

### 로봇 제어 및 안전

- **좌표 보정**: 그리퍼 접근 시 발생하는 수평 오프셋 정밀 보정
- **병 종류별 Z 오프셋**: 소주/맥주/주스 각각에 최적화된 Z축 파지 높이 정의
- **자세 전환 부드러움**: HOME → CHEERS → CONTACT 간 관절 공간 전환 최적화로 기계적 충격 감소
- **동적 그리퍼 힘 제어**: 개봉된 병 등 구조 변형이 있는 용기에 대한 힘 제어 개선
- **동적 장애물 회피**: RealSense 깊이 데이터 활용 실시간 충돌 회피
- **캘리브레이션 자동화**: 카메라-로봇 베이스 간 외부 파라미터 자동 보정

---

*2026년 3월 기준*

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

## 서드파티 의존성

이 프로젝트는 Doosan Robotics ROS 2 패키지 등 외부 로보틱스 패키지에 의존합니다.
해당 패키지들은 각자의 라이선스를 따르며 본 MIT 라이선스의 적용을 받지 않습니다.
자세한 내용은 [THIRD_PARTY.md](THIRD_PARTY.md)를 참고하세요.
