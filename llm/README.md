# 사용자 주문을 맞는 로직 개발

## 1. 개발 환경 설치
1) 가상환경 설치
- uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
- 활성화 : 
uv venv .venv --python 3.12
source .venv/bin/activate

2) 필수환경 설치
uv pip install -r requirements.txt

3) 웹 실행
python manage.py runserver
브라우저에서 http://127.0.0.1:8000/ 확인

4) 터미널 STT 테스트(OpenAI)
uv run python web/order_engine/stt_cli.py

