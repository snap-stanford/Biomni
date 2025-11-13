#!/bin/bash

# Chainlit 서버 실행 스크립트
# 
# 연결 끊김 문제 해결 방법:
# 1. 하트비트 간격을 2초로 설정하여 chunk가 안 와도 매우 자주 연결 유지
# 2. 무조건 2초마다 update() 호출하여 HTTP/웹소켓 연결 유지
# 3. 최대 타임아웃을 60분(3600초)으로 증가
# 4. session_timeout을 2시간으로 증가
# 5. 5초 이상 대기 시에만 visible 메시지 표시
#
# 참고: 만약 리버스 프록시나 로드 밸런서를 사용하는 경우,
# 해당 서버의 타임아웃 설정도 확인해야 할 수 있습니다.
# Nginx 예시: proxy_read_timeout 7200;

# 환경 변수 설정 - 웹소켓 타임아웃 증가
export CHAINLIT_WS_TIMEOUT=7200
export CHAINLIT_REQUEST_TIMEOUT=7200

# Chainlit 서버 실행
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chainlit run "${SCRIPT_DIR}/run.py" -h --host 0.0.0.0 --port 8004
