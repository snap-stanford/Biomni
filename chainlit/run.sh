#!/bin/bash

# Chainlit 서버 실행 스크립트
# 
# "Could not reach the server" 오류 해결 방법:
# 1. 하트비트 간격을 15초로 단축하여 연결을 더 자주 유지
# 2. 최대 타임아웃을 30분(1800초)으로 증가
# 3. 하트비트 메커니즘이 주기적으로 연결 상태를 확인하여 연결 유지
#
# 참고: 만약 리버스 프록시나 로드 밸런서를 사용하는 경우,
# 해당 서버의 타임아웃 설정도 확인해야 할 수 있습니다.

# Chainlit 서버 실행
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chainlit run "${SCRIPT_DIR}/run.py" -w -h --host 0.0.0.0 --port 8001
