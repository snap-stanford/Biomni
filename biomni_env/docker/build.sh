#!/bin/bash
# Docker 이미지 빌드 스크립트
# 
# 사용법: 
#   ./biomni_env/docker/build.sh              # 기본 빌드
#   ./biomni_env/docker/build.sh --no-cache   # 캐시 없이 새로 빌드

set -e

# 프로젝트 루트로 이동 (스크립트 위치 기준 2단계 상위)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

echo "====================================="
echo "Biomni HITS Docker Image Build"
echo "====================================="
echo "Project root: ${PROJECT_ROOT}"
echo ""

IMAGE_NAME="biomni-hits"
IMAGE_TAG="latest"

# 캐시 옵션 처리
BUILD_OPTS=""
if [[ "$1" == "--no-cache" ]]; then
    BUILD_OPTS="--no-cache"
    echo "Building without cache..."
fi

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# Docker 이미지 빌드
docker build ${BUILD_OPTS} \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    -f biomni_env/docker/Dockerfile \
    .

echo ""
echo "====================================="
echo "Build completed successfully!"
echo "====================================="
echo ""
echo "이미지 실행 방법:"
echo ""
echo "  1. 기본 실행 (인터랙티브 쉘):"
echo "     docker run -it --rm ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "  2. 현재 디렉토리 마운트:"
echo "     docker run -it --rm -v \$(pwd):/workspace ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "  3. Jupyter 서버 실행:"
echo "     docker run -it --rm -p 8888:8888 -v \$(pwd):/workspace ${IMAGE_NAME}:${IMAGE_TAG} \\"
echo "       jupyter notebook --ip=0.0.0.0 --allow-root"
echo ""
echo "  4. Chainlit 서버 실행:"
echo "     docker run -it --rm -p 8000:8000 -v \$(pwd):/workspace ${IMAGE_NAME}:${IMAGE_TAG} \\"
echo "       chainlit run app.py"
echo ""

