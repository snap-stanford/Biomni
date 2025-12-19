# Biomni HITS Docker

Docker 이미지로 Biomni HITS 환경을 빌드하고 실행하는 방법입니다.

## 빌드

프로젝트 루트 디렉토리에서 실행:

```bash
# 방법 1: 빌드 스크립트 사용
chmod +x biomni_env/docker/build.sh
./biomni_env/docker/build.sh

# 방법 2: docker build 직접 실행
docker build -t biomni-hits:latest -f biomni_env/docker/Dockerfile .

# 캐시 없이 새로 빌드
./biomni_env/docker/build.sh --no-cache
```

## 실행

```bash
# 기본 실행 (인터랙티브 쉘)
docker run -it --rm biomni-hits:latest

# 현재 디렉토리 마운트
docker run -it --rm -v $(pwd):/workspace biomni-hits:latest

# Jupyter 서버 실행
docker run -it --rm -p 8888:8888 -v $(pwd):/workspace biomni-hits:latest \
  jupyter notebook --ip=0.0.0.0 --allow-root

# Chainlit 서버 실행
docker run -it --rm -p 8000:8000 -v $(pwd):/workspace biomni-hits:latest \
  chainlit run app.py
```

## 파일 구조

```
biomni_env/docker/
├── Dockerfile           # Docker 이미지 정의
├── build.sh             # 빌드 스크립트
├── environment.yml      # Python 기본 패키지
├── bio_env.yml          # 생물정보학 패키지
├── r_packages.yml       # R 기본 패키지 (conda)
├── install_r_packages.R # Bioconductor R 패키지
└── README.md            # 이 파일
```

## 빌드 시간

- 첫 빌드: 약 30분~1시간 (R 패키지 포함)
- 캐시 사용 시: 훨씬 빠름

## 포트

| 포트 | 용도 |
|------|------|
| 8888 | Jupyter Notebook |
| 8000 | Chainlit |
| 7860 | Gradio |

