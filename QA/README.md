# HITS AI Agent QA System

HITS AI Agent의 답변 품질을 자동으로 평가하는 QA 시스템입니다.

## 📋 주요 기능

- ✅ **자동 평가**: LLM을 활용한 답변 정확도 평가
- 🖼️ **이미지 검증**: 마크다운 내 이미지 존재 확인 및 SSIM 기반 시각적 유사도 비교
- 📊 **상세 리포트**: JSON 및 Markdown 형식의 평가 리포트 생성
- 🔁 **반복 실행**: Task당 N번 반복 실행, 모든 시도가 성공해야 통과
- ⚡ **병렬 처리**: GNU parallel을 사용한 안정적인 병렬 실행
- 🔄 **확장 가능**: 새로운 QA 태스크를 쉽게 추가 가능

## 🏗️ 아키텍처

### 새로운 분리 아키텍처 (권장) ✨

깔끔하고 안정적인 2-tier 아키텍처:

```
QA/
├── qa_single_task.py      # 단일 task 실행기 (완전히 독립)
├── qa_runner_simple.py    # Parallel wrapper (전체 파이프라인 관리)
├── qa_runner.py           # 레거시 (하위 호환성)
└── qa_core/               # 공통 모듈
```

#### 1. `qa_single_task.py` - Single Task Executor

단일 QA task의 단일 attempt를 완전히 독립적으로 실행

**특징**:
- ✅ 완전히 독립된 스크립트
- ✅ 명령줄로 직접 실행 가능
- ✅ 디버깅 용이
- ✅ 외부 의존성 없음

**사용법**:
```bash
python qa_single_task.py \
    --task-id task_001 \
    --attempt 1 \
    --qa-datasets-dir qa_datasets \
    --output-dir qa_results/run_xxx \
    --pass-threshold 70 \
    --ssim-threshold 0.8
```

#### 2. `qa_runner_simple.py` - Parallel Wrapper

전체 QA 파이프라인 관리 및 병렬 실행

**특징**:
- ✅ 간단하고 명확한 코드
- ✅ GNU parallel 또는 xargs 사용
- ✅ 커맨드 생성 + 결과 수집만 담당
- ✅ 안정적이고 예측 가능

**실행 플로우**:
```
qa_runner_simple.py
    │
    ├─→ 커맨드 생성
    │   └─→ commands.txt
    │
    ├─→ GNU parallel / xargs 실행
    │   ├─→ qa_single_task.py (독립 프로세스)
    │   ├─→ qa_single_task.py (독립 프로세스)
    │   └─→ qa_single_task.py (독립 프로세스)
    │
    └─→ 결과 수집 및 리포트 생성
```

### 장점

#### 1. **완전한 격리**
- 각 `qa_single_task.py`는 독립된 Python 프로세스
- 전역 변수, 상태, 파일 시스템 충돌 없음
- 한 task 실패가 다른 task에 영향 없음

#### 2. **단순성**
```python
commands = generate_commands(tasks)  # 커맨드 생성
execute_parallel(commands)           # parallel 실행
results = collect_results()          # 결과 수집
print_summary(results)               # 통계 출력
```

#### 3. **디버깅 용이**
```bash
# 단일 task를 직접 실행하여 디버깅
python qa_single_task.py --task-id problematic_task ...
```

#### 4. **유연성**
```bash
# GNU parallel 고급 옵션
parallel --jobs 5 --eta --resume < commands.txt

# 실패한 것만 재실행
parallel --jobs 5 --resume-failed < commands.txt

# 분산 실행
parallel --sshloginfile servers.txt < commands.txt
```

## 🚀 빠른 시작

### 1. 기본 실행 (새 방식, 권장)

```bash
# 모든 태스크 실행
python qa_runner_simple.py

# 특정 태스크만 실행
python qa_runner_simple.py --tasks task_001 task_002

# 병렬 실행 (3개 동시)
python qa_runner_simple.py --repeat 3 --max-workers 3
```

### 2. 태스크 목록 확인

```bash
python qa_runner_simple.py --list-tasks
```

### 3. 단일 태스크 실행 (디버깅용)

```bash
python qa_single_task.py \
    --task-id task_001 \
    --attempt 1 \
    --qa-datasets-dir qa_datasets \
    --output-dir qa_results/debug
```

### 4. 레거시 방식 (하위 호환성)

```bash
python qa_runner.py --repeat 3 --max-workers 3
```

## 📁 폴더 구조

```
QA/
├── qa_datasets/              # QA 데이터셋
│   ├── task_001/
│   │   ├── question.md      # 질문
│   │   ├── answer.md        # 정답
│   │   └── metadata.json    # 메타데이터 (optional)
│   └── task_002/
│       ├── question.md
│       ├── answer.md
│       ├── histogram.png    # 정답 이미지
│       └── boxplot.png
│
├── qa_results/               # 실행 결과
│   └── run_20251128_120000/
│       ├── commands.txt     # 실행된 모든 커맨드
│       ├── summary_report.md
│       ├── task_001/
│       │   ├── attempt_1/
│       │   │   ├── question.md
│       │   │   ├── generated_answer.md    # 최종 답변
│       │   │   ├── full_response.md       # 전체 응답
│       │   │   ├── agent_steps.md         # 중간 단계
│       │   │   ├── evaluation.json
│       │   │   └── *.png                  # 생성된 이미지
│       │   ├── attempt_2/
│       │   └── attempt_3/
│       └── task_002/
│
├── qa_core/                  # 공통 모듈
│   ├── qa_manager.py
│   ├── evaluator.py
│   ├── image_comparator.py
│   └── report_generator.py
│
├── qa_single_task.py         # ✨ 단일 task 실행기
├── qa_runner_simple.py       # ✨ Parallel wrapper
└── qa_runner.py              # 레거시
```

## ⚙️ CLI 옵션

### qa_runner_simple.py (권장)

```bash
python qa_runner_simple.py [OPTIONS]

옵션:
  --qa-datasets-dir DIR      QA 데이터셋 디렉토리 (기본: qa_datasets)
  --output-dir DIR           결과 출력 디렉토리 (기본: qa_results)
  --tasks TASK_ID [...]      실행할 태스크 ID 리스트
  --category CATEGORY        카테고리로 필터링
  --pass-threshold SCORE     통과 기준 점수 0-100 (기본: 70)
  --ssim-threshold SCORE     SSIM 임계값 0-1 (기본: 0.8)
  --repeat N                 각 태스크당 반복 실행 횟수 (기본: 1)
  --max-workers N            병렬 실행 최대 워커 수 (기본: 1)
  --list-tasks               태스크 목록 출력 후 종료
```

### qa_single_task.py (디버깅용)

```bash
python qa_single_task.py [OPTIONS]

필수 옵션:
  --task-id TASK_ID          실행할 태스크 ID
  --attempt N                시도 번호 (1부터 시작)
  --qa-datasets-dir DIR      QA 데이터셋 디렉토리
  --output-dir DIR           출력 디렉토리

선택 옵션:
  --total-attempts N         총 시도 횟수 (기본: 1)
  --pass-threshold SCORE     통과 기준 (기본: 70)
  --ssim-threshold SCORE     SSIM 임계값 (기본: 0.8)
```

## 📝 사용 예제

### 기본 실행

```bash
# 모든 태스크 실행 (순차)
python qa_runner_simple.py

# 병렬 실행 (권장)
python qa_runner_simple.py --max-workers 3
```

### 반복 실행

```bash
# 각 task를 3번씩 반복 (모든 시도가 성공해야 통과)
python qa_runner_simple.py --repeat 3

# 반복 + 병렬
python qa_runner_simple.py --repeat 3 --max-workers 3
```

### 특정 태스크만

```bash
# 특정 태스크만 실행
python qa_runner_simple.py --tasks task_001 task_002

# 카테고리별 실행
python qa_runner_simple.py --category genomics
```

### 디버깅

```bash
# 단일 태스크 디버깅
python qa_single_task.py \
    --task-id problematic_task \
    --attempt 1 \
    --qa-datasets-dir qa_datasets \
    --output-dir qa_results/debug

# 로그 저장
python qa_single_task.py ... 2>&1 | tee debug.log
```

### 고급 사용

```bash
# 실패한 것만 재실행
cd qa_results/run_20251128_120000
parallel --jobs 5 --resume-failed < commands.txt

# 진행률 표시
parallel --jobs 5 --eta < commands.txt

# 분산 실행 (여러 서버)
parallel --sshloginfile servers.txt < commands.txt
```

## 📊 평가 기준

### 텍스트 평가 (LLM 기반)

1. **Content Accuracy (50%)**: 내용 정확도
   - 90-100: 거의 동일하거나 동등하게 정확
   - 70-89: 대부분 정확하나 미세한 차이
   - 50-69: 부분적으로 정확하나 핵심 정보 누락
   - 0-49: 부정확하거나 크게 다름

2. **Completeness (30%)**: 완성도
   - 모든 질문에 답변했는가?
   - 필요한 모든 섹션이 있는가?
   - 설명 깊이가 적절한가?

3. **Format Compliance (20%)**: 형식 준수
   - 마크다운 문법 정확성
   - 구조 및 가독성

### 이미지 평가

1. **존재 확인**: 정답에 포함된 모든 이미지가 생성되었는가?
2. **SSIM 비교**: 생성된 이미지가 정답 이미지와 시각적으로 유사한가?
   - SSIM > 0.8: 매우 유사
   - SSIM 0.6-0.8: 유사
   - SSIM < 0.6: 다름

### 통과 기준

- 텍스트 점수 ≥ 70점 (기본값)
- 모든 필수 이미지 생성됨
- 이미지 SSIM ≥ 0.8 (기본값)
- **반복 실행 시**: 모든 시도가 위 기준을 충족해야 최종 통과

## 📄 리포트 형식

### summary_report.md

```markdown
# QA Pipeline Summary Report

## Configuration
- Run ID: run_20251128_120000
- Total Tasks: 10
- Repeats per Task: 3
- Max Workers: 3

## Overall Results
| Metric | Value |
|--------|-------|
| ✅ Passed Tasks | 8/10 |
| ❌ Failed Tasks | 2/10 |
| 📊 Success Rate | 80.0% |

## Task Details
| Status | Task ID | Attempts Passed | Avg Time |
|--------|---------|----------------|----------|
| ✅ PASS | task_001 | 3/3 | 45.2s |
| ❌ FAIL | task_002 | 2/3 | 38.1s |
```

### evaluation.json (각 attempt)

```json
{
  "task_id": "task_001_attempt1",
  "execution_time_seconds": 45.2,
  "text_evaluation": {
    "scores": {
      "content_accuracy": 85,
      "completeness": 90,
      "format_compliance": 95
    },
    "overall_score": 88.5,
    "passed": true
  },
  "image_evaluation": {
    "all_images_present": true,
    "average_similarity": 0.92
  },
  "summary": {
    "overall_passed": true
  }
}
```

## 💡 권장 사항

### 안정성 우선

```bash
# 순차 실행 (가장 안정적)
python qa_runner_simple.py --repeat 3 --max-workers 1
```

### 속도와 안정성 균형 (권장)

```bash
# 3-5개 동시 실행
python qa_runner_simple.py --repeat 3 --max-workers 3
```

### 최대 성능 (리소스 충분 시)

```bash
# 5-10개 동시 실행
python qa_runner_simple.py --repeat 3 --max-workers 5
```

## 🔧 트러블슈팅

### Q: GNU parallel이 없다는 메시지가 나옵니다

**A**: 자동으로 `xargs`로 fallback됩니다. 하지만 GNU parallel 설치를 권장합니다:

```bash
# Ubuntu/Debian
sudo apt install parallel

# macOS
brew install parallel
```

### Q: 병렬 실행 시 일부 task가 실패합니다

**A**: 워커 수를 줄이거나 순차 실행을 시도하세요:

```bash
# 워커 수 감소
python qa_runner_simple.py --max-workers 1

# 또는 단일 task 디버깅
python qa_single_task.py --task-id failing_task ...
```

### Q: 실패한 task만 재실행하고 싶습니다

**A**: GNU parallel의 재개 기능을 사용하세요:

```bash
cd qa_results/run_xxx
parallel --jobs 5 --resume-failed < commands.txt
```

### Q: 디버깅을 어떻게 하나요?

**A**: 단일 task를 직접 실행하세요:

```bash
python qa_single_task.py \
    --task-id problematic_task \
    --attempt 1 \
    --qa-datasets-dir qa_datasets \
    --output-dir qa_results/debug \
    2>&1 | tee debug.log
```

## 📦 의존성

- Python 3.10+
- Biomni HITS (AI agent)
- PIL (Pillow) - 이미지 처리
- scikit-image - SSIM 계산
- numpy - 배열 연산
- GNU parallel (선택사항, 권장)

## 🆕 새 태스크 추가

### 1. 태스크 폴더 생성

```bash
cd qa_datasets
mkdir task_004
cd task_004
```

### 2. 필수 파일 작성

```bash
# question.md - 질문 작성
cat > question.md << 'EOF'
# 데이터 분석

다음 데이터를 분석하고 시각화해주세요:
[10, 20, 15, 25, 30, 18, 22, 28, 16, 24]
EOF

# answer.md - 정답 작성
cat > answer.md << 'EOF'
# 분석 결과

평균: 20.8
중앙값: 21.0

![분석 결과](analysis.png)
EOF
```

### 3. 메타데이터 추가 (선택)

```bash
cat > metadata.json << 'EOF'
{
  "category": "data_analysis",
  "difficulty": "easy",
  "created_at": "2024-11-28",
  "requires_input_data": false
}
EOF
```

### 4. 실행

```bash
python qa_runner_simple.py --tasks task_004
```

## 📈 성능 비교

### 레거시 vs 새 아키텍처

| 항목 | 레거시 (qa_runner.py) | 새 아키텍처 (qa_runner_simple.py) |
|------|----------------------|----------------------------------|
| 코드 복잡도 | 높음 (1000+ lines) | 낮음 (~800 lines total) |
| 디버깅 | 어려움 | 쉬움 (단일 task 직접 실행) |
| 안정성 | 중간 (subprocess 관리) | 높음 (완전한 격리) |
| 유연성 | 제한적 | 높음 (GNU parallel) |
| 확장성 | 제한적 | 높음 (분산 실행 가능) |
| 유지보수 | 어려움 | 쉬움 (명확한 분리) |

## 🎯 마이그레이션 가이드

### 레거시에서 새 방식으로

**기존**:
```bash
python qa_runner.py --repeat 3 --max-workers 3 --tasks task_001
```

**새 방식** (동일한 기능):
```bash
python qa_runner_simple.py --repeat 3 --max-workers 3 --tasks task_001
```

동일한 인터페이스, 더 안정적인 실행!

## 🤝 기여

새로운 평가 기준이나 기능을 추가하려면:

1. `qa_single_task.py` - 단일 task 실행 로직 수정
2. `qa_core/` - 공통 모듈 수정
3. `qa_runner_simple.py` - 파이프라인 관리 (거의 수정 불필요)

## 📝 라이선스

Biomni HITS 프로젝트의 라이선스를 따릅니다.

---

**문의사항**: Biomni HITS 팀
