# 빠른 시작 가이드

HITS AI Agent QA 시스템을 빠르게 시작하는 방법입니다.

## 1️⃣ 설치 확인

필수 의존성 설치:

```bash
pip install pillow scikit-image numpy
```

## 2️⃣ 태스크 확인

현재 등록된 QA 태스크 확인:

```bash
cd /path/to/Biomni_HITS/QA
python qa_runner.py --list-tasks
```

예상 출력:
```
📋 Available Tasks:
============================================================
  - task_001 (Category: genetics, Difficulty: easy)
  - task_002 (Category: statistics, Difficulty: medium)
============================================================
Total: 2 tasks
```

## 3️⃣ 단일 태스크 실행

하나의 태스크만 테스트:

```bash
python qa_runner.py --tasks task_001
```

## 4️⃣ 전체 실행

모든 태스크 실행:

```bash
# 기본 실행 (순차, 1회)
python qa_runner.py

# 각 task를 3번씩 반복 실행
python qa_runner.py --repeat 3

# 10개의 작업을 동시에 병렬 실행
python qa_runner.py --max-workers 10

# 각 task를 3번씩 반복 + 10개 동시 병렬 실행
python qa_runner.py --repeat 3 --max-workers 10
```

## 5️⃣ 결과 확인

실행 완료 후:

```bash
# 결과 디렉토리로 이동
cd qa_results/run_20241124_120000/  # 실제 타임스탬프로 변경

# 종합 리포트 보기
cat summary_report.md

# 개별 태스크 결과 보기 (--repeat 미사용 시)
cat task_001/evaluation.json

# 반복 실행 시 각 시도별 결과 확인
cat task_001/attempt_1/evaluation.json
cat task_001/attempt_1/generated_answer.md  # 최종 답변 (solution 태그 추출)
cat task_001/attempt_1/full_response.md     # Agent 전체 응답
cat task_001/attempt_1/agent_steps.md       # Agent 실행 중간 단계
```

**반복 실행 (--repeat) 사용 시:**
- 각 task는 여러 번 실행되며, 각 시도는 `attempt_1`, `attempt_2`, ... 폴더에 저장됩니다
- **모든 시도가 성공해야** 해당 task가 최종 통과로 간주됩니다
- 종합 리포트에 각 task의 성공한 시도 수가 표시됩니다

**각 attempt 폴더 구조:**
```
attempt_1/
├── question.md              # 질문
├── generated_answer.md      # 최종 답변 (평가 대상)
├── full_response.md         # Agent 전체 응답
├── agent_steps.md           # 실행 중간 단계 (디버깅용)
├── evaluation.json          # 평가 결과
└── *.png                    # 생성된 이미지들
```

## 📊 결과 해석

### 단일 실행 (--repeat 미사용)

evaluation.json 예시:

```json
{
  "task_id": "task_001",
  "text_evaluation": {
    "scores": {
      "content_accuracy": 85,
      "completeness": 90,
      "format_compliance": 95
    },
    "overall_score": 88.5,
    "passed": true
  },
  "summary": {
    "overall_passed": true
  }
}
```

- `overall_score >= 70`: 통과 ✅
- `overall_score < 70`: 실패 ❌

### 반복 실행 (--repeat 사용)

summary_report.md 예시:

```markdown
### task_001: ✅ PASS
- Attempts: 3/3 passed
- All attempts must pass: True

### task_002: ❌ FAIL
- Attempts: 2/3 passed
- All attempts must pass: False
```

- 모든 시도가 성공 (3/3): 통과 ✅
- 하나라도 실패 (2/3): 실패 ❌

## 🔧 문제 해결

### AI Agent가 초기화되지 않는 경우

```python
# Python에서 직접 테스트
from biomni.agent.a1_hits import A1_HITS

agent = A1_HITS()
response = agent.run("Hello, test")
print(response)
```

### LLM이 응답하지 않는 경우

```python
# LLM 테스트
from biomni.llm import get_llm

llm = get_llm()
result = llm("Hello")
print(result)
```

## 📝 새 태스크 추가

1. 폴더 생성:
```bash
mkdir qa_datasets/task_003
```

2. 파일 작성:
```bash
# question.md 작성
nano qa_datasets/task_003/question.md

# answer.md 작성
nano qa_datasets/task_003/answer.md

# 이미지 추가 (있는 경우, task 폴더 바로 아래에)
cp /path/to/image.png qa_datasets/task_003/

# metadata.json 작성 (optional)
nano qa_datasets/task_003/metadata.json
```

3. 실행:
```bash
python qa_runner.py --tasks task_003
```

## ⚡ 성능 최적화 팁

### 병렬 실행 권장 사항

```bash
# Task가 많을 때 (10개 이상)
python qa_runner.py --max-workers 10

# Task가 적고 반복이 많을 때 (task 5개 × 3번 = 15개 실행)
python qa_runner.py --repeat 3 --max-workers 15

# 안정성 중시 (순차 실행)
python qa_runner.py --repeat 5 --max-workers 1
```

**주의사항:**
- `max-workers`를 너무 크게 설정하면 LLM API 요청 제한에 걸릴 수 있습니다
- CPU/메모리 리소스를 고려하여 적절한 값을 설정하세요
- 권장: 5-10 정도가 적당합니다

**문제 해결:**

**증상**: 병렬 실행 시 프로세스가 멈추거나 파일 충돌 발생

**원인**:
- Python 코드 실행 시 전역 변수 충돌 가능
- 상대 경로로 파일 저장 시 충돌 (`plt.savefig("plot.png")`)
- 임시 파일 생성 충돌

**해결**:
1. ✅ **완전한 격리**: 각 작업이 독립 Python 프로세스 + 독립 working directory
2. ✅ **고유 식별자**: 환경 변수로 고유 ID 전달 (`QA_RUNNER_UNIQUE_ID`)
3. ✅ **Working directory 분리**: 각 attempt가 자신의 디렉토리에서 실행
4. 💡 **권장 워커 수**: 3-5 (안정성과 속도의 균형)
5. 🛡️ **안정성 우선**: `--max-workers 1`로 순차 실행

## 🎯 다음 단계

- [README.md](README.md) - 전체 문서 보기
- `qa_core/` - 코드 커스터마이징
- `qa_config/` - 평가 프롬프트 수정

---

**Happy Testing!** 🚀

