# Biomni HITS Benchmark Evaluation

이 디렉토리는 Biomni HITS 에이전트의 벤치마크 실행 및 평가를 위한 도구를 제공합니다.

## 📁 파일 구조

```
eval/
├── benchmark.py              # 메인 벤치마크 실행 스크립트 (GNU parallel/xargs 방식)
├── benchmark_single_task.py  # 단일 작업 실행 스크립트 (benchmark.py가 내부적으로 호출)
├── evaluate.py               # 결과 평가 스크립트
├── biomni_eval1.py          # BiomniEval1 평가 로직 클래스
└── README.md                 # 이 파일
```

## 🚀 빠른 시작

### 1. 벤치마크 실행

```bash
# 기본 실행 (모든 데이터셋, 8개 병렬)
python benchmark.py

# 특정 데이터셋 실행
python benchmark.py -d DbQA -n 8

# 여러 데이터셋 동시 실행
python benchmark.py -d DbQA SeqQA HLE -n 8
```

### 2. 결과 평가

```bash
# 기본 평가
python evaluate.py results/20241128_120000

# 오류 인덱스 표시
python evaluate.py results/20241128_120000 --show-errors

# 상세 오류 정보 표시 (질문/정답/예측 포함)
python evaluate.py results/20241128_120000 --show-errors --verbose
```

## 📊 지원 데이터셋

### Original Benchmarks
- **DbQA**: Database Question Answering (60 instances)
- **SeqQA**: Sequence Question Answering (60 instances)
- **HLE**: Humanity Last Exam (52 instances)

### BiomniEval1 Tasks
- `crispr_delivery`: CRISPR 전달 방법 선택
- `gwas_causal_gene_opentargets`: GWAS 원인 유전자 (OpenTargets)
- `gwas_causal_gene_pharmaprojects`: GWAS 원인 유전자 (PharmaProjects)
- `gwas_causal_gene_gwas_catalog`: GWAS 원인 유전자 (GWAS Catalog)
- `gwas_variant_prioritization`: GWAS 변이 우선순위
- `lab_bench_dbqa`: Lab Bench DbQA
- `lab_bench_seqqa`: Lab Bench SeqQA
- `rare_disease_diagnosis`: 희귀 질환 진단
- `screen_gene_retrieval`: 스크린 유전자 검색
- `patient_gene_detection`: 환자 유전자 검출

## 🔧 benchmark.py 상세 사용법

### 기본 옵션

```bash
python benchmark.py [OPTIONS]
```

| 옵션 | 짧은 형식 | 기본값 | 설명 |
|------|-----------|--------|------|
| `--dataset` | `-d` | `all` | 실행할 데이터셋 (여러 개 지정 가능) |
| `--indices` | `-i` | (전체) | 실행할 인덱스 (예: "0,5,10" 또는 "0-10") |
| `--max-workers` | `-n` | `8` | 병렬 작업 수 |
| `--folder` | `-f` | (타임스탬프) | 결과 저장 폴더명 |
| `--skip-existing` | `-s` | `False` | 기존 결과 건너뛰기 |
| `--llm` | `-l` | `gemini-2.5-pro` | 사용할 LLM 모델 |

### 실행 예제

#### 기본 실행
```bash
# 모든 데이터셋 실행
python benchmark.py

# 특정 데이터셋 실행
python benchmark.py -d DbQA
python benchmark.py -d gwas_causal_gene_opentargets
```

#### 병렬 처리
```bash
# 4개 병렬로 실행
python benchmark.py -d SeqQA -n 4

# 16개 병렬로 실행 (고성능 서버)
python benchmark.py -d all -n 16
```

#### 인덱스 선택
```bash
# 특정 인덱스만 실행
python benchmark.py -d DbQA -i "0,5,10,15"

# 범위 지정
python benchmark.py -d SeqQA -i "0-20"

# 혼합 사용
python benchmark.py -d HLE -i "0,5-10,15,20-25"
```

#### 결과 폴더 관리
```bash
# 커스텀 폴더명 사용
python benchmark.py -d DbQA -f test_run_1

# 기존 결과 건너뛰기 (재실행 시 유용)
python benchmark.py -d DbQA -s
```

#### 다양한 LLM 모델
```bash
# Gemini Flash 사용
python benchmark.py -d DbQA -l gemini-2.5-flash

# Claude 사용
python benchmark.py -d SeqQA -l claude-sonnet-4
```

#### 복합 사용
```bash
# 여러 옵션 조합
python benchmark.py -d gwas_causal_gene_opentargets gwas_variant_prioritization \
  -i "0-50" -n 8 -f gwas_test -s -l gemini-2.5-flash
```

## 📈 evaluate.py 상세 사용법

### 기본 옵션

```bash
python evaluate.py [DIRECTORY] [OPTIONS]
```

| 옵션 | 짧은 형식 | 설명 |
|------|-----------|------|
| `--show-errors` | `-e` | 오류 인덱스 표시 |
| `--verbose` | `-v` | 상세 오류 정보 표시 (질문/정답/예측) |

### 실행 예제

```bash
# 기본 평가 (정확도만 표시)
python evaluate.py results/20241128_120000

# 오류 인덱스 표시
python evaluate.py results/20241128_120000 -e

# 상세 오류 정보 (질문 미리보기 포함)
python evaluate.py results/20241128_120000 -e -v

# 현재 디렉토리 평가
python evaluate.py . -e
```

### 출력 예제

```
==============================
DbQA / gemini-2.5-pro
[BiomniEval1 Task: using task-specific evaluation logic]
Number of correct predictions: 45
Number of no answer: 2
Total predictions: 60
Accuracy: 75.00%
Incorrect predictions (indices): [3, 7, 12, 18, 25, ...]
No answer (indices): [10, 42]
```

## 🔄 실행 흐름

### 벤치마크 실행 흐름

```
1. benchmark.py 실행
   ↓
2. 데이터셋별 인스턴스 수 확인 (BiomniEval1 로드)
   ↓
3. 실행 커맨드 생성 (각 인덱스마다)
   ↓
4. commands.txt 파일에 저장
   ↓
5. GNU parallel 또는 xargs로 병렬 실행
   ↓ (각 커맨드는 benchmark_single_task.py 호출)
6. 각 작업 독립적으로 실행
   ↓
7. 결과 파일 생성
   - log_{index}.txt: 실행 로그
   - ans_{index}.json: 답변 및 메타데이터
```

### 평가 흐름

```
1. evaluate.py 실행
   ↓
2. 결과 디렉토리에서 ans_*.json 파일 탐색
   ↓
3. BiomniEval1 클래스 로드 (task-specific 평가)
   ↓
4. 각 파일의 정답/예측 비교
   ↓
5. 통계 계산 및 출력
```

## 🛠️ 고급 사용

### 병렬 처리 방식

이 스크립트는 Python의 `multiprocessing.Pool` 대신 **GNU parallel** 또는 **xargs**를 사용합니다.

**장점:**
- ✅ Hang 문제 해결 (각 작업이 독립 프로세스)
- ✅ 실시간 진행률 표시 (`--bar`)
- ✅ 실패 내성 (`--halt never`)
- ✅ 재시작 용이 (`commands.txt` 재활용)

**시스템 요구사항:**
- GNU parallel 권장 (없으면 자동으로 xargs 사용)
- Linux/Unix 환경

### GNU parallel 설치 (선택사항)

```bash
# Ubuntu/Debian
sudo apt-get install parallel

# CentOS/RHEL
sudo yum install parallel

# macOS
brew install parallel
```

### 실패한 작업만 재실행

```bash
# 1. 실행 (일부 실패)
python benchmark.py -d DbQA -n 8

# 2. 결과 확인
python evaluate.py results/20241128_120000 -e

# 3. 실패한 인덱스만 재실행
python benchmark.py -d DbQA -i "3,7,12,18" -f 20241128_120000
```

### commands.txt 수동 편집

```bash
# 1. 커맨드 파일 확인
cat results/20241128_120000/commands.txt

# 2. 필요한 커맨드만 남기고 편집
vim results/20241128_120000/commands.txt

# 3. 수동 실행
parallel --jobs 8 --bar < results/20241128_120000/commands.txt
```

## 📝 출력 파일 구조

```
results/
└── 20241128_120000/
    ├── commands.txt          # 실행된 모든 커맨드
    ├── DbQA/
    │   ├── log_0.txt        # 실행 로그
    │   ├── ans_0.json       # 답변 + 메타데이터
    │   ├── log_1.txt
    │   ├── ans_1.json
    │   └── ...
    ├── SeqQA/
    │   └── ...
    └── HLE/
        └── ...
```

### ans_*.json 파일 형식

```json
{
  "index": 0,
  "dataset": "DbQA",
  "llm": "gemini-2.5-pro",
  "question": "What is the function of BRCA1?",
  "choices": null,
  "correct_answer": "DNA repair",
  "predicted_answer": "DNA repair",
  "full_output": "..."
}
```

## 🔍 문제 해결

### 문제: GNU parallel not found

**해결:**
- xargs가 자동으로 사용됩니다 (기능 동일)
- 또는 GNU parallel 설치

### 문제: 일부 작업이 실패

**해결:**
```bash
# --skip-existing 옵션으로 성공한 작업 건너뛰고 재실행
python benchmark.py -d DbQA -s
```

### 문제: 메모리 부족

**해결:**
```bash
# 병렬 작업 수 줄이기
python benchmark.py -d DbQA -n 2
```

### 문제: Hang 발생

**원인:**
- 이 새로운 버전에서는 해결되었습니다 (subprocess 기반)

**확인:**
```bash
# 실행 중인 프로세스 확인
ps aux | grep benchmark_single_task

# 진행 상황 확인 (GNU parallel 사용 시)
# 자동으로 진행률 바 표시됨
```

## 📚 참고

### BiomniEval1 평가 로직

각 Task는 고유한 평가 로직을 사용합니다:

- **문자 매칭**: `crispr_delivery`, `hle`, `lab_bench_*` (대소문자 무시)
- **유전자명 매칭**: `gwas_causal_gene_*`, `screen_gene_retrieval` (대문자 변환)
- **변이 정확 매칭**: `gwas_variant_prioritization` (정확히 일치)
- **JSON 비교**: `rare_disease_diagnosis` (OMIM_ID), `patient_gene_detection` (causal_gene 교집합)

상세 로직은 `biomni_eval1.py`의 `_compute_reward` 메서드 참조

## 💡 팁

1. **개발 중**: 작은 인덱스로 먼저 테스트
   ```bash
   python benchmark.py -d DbQA -i "0-5" -n 2
   ```

2. **대규모 실행**: 백그라운드 실행 + 로그 저장
   ```bash
   nohup python benchmark.py -d all -n 16 > benchmark.log 2>&1 &
   ```

3. **결과 비교**: 폴더명 활용
   ```bash
   python benchmark.py -d DbQA -f gemini_flash -l gemini-2.5-flash
   python benchmark.py -d DbQA -f gemini_pro -l gemini-2.5-pro
   python evaluate.py results/gemini_flash
   python evaluate.py results/gemini_pro
   ```

4. **빠른 디버깅**: 순차 실행으로 에러 메시지 확인
   ```bash
   python benchmark.py -d DbQA -i "0" -n 1
   ```

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

