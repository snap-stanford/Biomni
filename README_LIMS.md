# 🔬 Integrated LIMS & Analysis Platform

실험실 데이터 관리와 특화된 분석 애플리케이션을 통합한 플랫폼입니다.

## 시스템 개요

### 아키텍처
```
┌─────────────────┐    ┌──────────────────────┐
│   Laboratory    │    │  Specialized Apps    │
│   Information   │    │                      │
│   Management    │◄──►│ • OmicsHorizon      │
│   System (LIMS) │    │   (Transcriptomics)  │
│                 │    │ • Proteomics App     │
│ • Data Storage  │    │ • Metabolomics App   │
│ • File Mgmt     │    │ • ...                │
│ • Sample Mgmt   │    │                      │
└─────────────────┘    └──────────────────────┘
```

### 주요 기능

1. **통합 데이터 관리 (LIMS)**
   - 실험실 데이터 파일 중앙 집중 관리
   - 다양한 데이터 타입 지원 (.csv, .xlsx, .tsv, .gz 등)
   - 메타데이터 관리 (파일 크기, 수정일, 타입 등)

2. **특화된 분석 앱**
   - **OmicsHorizon™**: 전사체 분석 플랫폼
   - 확장 가능한 아키텍처로 미래 앱 추가 가능
   - 각 앱별 특화된 UI와 워크플로우

3. **지능적 데이터 라우팅**
   - 파일 타입 기반 자동 앱 추천
   - 데이터 호환성 검증
   - 원활한 앱 간 전환

## 설치 및 설정

### 요구사항
- Python 3.8+
- Streamlit
- 관련 분석 라이브러리들

### 파일 구조
```
Biomni/
├── streamlit/
│   ├── main_app.py          # 메인 LIMS 애플리케이션
│   └── streamlit_app.py     # OmicsHorizon 분석 앱
├── data/                    # LIMS 데이터 저장소
├── workspace/               # 분석 앱 워크스페이스
├── biomni_data/             # 분석용 참조 데이터
├── logo/                    # 로고 파일들
└── README_LIMS.md           # 이 파일
```

### 실행 방법

1. **메인 LIMS 시스템 실행**:
```bash
streamlit run streamlit/main_app.py
```

2. **개별 분석 앱 실행** (기존 방식):
```bash
streamlit run streamlit/streamlit_app.py
```

## 사용 방법

### 1. LIMS 대시보드

메인 화면에서 다음을 수행할 수 있습니다:

- **데이터 파일 목록 조회**: `data/` 디렉토리의 모든 파일 표시
- **파일 선택**: 분석에 사용할 파일들을 체크박스로 선택
- **분석 앱 선택**: 호환되는 앱들이 자동으로 활성화됨
- **앱 실행**: 선택된 데이터로 분석 앱 실행

### 2. OmicsHorizon 분석 앱

데이터 선택 후 OmicsHorizon을 실행하면:

- **데이터 업로드 및 브리핑**: 선택된 파일들을 자동으로 로드
- **논문 워크플로우 추출**: PDF에서 분석 방법 추출
- **인터랙티브 분석**: 단계별 분석 실행
- **결과 시각화**: 자동 생성된 플롯과 다운로드
- **Q&A 기능**: 분석 결과에 대한 질문 답변

### 3. 데이터 흐름

```
1. LIMS에서 데이터 파일 선택
2. 호환되는 분석 앱 선택
3. 파일들을 앱 워크스페이스로 복사
4. 분석 앱에서 특화된 UI로 분석 수행
5. 결과 저장 및 다운로드
```

## 확장 방법

### 새로운 분석 앱 추가

`streamlit/main_app.py`의 `ANALYSIS_APPS` 사전에 앱 정보를 추가:

```python
ANALYSIS_APPS = {
    'new_app': {
        'name': 'New Analysis App',
        'description': 'Description of the app',
        'icon': '🔬',
        'function': run_new_app_function,
        'data_types': ['csv', 'txt', 'xlsx'],
        'category': 'Analysis Type',
        'enabled': True
    }
}
```

### 앱 개발 가이드라인

1. **함수 기반 모듈화**: 각 앱을 단일 함수로 구현
2. **독립적 세션 상태**: 앱별 고유 키 사용
3. **표준화된 UI**: 일관된 디자인 패턴 적용
4. **데이터 호환성**: 지원하는 파일 타입 명시

## API 및 통합

### 데이터 공유 메커니즘

- `DataManager`: 파일 목록 및 복사 기능
- `AppRouter`: 앱 간 네비게이션 관리
- 세션 상태를 통한 데이터 전달

### 확장 포인트

- **데이터 소스**: 데이터베이스나 외부 API 연동 가능
- **사용자 관리**: 인증 및 권한 시스템 추가 가능
- **워크플로우**: 복잡한 분석 파이프라인 지원
- **보고서**: 자동화된 결과 보고서 생성

## 문제 해결

### 일반적인 문제들

1. **Import 에러**: Python 경로와 모듈 의존성 확인
2. **데이터 파일 없음**: `data/` 디렉토리에 파일 추가
3. **앱 실행 실패**: 해당 앱의 함수가 제대로 구현되었는지 확인
4. **메모리 부족**: 대용량 파일 처리 시 메모리 최적화

### 디버깅

- 각 앱을 개별적으로 실행하여 테스트
- 로그 파일 확인 (`logs/` 디렉토리)
- 세션 상태 초기화로 문제 해결

## 향후 개발 계획

- [ ] 추가 분석 앱 개발 (Proteomics, Metabolomics)
- [ ] 사용자 인증 및 권한 관리
- [ ] 데이터베이스 통합
- [ ] 클라우드 배포 지원
- [ ] API 엔드포인트 제공
- [ ] 배치 처리 기능

## 기여 방법

1. 새로운 분석 앱 개발
2. UI/UX 개선
3. 버그 리포트 및 수정
4. 문서화 개선

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

---

**개발자**: JHJeon
**버전**: 1.0.0
**최종 업데이트**: 2025년 10월 30일</content>
</xai:function_call">Now let me create a simple test to verify the system works. I'll create a test script that can run without streamlit to verify the core functionality.">
<xai:function_call name="write">
<parameter name="file_path">test_system.py
