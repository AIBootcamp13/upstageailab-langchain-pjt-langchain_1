# 이전 버전 파일 아카이브

아카이브 생성일시: 2025-08-23 14:44 (한국시간)
아카이브 버전: v08231444

## 아카이브된 파일들

### 웹 인터페이스 (사용 중단)
- `gradio_app_v2.py` - 원본 Gradio 인터페이스 (AI Brief용)
- `gradio_law_app_v4.py` - 법률용 Gradio v4.0 (Import 의존성 오류)
- `streamlit_app_v1.py` - 원본 Streamlit 인터페이스 (AI Brief용)
- `streamlit_law_app_v3.py` - 법률용 Streamlit v3.0 (Import 오류)

### 설정 파일 (이전 버전)
- `config_08222349.yaml` - 초기 설정 파일
- `config_08230046.yaml` - 중간 버전 설정 파일  
- `config_08230058.yaml` - 이전 버전 설정 파일

### 컴포넌트 파일 (이전 버전)
- `vectorstores_08222349.py` - 초기 벡터스토어 구현
- `vectorstores_08230046.py` - 중간 버전 벡터스토어
- `vectorstores_08230058.py` - 이전 버전 벡터스토어

### 메인 파일 (이전 버전)
- `main_08222349.py` - 초기 메인 실행 파일
- `main_08230046.py` - 중간 버전 메인 파일
- `main_08230058.py` - 이전 버전 메인 파일

### 유틸리티 (이전 버전)
- `document_loaders_08222349.py` - 초기 문서 로더
- `document_loaders_08230046.py` - 중간 버전 문서 로더
- `document_loaders_08230058.py` - 이전 버전 문서 로더

### 결과 파일 (이전 분석)
- `model_comparison_08230049.json` - 이전 모델 비교 결과
- `model_comparison_report_08230049.md` - 이전 모델 비교 보고서

## 현재 활성 파일들

### 웹 인터페이스
- `src/web/streamlit_law_simple_v6.py` - Streamlit 법률 인터페이스 (포트 8503)
- `src/web/gradio_law_simple_v5.py` - Gradio 법률 인터페이스 (포트 7863)

### 핵심 시스템
- `src/rag_improvement_comparison_08231426.py` - RAG 성능 개선 비교 시스템
- `src/law_model_compare_main.py` - 법률 도메인 모델 비교 (미완성)

### 최신 결과
- `results/rag_improvement_comparison/rag_improvement_results_*.json`
- `results/rag_improvement_comparison/rag_improvement_report_*.md`

## 정리 사유

1. **웹 인터페이스**: Import 오류나 의존성 문제로 사용 불가능
2. **타임스탬프 파일들**: 이전 개발 과정에서 생성된 중간 버전들
3. **결과 파일들**: 새로운 RAG 개선 분석으로 대체됨

## 복구 방법

필요시 이 디렉토리에서 원본 위치로 파일을 복사하여 복구 가능합니다.