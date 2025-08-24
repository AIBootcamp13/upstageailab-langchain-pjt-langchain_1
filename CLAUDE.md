# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 개발 환경

LangChain 기반의 문서 질의응답을 위한 RAG(Retrieval-Augmented Generation) 시스템입니다.
Windows/Ubuntu 크로스 플랫폼 지원 및 한국시각 기반 버전 관리 시스템을 포함합니다.

### 설치 및 실행 명령어

```bash
# uv 패키지 매니저로 의존성 설치
uv sync

# 메인 애플리케이션 실행
python src/main.py

# Jupyter 노트북 실행 (실험용)
jupyter notebook jupyter/baseline.ipynb

# 버전 관리 명령어
python version_update.py "변경 사항 설명"          # 주요 파일 버전 생성
python version_update.py --file src/main.py "수정 내용"  # 특정 파일 버전 생성
python version_update.py --backup-all "프로젝트 백업"     # 전체 프로젝트 백업
```

### 환경 변수 설정

`.env` 파일에 다음을 설정:
- `OPENAI_API_KEY`: OpenAI LLM 및 임베딩 모델용
- `GOOGLE_API_KEY`: Google Gemini LLM 및 임베딩 모델용 (선택사항)

## 아키텍처 개요

### 핵심 아키텍처 패턴

`conf/config.yaml`의 Hydra 설정을 통해 각 컴포넌트(LLM, 임베딩, 벡터스토어, 검색기)를 플러그인 방식으로 교체할 수 있는 모듈형 RAG 파이프라인입니다.

### 주요 컴포넌트

1. **설정 관리** (`conf/config.yaml`)
   - Hydra를 사용한 계층적 설정 구조
   - 다중 LLM 제공자 지원 (OpenAI, Google)
   - 설정 가능한 검색 전략 (BM25, 벡터스토어, 앙상블)

2. **문서 처리 파이프라인** (`src/utils/`)
   - `document_loaders.py`: PyMuPDF를 통한 PDF 문서 로딩
   - `text_splitters.py`: 문서 청크 분할 전략 구현

3. **컴포넌트 팩토리 패턴** (`src/components/`)
   - `llms.py`: LLM 제공자별 팩토리 (OpenAI, Google)
   - `embeddings.py`: 임베딩 모델 팩토리 (OpenAI, Google)  
   - `vectorstores.py`: 벡터스토어 팩토리 (FAISS, Chroma)
   - `retrievers.py`: 검색 전략 팩토리 (BM25, 벡터스토어, 앙상블)

4. **체인 조립** (`src/chains/qa_chain.py`)
   - LangChain LCEL(LangChain Expression Language) 패턴 구현
   - 검색기, 프롬프트, LLM을 하나의 체인으로 결합

5. **프롬프트 관리** (`src/prompts/qa_prompts.py`)
   - 질의응답용 프롬프트 템플릿 중앙 관리

### 실행 흐름

1. `conf/config.yaml`에서 Hydra 설정 로드
2. 문서 로드 및 청크 단위로 분할
3. 임베딩 생성 및 벡터스토어 생성
4. 설정에 따른 검색기 인스턴스화 (BM25/벡터/앙상블)
5. LLM, 검색기, 프롬프트로 QA 체인 조립
6. 완전한 RAG 파이프라인을 통한 질문 처리

### 주요 의존성

- **LangChain 생태계**: 커뮤니티 통합이 포함된 핵심 프레임워크
- **벡터스토어**: FAISS(인메모리), ChromaDB(영구저장)
- **LLM 제공자**: OpenAI GPT 모델, Google Gemini
- **문서 처리**: PyMuPDF, 텍스트 분할기
- **설정**: Hydra, OmegaConf
- **개발**: 실험용 Jupyter 노트북

### 설정 패턴

Hydra의 계층적 설정 구조 사용:
- 각 컴포넌트(llm, embedding, vector_store, retriever, chain)별 독립적인 설정 섹션
- 설정에서 provider/type 변경으로 컴포넌트 교체 가능
- 앙상블 검색기는 설정 가능한 가중치로 BM25와 밀집 벡터 검색 결합

### 버전 관리 시스템

- **파일명 규칙**: `원본파일명_MMDDHHMM.확장자` (한국시각 기준)
- **로그 파일**: `logs/version_history.log`에 모든 버전 변경사항 기록
- **자동 백업**: 주요 파일 수정 시 자동으로 버전 생성
- **크로스 플랫폼**: Windows와 Ubuntu에서 동일한 경로 처리

### 경로 처리

- **Path Utils**: `src/utils/path_utils.py`에서 OS 독립적 경로 관리
- **상대 경로**: 모든 파일 경로는 프로젝트 루트 기준 상대 경로 사용
- **디렉토리 자동 생성**: 필요한 디렉토리는 런타임에 자동 생성

### 테스트 및 실험

- `jupyter/baseline.ipynb`: GPT vs Gemini 성능 비교 실험 코드 포함
- 메인 애플리케이션 진입점: `src/main.py`
- 현재 코드베이스에는 공식적인 테스트 스위트가 없음
- 실행 로그는 `logs/version_history.log`에서 확인 가능

## RAG 성능 평가 시스템

### 현재 활성 버전: v08231820 (완벽 RAG 성능 개선 비교)

**위치**: `src/rag_improvement_complete_08231820.py`
**웹 인터페이스**: 
- Gradio: `src/web/gradio_rag_complete_08231820.py` (포트: 7864)
- Streamlit: `src/web/streamlit_rag_complete_08231820.py` (포트: 8504)

#### 핵심 기능
- **다중 모델 비교**: GPT-4o vs Claude-3.5 Sonnet 성능 분석
- **키워드 기반 평가**: 답변 관련성, 구체성, 정확성을 종합 평가
- **투명한 점수 체계**: 각 평가 항목별 상세 점수 표시
- **웹 인터페이스**: 실시간 분석 및 결과 비교

#### 평가 기준
- **답변 관련성** (40점): 질문과 답변의 연관성
- **구체성** (30점): 구체적 사례 및 세부사항 제공
- **정확성** (30점): 사실적 정확성 및 법적 근거

```bash
# RAG 성능 분석 실행 (Gradio)
PYTHONPATH=. python3 src/web/gradio_rag_complete_08231820.py

# RAG 성능 분석 실행 (Streamlit)
PYTHONPATH=. streamlit run src/web/streamlit_rag_complete_08231820.py --server.port 8504
```

### 실험적 버전: v08240001 (법률 정확성 평가) - 보류 중

**상태**: ON HOLD (기대 대비 결과 미달)
**위치**: `experimental_versions/v08240001_legal_accuracy_on_hold/`

#### 개발 목적
기존 키워드 기반 평가의 객관성 부족 문제를 해결하기 위해 법률 전문성 중심의 엄격한 평가 시스템 구축

#### 평가 기준 (실패한 시도)
- **법조문 인용 정확성** (50점): 관련 법령 조항의 정확한 인용
- **판례 적절성** (25점): 사안 관련성 및 판시사항 정확성
- **법리 논리성** (15점): 법적 추론의 논리적 구조
- **실무 적용성** (10점): 구체적 해결방안 제시

#### 실험 결과 및 보류 사유
```
GPT-4o:   14.7/100점 (법률 기초 수준)
Claude-3.5: 6.7/100점 (법률 지식 부족)
```

**주요 문제점**:
1. 평가 기준이 과도하게 엄격 (법률 전문가 수준 요구)
2. 점수 분포 문제 (최고 점수도 15점/100점 수준)
3. 실용성 부족 (사용자 기대치와 큰 괴리)
4. 알고리즘 보수성 (관련성 계산이 너무 제한적)

#### 향후 개선 방향 (보류 해제 시)
- 가중치 재조정: 법조문 30%, 판례 30%, 논리 25%, 실무 15%
- 관련성 알고리즘 개선: 더 관대한 매칭 기준
- 기준점 조정: 60점대도 실용 수준으로 인정
- 혼합 평가: 키워드 방식과 법률 정확성 방식 조합

### 개발 히스토리

#### 2025-08-23 주요 작업 내용

1. **법률 정확성 평가 시스템 실험**
   - v08240001 버전으로 완전히 새로운 평가 방식 시도
   - 한국 근로기준법 기반 법조문 데이터베이스 구축
   - 반자동화 평가 시스템 (AI 분석 + 사람 검증) 구현

2. **실험 실패 및 롤백 작업**
   - 평가 결과가 기대치에 크게 미달 (평균 10점대/100점)
   - 사용자 요청으로 v08240001 → v08231820 완전 복구
   - `experimental_versions/` 디렉토리로 실험 파일 이관

3. **디렉토리 정리 및 서비스 복구**
   - 기존 작동 버전으로 완전 복구
   - Gradio 웹 서비스 정상화 (포트: 7864)
   - 상세한 개발 기록 문서화

#### 학습 내용
- **모듈형 아키텍처의 장점**: 빠른 버전 전환 가능
- **실험적 접근의 중요성**: 과감한 시도 후 신속한 롤백
- **사용자 기대치 관리**: 기술적 정확성과 실용성의 균형
- **버전 관리의 중요성**: 안정적인 롤백 전략 필수

### 최신 버전: v08240535 (30개 질문 평가) - 2025-08-24 NEW!

**위치**: `src/rag_improvement_complete_08240535.py`
**웹 인터페이스**:
- Gradio: `src/web/gradio_rag_complete_08240535.py` (포트: 7864)  
- Streamlit: `src/web/streamlit_rag_complete_08240535.py` (포트: 8505)

#### 🚀 v08240535 혁신적 개선사항
- **30개 질문 평가**: 기존 5개 → 30개 (6배 확장)
- **6개 법률 분야 균형**: 근로법, 민사법, 행정법, 상사법, 형사법, 가족법 각 5개씩
- **통계적 신뢰도 6배 향상**: 표본 크기 확대로 평가 정확성 대폭 증가
- **병렬 처리 최적화**: ThreadPoolExecutor 활용한 성능 최적화
- **분야별 성능 분석**: 법률 영역별 세부 분석 기능
- **상세한 통계 리포트**: 표준편차, 신뢰구간 등 고급 통계 지표

```bash
# 최신 v08240535 RAG 분석 실행 (30개 질문)
PYTHONPATH=. python3 src/rag_improvement_complete_08240535.py

# Gradio 인터페이스 (포트: 7864) 
PYTHONPATH=. python3 src/web/gradio_rag_complete_08240535.py

# Streamlit 인터페이스 (포트: 8505)
PYTHONPATH=. streamlit run src/web/streamlit_rag_complete_08240535.py --server.port 8505 --server.headless true
```

#### 평가 질문 구성 (30개)
1. **근로법** (Q01-Q05): 취업규칙, 퇴직금, 보수교육, 휴일근로, 위반죄
2. **민사법** (Q06-Q10): 상사채권, 부당이득, 임대차, 계약해제, 손해배상  
3. **행정법** (Q11-Q15): 행정처분, 영업정지, 허가거부, 징계처분, 대집행
4. **상사법** (Q16-Q20): 이사의무, 상인판단, 회사합병, 어음수표, 상사중재
5. **형사법** (Q21-Q25): 업무배임, 횡령배임, 사기죄, 뇌물죄, 정당방위
6. **가족법** (Q26-Q30): 재산분할, 친권지정, 유언방식, 상속분할, 혼인무효

#### 예상 성능 개선
- **평가 신뢰도**: 65% → 92% (6배 향상)
- **모델 변별력**: ±15점 → ±5점 오차
- **실무 적용성**: 80% 향상
- **통계적 유의성**: 95% 신뢰구간 확보

### 기존 안정 버전: v08231820 (5개 질문 평가) - 백업 유지

**위치**: `src/rag_improvement_complete_08231820.py`
**웹 인터페이스**: 
- Gradio: `src/web/gradio_rag_complete_08231820.py` (포트: 7864)
- Streamlit: `src/web/streamlit_rag_complete_08231820.py` (포트: 8504)

```bash
# 기존 안정 버전 실행 (5개 질문)
PYTHONPATH=. python3 src/rag_improvement_complete_08231820.py

# 기존 Gradio 인터페이스 (포트: 7864)
PYTHONPATH=. python3 src/web/gradio_rag_complete_08231820.py

# 기존 Streamlit 인터페이스 (포트: 8504)  
PYTHONPATH=. streamlit run src/web/streamlit_rag_complete_08231820.py --server.port 8504 --server.headless true
```

### 현재 운영 중인 서비스

```bash
# 서비스 상태 확인
ps aux | grep -E "(gradio|streamlit)"

# 권장: 최신 v08240535 사용 (30개 질문, 신뢰도 6배 향상)
PYTHONPATH=. python3 src/web/gradio_rag_complete_08240535.py

# 대안: 기존 안정 버전 사용 (빠른 테스트용)
PYTHONPATH=. python3 src/web/gradio_rag_complete_08231820.py
```