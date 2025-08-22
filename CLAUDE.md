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