# LangSmith 통합 가이드

## 개요

이 프로젝트는 LangSmith를 통해 RAG 파이프라인의 모든 단계를 추적하고 모니터링할 수 있습니다.

## 설정 방법

### 1. LangSmith API 키 설정

`.env` 파일에 다음 내용을 추가:

```bash
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

### 2. 설정 파일 수정

`conf/config.yaml`에서 LangSmith 설정을 조정할 수 있습니다:

```yaml
langsmith:
  enabled: true                    # LangSmith 추적 활성화/비활성화
  project_name: "langchain-rag-project"  # 프로젝트 이름
  session_name: "rag-session"     # 세션 이름 (한국시각 타임스탬프가 자동 추가됨)
  tags: ["rag", "pdf", "qa"]      # 추적에 사용할 태그들
  tracing: true                   # 추적 활성화
```

## 추적되는 단계

LangSmith가 추적하는 RAG 파이프라인의 각 단계:

### 1. 문서 로딩 (Document_Loading)
- 입력: PDF 파일 경로
- 출력: 로드된 페이지 수

### 2. 문서 분할 (Document_Splitting)
- 입력: 청크 크기, 오버랩, 총 페이지 수
- 출력: 생성된 청크 수

### 3. 임베딩 생성 (Embedding_Generation)
- 입력: 임베딩 모델, 제공자, 청크 수
- 출력: 사용된 임베딩 모델

### 4. 벡터스토어 생성 (VectorStore_Creation)
- 입력: 벡터스토어 타입, 청크 수
- 출력: 벡터스토어 타입

### 5. 검색기 생성 (Retriever_Creation)
- 입력: 검색기 타입, 가중치, k값
- 출력: 검색기 타입

### 6. QA 체인 생성 (QA_Chain_Creation)
- 입력: LLM 제공자, 모델명, 온도값
- 출력: 체인 타입

### 7. 질문 답변 (Question_Answering)
- 입력: 질문
- 출력: 답변

## 사용법

### 기본 실행

```bash
# LangSmith가 활성화된 상태로 실행
PYTHONPATH=. python3 src/main.py
```

### LangSmith 비활성화 실행

`conf/config.yaml`에서 `langsmith.enabled: false`로 설정

## 대시보드 확인

실행 완료 후 콘솔에 표시되는 LangSmith 대시보드 URL을 통해 추적 정보를 확인할 수 있습니다:

```
🔍 LangSmith 추적 정보:
   프로젝트: langchain-rag-project
   대시보드: https://smith.langchain.com/projects/p/langchain-rag-project
   세션 ID: rag-session_20231222_143052
```

## 버전 관리와의 통합

버전 관리 시스템과 LangSmith가 통합되어 있어 파일 버전 생성도 추적됩니다:

```bash
# 버전 생성 시 LangSmith에도 기록
python3 version_update.py "새로운 기능 추가"
```

## 주요 태그

- `main_pipeline`: 전체 RAG 파이프라인 실행
- `document_loading`: 문서 로딩 단계
- `document_splitting`: 문서 분할 단계
- `embedding`: 임베딩 생성 단계
- `vectorstore`: 벡터스토어 생성 단계
- `retriever`: 검색기 생성 단계
- `qa_chain`: QA 체인 생성 단계
- `qa`: 질문 답변 단계
- `inference`: 추론 단계
- `version_management`: 버전 관리 단계

## 문제해결

### LANGCHAIN_API_KEY가 설정되지 않은 경우
LangSmith 추적이 자동으로 비활성화되고 경고 메시지가 표시됩니다.

### LangSmith 서비스 연결 실패
오류가 발생해도 메인 RAG 파이프라인 실행에는 영향을 주지 않습니다.