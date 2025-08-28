# 실행 명령어

```shell
uv run main.py
```

# config.yaml

```yaml
opts: # 아래 옵션들 선택지 (구현되지 않은 항목도 있음)
  llms: [ "openai", "anthropic", "upstage", "custom", "mock" ]
  datasets: [ "case_docs", "default", "dataframe" ]
  scenarios: [ "simple", "comparison", "demo" ]
  retrievers: [ "naive", "legalQA", "frequency", "bm25", "dense", "hybrid" ]
  prompts: [ "qa_legal", "qa_lega_free" ]

defaults:
  - _self_
  - llm: upstage                #  챗봇으로 전달되는 메시지를 바로 보고 싶으면 mock (==echo)
  - embedder: jinav3            # 차후 vector_store build에 전달할 목적으로 구현만 되어 있고 설계 연결 안됨
  - retriever: naive            # naive: 빈도 기반, dense: faiss를 활용한 임베딩 기반
  - dataset: case_docs
  - prompt: qa_legal_free       # qa_legal: 판례 기반, qa_legal_free: 법률 전반 (multi-turn 질의용)
  - evaluation: langsmith       # 불필요
  - evaluation/metrics: metrics # 개선 정량화 평가 지표 (미구현)

data:
  top_k: 3
  chunk_size: 4000
  chunk_overlap: 400
  law_cases_dir: "data/law"
  # law_cases_dir: "data/dummy"

exp:
  sid: sess_001        # 세션을 위한 임시 변수 (멀티턴 채팅 이력용)
  scenario: simple
  temperature: 0.8
  max_tokens: 1024
  make_embedding: True
  question: "취업규칙을 근로자에게 불리하게 변경할 때 사용자가 지켜야 할 법적 요건은 무엇인가요?"
  #  question: "외계인"
  questions: [
    "안녕하세요. 당신은 법률 관련 답변을 할 수 있나요?",
    "어떤 사건을 위해 선임된 변호사는 판결 결과에 대한 책임이 없나요?",
    "직전에 제가 뭐라고 질문을 했었죠?",
    "아까 질문에서 형사적인 책임은 없더라도 패소하였다면 변호사 비용은 일부만 부담하는 것이 맞지 않나요?",
    "위에서 논의한 내용을 영어로 정리해주세요"
  ]

langsmith:
  enabled: false         # 이것을 켜고 langsmith_key .env에 있어야 trace 활성화 됨
  project: "langchain benchmark"
  metadata:
    ver: "2025.08.27"    # 아무 의미 없음

```

# upstage 서버 환경에서 faiss-gpu

```shell
# Install with fixed CUDA 12.1 (requires NVIDIA Driver ≥R530)
pip install 'faiss-gpu-cu12[fix-cuda]'
```

## GPU 기반 인덱스 체크

```python
import faiss

print(f"FAISS version: {faiss.__version__}")

try:
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, 128)  # Example for a 128-dimension index
    print("FAISS GPU index created successfully.")
except Exception as e:
    print(f"Error creating FAISS GPU index: {e}")
```

# Streamlit demo.py

```shell
uv run streamlit run demo.py --server.port=30399
```