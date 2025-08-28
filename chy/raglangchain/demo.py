import time

import streamlit as st

from rag.pipeline.factory import load_pipeline_with_opts
from rag.utils.logger import get_logger

logger = get_logger(__name__)


def get_session_id():
    if "current_session_id" in st.session_state:
        return st.session_state.current_session_id
    return ""


st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("⚖️ 법률 RAG 챗봇 데모")
st.subheader(f"세션: [{get_session_id()}]")


# -------------------------
# 세션 상태 초기화
# -------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user|assistant","content":str}]
if "waiting_for_answer" not in st.session_state:
    st.session_state.waiting_for_answer = False
if "current_cfg_overrides" not in st.session_state:
    st.session_state.current_cfg_overrides = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = "sid_01"

# -------------------------
# 사이드바: Hydra 파라미터 UI
# -------------------------
with st.sidebar:
    st.header("설정")

    scenario = st.selectbox("scenario (exp.scenario)", ["simple", "demo", "comparison"], index=0)
    sid = st.text_input("세션 ID (exp.sid)", value=st.session_state.current_session_id)
    llm_name = st.selectbox("LLM (llm)", ["mock", "openai", "upstage", "anthropic"], index=0)
    retriever_name = st.selectbox("Retriever ", ["naive", "dense"], index=0)
    prompt_name = st.selectbox("Prompt", ["qa_legal_free", "qa_legal"], index=0)
    make_embedding = st.checkbox("문서 임베딩 생성", value=False)

    temperature = st.slider("Temperature (exp.temperature)", 0.0, 2.5, 0.8, 0.1)
    max_tokens = st.number_input("Max tokens (exp.max_tokens)", min_value=128, max_value=8192, value=1024, step=64)
    top_k = st.number_input("Top-K", min_value=1, max_value=50, value=3, step=1)
    chunk_size = st.number_input("chunk size", min_value=100, max_value=5000, value=2000, step=1)
    chunk_overlap = st.number_input("chunk_overlap", min_value=10, max_value=2500, value=200, step=1)

    # LangSmith 옵션
    langsmith_enabled = st.checkbox("LangSmith enabled (langsmith.enabled)", value=False)
    langsmith_project = st.text_input("LangSmith project", value="langchain benchmark")

    clear_on_recreate = st.checkbox("파이프라인 재생성 시 대화 초기화", value=True)

    # TODO: hydra-zen으로 변경 (yaml이 아닌 파이썬 코드로 설정)
    if st.button("파이프라인 생성", type="primary"):
        overrides = [
            # exp 섹션
            f"exp.scenario={scenario}",
            f"exp.sid={sid}",
            f"exp.temperature={float(temperature)}",
            f"exp.max_tokens={int(max_tokens)}",
            f"exp.make_embedding={make_embedding}",
            # data 섹션
            f"data.top_k={int(top_k)}",
            f"data.chunk_size={int(chunk_size)}",
            f"data.chunk_overlap={int(chunk_overlap)}",
            # langsmith 섹션
            f"langsmith.enabled={bool(langsmith_enabled)}",
            # config groups
            f"llm={llm_name}",
            # f"embedder={embedder_name}",
            f"retriever={retriever_name}",
            f"prompt={prompt_name}",
        ]

        st.session_state.current_cfg_overrides = overrides
        st.session_state.current_session_id = sid
        st.session_state.pipeline = load_pipeline_with_opts(config_dir="conf", opts=overrides)
        if clear_on_recreate:
            st.session_state.messages = []

        st.success("파이프라인이 생성되었습니다.")
        time.sleep(2)
        st.rerun()

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("질문을 입력하세요...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if st.session_state.pipeline is None:
        st.warning("먼저 사이드바에서 파이프라인을 생성해 주세요.")
        st.session_state.waiting_for_answer = False
    else:
        # TODO: 응답 중에는 챗 입력 비-활성화 처리
        with st.chat_message("assistant"):
            with st.spinner("생성 중..."):
                try:
                    answer = st.session_state.pipeline.run(prompt)
                except Exception as e:
                    answer = f"오류가 발생했습니다: {e}"
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.waiting_for_answer = False
