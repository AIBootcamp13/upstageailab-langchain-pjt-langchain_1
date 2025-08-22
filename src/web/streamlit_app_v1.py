import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import sys
import os
from datetime import datetime
import pytz

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.version_manager import VersionManager
from src.utils.langsmith_simple import LangSmithSimple
from src.components.llms import get_llm
from src.components.embeddings import get_embedding_model
from src.components.vectorstores import get_vector_store
from src.components.retrievers import get_retriever
from src.utils.document_loaders import load_documents
from src.utils.text_splitters import split_documents
from src.prompts.qa_prompts import get_qa_prompt
from src.chains.qa_chain import get_qa_chain
from omegaconf import OmegaConf
from dotenv import load_dotenv

# 페이지 설정
st.set_page_config(
    page_title="🤖 LLM 모델 비교 시스템 v1.0 (Streamlit)",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .model-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_rag_pipeline():
    """RAG 파이프라인 초기화 (캐싱)"""
    load_dotenv()
    
    # 기본 설정
    cfg = OmegaConf.create({
        "data": {
            "path": "data/SPRI_AI_Brief_2023년12월호_F.pdf",
            "chunk_size": 1000,
            "chunk_overlap": 50
        },
        "embedding": {
            "provider": "openai",
            "model_name": "text-embedding-ada-002"
        },
        "vector_store": {
            "type": "faiss",
            "persist_directory": "faiss_db"
        },
        "retriever": {
            "type": "ensemble",
            "weights": [0.5, 0.5]
        },
        "chain": {
            "retriever_k": 4
        },
        "langsmith": {
            "enabled": True,
            "project_name": "langchain-rag-project-streamlit",
            "session_name": "streamlit-session",
            "tags": ["streamlit", "web", "comparison"]
        }
    })
    
    try:
        with st.spinner("📚 문서 로딩 및 RAG 파이프라인 초기화 중..."):
            documents = load_documents(cfg)
            split_documents_list = split_documents(cfg, documents)
            embeddings = get_embedding_model(cfg)
            vectorstore = get_vector_store(cfg, split_documents_list, embeddings)
            
            if cfg.retriever.type == "bm25" or cfg.retriever.type == "ensemble":
                retriever = get_retriever(cfg, vectorstore, documents=split_documents_list)
            else:
                retriever = get_retriever(cfg, vectorstore)
            
            prompt = get_qa_prompt()
            
        return cfg, retriever, prompt, len(documents), len(split_documents_list)
    except Exception as e:
        st.error(f"RAG 파이프라인 초기화 실패: {e}")
        return None, None, None, 0, 0

def run_model_comparison(question, model_configs, retriever, prompt):
    """모델 비교 실행"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, model_config in enumerate(model_configs):
        status_text.text(f"🤖 {model_config['name']} 테스트 중...")
        progress = (i + 1) / len(model_configs)
        progress_bar.progress(progress)
        
        start_time = time.time()
        
        try:
            # 모델별 설정
            model_cfg = OmegaConf.create({
                "llm": {
                    "provider": model_config["provider"],
                    "model_name": model_config["model_name"], 
                    "temperature": model_config.get("temperature", 0.7)
                }
            })
            
            llm = get_llm(model_cfg)
            qa_chain = get_qa_chain(llm, retriever, prompt)
            response = qa_chain.invoke(question)
            execution_time = time.time() - start_time
            
            result = {
                "model": model_config['name'],
                "provider": model_config['provider'],
                "response": response,
                "execution_time": execution_time,
                "response_length": len(str(response)),
                "word_count": len(str(response).split()),
                "success": True,
                "error": None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = {
                "model": model_config['name'],
                "provider": model_config['provider'],
                "response": f"오류 발생: {str(e)}",
                "execution_time": execution_time,
                "response_length": 0,
                "word_count": 0,
                "success": False,
                "error": str(e)
            }
        
        results.append(result)
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def main():
    # 메인 헤더
    st.markdown('<h1 class="main-header">🤖 LLM 모델 비교 시스템 v1.0</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Streamlit으로 구현된 실시간 RAG 모델 성능 비교</p>', unsafe_allow_html=True)
    
    # 사이드바 설정
    st.sidebar.title("⚙️ 설정")
    st.sidebar.markdown("---")
    
    # RAG 파이프라인 로드
    pipeline_data = load_rag_pipeline()
    if pipeline_data[0] is None:
        st.error("RAG 파이프라인 초기화에 실패했습니다. API 키를 확인해주세요.")
        st.stop()
    
    cfg, retriever, prompt, doc_count, chunk_count = pipeline_data
    
    # 파이프라인 정보 표시
    st.sidebar.success("✅ RAG 파이프라인 준비 완료")
    st.sidebar.metric("📄 문서 페이지", f"{doc_count}개")
    st.sidebar.metric("📊 문서 청크", f"{chunk_count}개")
    
    st.sidebar.markdown("---")
    
    # 모델 선택
    st.sidebar.subheader("🤖 비교 모델 선택")
    
    available_models = {
        "GPT-4o": {"provider": "openai", "model_name": "gpt-4o"},
        "Claude-3.5-Haiku": {"provider": "anthropic", "model_name": "claude-3-5-haiku-20241022"},
    }
    
    selected_models = st.sidebar.multiselect(
        "모델을 선택하세요",
        list(available_models.keys()),
        default=list(available_models.keys()),
        help="비교할 모델들을 선택하세요"
    )
    
    if len(selected_models) < 2:
        st.warning("⚠️ 비교를 위해 최소 2개 모델을 선택해주세요.")
        st.stop()
    
    # 온도 설정
    temperature = st.sidebar.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="응답의 창의성을 조절합니다"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 질문을 입력하고 '비교 시작' 버튼을 눌러주세요!")
    
    # 메인 컨텐츠
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 질문 입력")
        
        # 미리 정의된 질문들
        sample_questions = [
            "미국 바이든 대통령이 몇년 몇월 몇일에 연방정부 차원에서 안전하고 신뢰할 수 있는 AI 개발과 사용을 보장하기 위한 행정명령을 발표했나요?",
            "AI 안전성 정상회의에 참가한 28개국들이 AI 안전 보장을 위한 협력 방안을 담은 블레츨리 선언을 발표한 나라는 어디인가요?",
            "구글이 앤스로픽에 투자한 금액은 총 얼마인가요?",
            "삼성전자가 자체 개발한 생성 AI 모델의 이름은 무엇인가요?",
        ]
        
        selected_question = st.selectbox(
            "📋 샘플 질문 선택",
            ["직접 입력"] + sample_questions,
            help="미리 준비된 질문을 선택하거나 직접 입력하세요"
        )
        
        if selected_question == "직접 입력":
            question = st.text_area(
                "질문을 입력하세요",
                height=100,
                placeholder="RAG 시스템에 물어볼 질문을 입력하세요..."
            )
        else:
            question = selected_question
            st.text_area("선택된 질문", value=question, height=100, disabled=True)
    
    with col2:
        st.subheader("🎯 실행 상태")
        
        if st.button("🚀 비교 시작", type="primary", use_container_width=True):
            if not question.strip():
                st.error("❌ 질문을 입력해주세요!")
                st.stop()
            
            # 모델 설정 생성
            model_configs = []
            for model_name in selected_models:
                model_info = available_models[model_name].copy()
                model_info["name"] = model_name
                model_info["temperature"] = temperature
                model_configs.append(model_info)
            
            # 비교 실행
            st.markdown("### 🔄 모델 비교 진행 중...")
            results = run_model_comparison(question, model_configs, retriever, prompt)
            
            # 세션 스테이트에 결과 저장
            st.session_state['comparison_results'] = results
            st.session_state['question'] = question
            st.session_state['timestamp'] = datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S")
    
    # 결과 표시
    if 'comparison_results' in st.session_state:
        st.markdown("---")
        st.markdown("## 📊 비교 결과")
        
        results = st.session_state['comparison_results']
        question = st.session_state['question']
        timestamp = st.session_state['timestamp']
        
        st.info(f"🕒 실행 시간: {timestamp}")
        st.markdown(f"**❓ 질문:** {question}")
        
        # 성능 메트릭 차트
        successful_results = [r for r in results if r['success']]
        
        if len(successful_results) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                # 응답 시간 비교 차트
                fig_time = px.bar(
                    x=[r['model'] for r in successful_results],
                    y=[r['execution_time'] for r in successful_results],
                    title="⏱️ 응답 시간 비교 (초)",
                    color=[r['model'] for r in successful_results],
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig_time.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_time, use_container_width=True)
            
            with col2:
                # 응답 길이 비교 차트
                fig_length = px.bar(
                    x=[r['model'] for r in successful_results],
                    y=[r['response_length'] for r in successful_results],
                    title="📏 응답 길이 비교 (글자수)",
                    color=[r['model'] for r in successful_results],
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_length.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_length, use_container_width=True)
        
        # 상세 결과
        st.markdown("### 📝 상세 응답 결과")
        
        for i, result in enumerate(results):
            with st.expander(f"🤖 {result['model']} - {'✅ 성공' if result['success'] else '❌ 실패'}", expanded=True):
                if result['success']:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("⏱️ 실행시간", f"{result['execution_time']:.2f}초")
                    with col2:
                        st.metric("📏 응답길이", f"{result['response_length']}자")
                    with col3:
                        st.metric("📊 단어수", f"{result['word_count']}개")
                    
                    st.markdown("**📝 응답 내용:**")
                    st.markdown(f"> {result['response']}")
                    
                else:
                    st.error(f"❌ 오류: {result['error']}")
        
        # 성능 비교 분석
        if len(successful_results) >= 2:
            st.markdown("### 🏆 성능 분석")
            
            fastest = min(successful_results, key=lambda x: x['execution_time'])
            slowest = max(successful_results, key=lambda x: x['execution_time'])
            longest = max(successful_results, key=lambda x: x['response_length'])
            shortest = min(successful_results, key=lambda x: x['response_length'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="success-message">
                    <h4>⚡ 속도 우승자</h4>
                    <p><strong>{fastest['model']}</strong>가 가장 빠름</p>
                    <p>📊 {fastest['execution_time']:.2f}초 vs {slowest['execution_time']:.2f}초</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="success-message">
                    <h4>📏 상세함 우승자</h4>
                    <p><strong>{longest['model']}</strong>가 가장 상세함</p>
                    <p>📊 {longest['response_length']}자 vs {shortest['response_length']}자</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()