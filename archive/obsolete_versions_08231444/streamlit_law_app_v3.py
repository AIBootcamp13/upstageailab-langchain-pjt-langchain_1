#!/usr/bin/env python3
"""
법률 도메인 Streamlit 웹 인터페이스 v3.0
JSON 판례 데이터 기반 RAG 시스템으로 법률 관련 질문 답변 및 모델 비교
"""

import os
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from omegaconf import OmegaConf

# 프로젝트 루트 디렉토리를 Python path에 추가
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.law_model_compare_main import LawDocumentLoader, create_law_rag_pipeline
from src.components.llms import create_llm
from src.chains.qa_chain import create_qa_chain
from src.prompts.qa_prompts import get_qa_prompt
from src.utils.langsmith_simple import LangSmithManager

# 페이지 설정
st.set_page_config(
    page_title="⚖️ 법률 AI 분석 시스템 v3.0",
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2c3e50, #3498db);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .law-question-box {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
    }
    .model-comparison-box {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_law_rag_pipeline():
    """법률 RAG 파이프라인 초기화 (캐싱)"""
    load_dotenv()
    
    # 기본 설정
    cfg = OmegaConf.create({
        'embedding': {
            'provider': 'openai',
            'model_name': 'text-embedding-3-small',
            'chunk_size': 1000
        },
        'vector_store': {
            'type': 'faiss',
            'persist_directory': 'faiss_db_law'
        },
        'retriever': {
            'search_type': 'similarity', 
            'search_kwargs': {'k': 5}
        },
        'llm': {
            'temperature': 0.1,
            'max_tokens': 1000
        },
        'langsmith': {
            'enabled': True,
            'project_name': 'law-streamlit-v3',
            'session_name': f'law-streamlit-session_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    })
    
    try:
        retriever = create_law_rag_pipeline(cfg)
        langsmith_manager = LangSmithManager(cfg.langsmith)
        return retriever, cfg, langsmith_manager, None
    except Exception as e:
        return None, None, None, str(e)

def get_law_model_response(retriever, model_name, provider, model_id, question, temperature, cfg, langsmith_manager):
    """법률 모델 응답 생성"""
    
    # LLM 설정
    llm_cfg = OmegaConf.create({
        'llm': {
            'provider': provider,
            'model_name': model_id, 
            'temperature': temperature
        }
    })
    
    start_time = time.time()
    
    try:
        # LLM 생성
        llm = create_llm(llm_cfg)
        
        # QA 체인 생성
        qa_chain = create_qa_chain(retriever, llm, get_qa_prompt())
        
        # LangSmith 추적
        if langsmith_manager and langsmith_manager.enabled:
            run_id = langsmith_manager.start_run(
                name=f"Law_Streamlit_{model_name}",
                inputs={"question": question}
            )
        
        # 질문 실행
        response = qa_chain.invoke({"question": question})
        answer = response['answer'] if isinstance(response, dict) else str(response)
        
        response_time = time.time() - start_time
        
        if langsmith_manager and langsmith_manager.enabled:
            langsmith_manager.end_run(run_id, outputs={"answer": answer})
        
        return {
            'success': True,
            'answer': answer,
            'response_time': response_time
        }
        
    except Exception as e:
        response_time = time.time() - start_time
        error_msg = f"오류: {str(e)}"
        
        if langsmith_manager and langsmith_manager.enabled:
            langsmith_manager.end_run(run_id, outputs={"error": error_msg})
        
        return {
            'success': False,
            'answer': error_msg,
            'response_time': response_time
        }

def create_response_time_chart(results):
    """응답 시간 비교 차트 생성"""
    models = list(results.keys())
    times = [results[model]['response_time'] for model in models]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=times,
            marker_color=['#3498db', '#e74c3c'],
            text=[f'{t:.2f}s' for t in times],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="모델별 응답 시간 비교",
        xaxis_title="모델",
        yaxis_title="응답 시간 (초)",
        height=400
    )
    
    return fig

def main():
    """메인 애플리케이션"""
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ 법률 AI 분석 시스템 v3.0</h1>
        <p>대법원 판례 기반 RAG 시스템 • GPT-4o vs Claude-3.5-Haiku 비교</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 시스템 설정")
        
        # 모델 선택
        selected_models = st.multiselect(
            "비교할 모델 선택",
            ["GPT-4o", "Claude-3.5-Haiku"],
            default=["GPT-4o", "Claude-3.5-Haiku"]
        )
        
        # 온도 설정
        temperature = st.slider(
            "Temperature (창의성 조절)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="낮을수록 일관된 답변, 높을수록 창의적 답변"
        )
        
        st.markdown("---")
        st.markdown("""
        **📚 데이터 소스**
        - 대법원 판례 JSON 데이터
        - 민사/형사 사건 포함
        - 실시간 RAG 검색
        """)
    
    # RAG 파이프라인 로드
    pipeline_data = load_law_rag_pipeline()
    if pipeline_data[0] is None:
        st.error("법률 RAG 파이프라인 초기화에 실패했습니다. API 키를 확인해주세요.")
        st.error(f"오류: {pipeline_data[3]}")
        st.stop()
    
    retriever, cfg, langsmith_manager, _ = pipeline_data
    st.success("✅ 법률 판례 데이터베이스 로드 완료")
    
    # 메인 인터페이스
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 법률 질문 입력")
        
        # 샘플 질문 선택
        sample_questions = [
            "직접 입력",
            "취업규칙을 근로자에게 불리하게 변경할 때 사용자가 지켜야 할 법적 요건은 무엇인가요?",
            "퇴직금 지급 기일을 연장하는 합의를 했더라도 연장된 기일까지 지급하지 않으면 형사처벌을 받나요?",
            "근로기준법에서 규정하는 퇴직금 지급 의무에 대해 설명해주세요.",
            "사용자가 취업규칙 변경 시 근로자의 동의를 얻지 못했을 때의 법적 효과는?",
            "퇴직급여보장법 위반 시 어떤 형사처벌을 받게 되나요?"
        ]
        
        selected_question = st.selectbox("샘플 질문 선택", sample_questions)
        
        if selected_question == "직접 입력":
            question = st.text_area(
                "법률 질문을 입력하세요",
                placeholder="예: 근로자의 퇴직금 지급과 관련된 법적 규정은 무엇인가요?",
                height=100
            )
        else:
            question = st.text_area("선택된 질문", value=selected_question, height=100)
    
    with col2:
        st.subheader("📊 시스템 상태")
        
        # 로드된 판례 정보 표시
        try:
            law_loader = LawDocumentLoader()
            documents = law_loader.load_legal_documents()
            
            st.metric("로드된 판례", f"{len(documents)}건")
            
            # 판례 정보 요약
            if documents:
                case_types = [doc['metadata'].get('case_type', 'Unknown') for doc in documents]
                case_type_counts = {case_type: case_types.count(case_type) for case_type in set(case_types)}
                
                st.write("**사건 유형별 분포**")
                for case_type, count in case_type_counts.items():
                    st.write(f"- {case_type}: {count}건")
                    
        except Exception as e:
            st.error(f"판례 정보 로드 오류: {e}")
    
    # 분석 실행
    if st.button("🔍 법률 분석 시작", type="primary", use_container_width=True):
        if not question:
            st.warning("질문을 입력해주세요.")
            return
            
        if not selected_models:
            st.warning("최소 하나의 모델을 선택해주세요.")
            return
        
        st.markdown(f"""
        <div class="law-question-box">
            <h4>📋 분석 질문</h4>
            <p>{question}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 모델별 설정
        model_configs = {
            "GPT-4o": {"provider": "openai", "model_id": "gpt-4o"},
            "Claude-3.5-Haiku": {"provider": "anthropic", "model_id": "claude-3-5-haiku-20241022"}
        }
        
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 각 모델 실행
        for i, model_name in enumerate(selected_models):
            status_text.text(f"🤖 {model_name} 분석 중...")
            progress_bar.progress((i) / len(selected_models))
            
            config = model_configs[model_name]
            result = get_law_model_response(
                retriever, model_name, config["provider"], config["model_id"], 
                question, temperature, cfg, langsmith_manager
            )
            
            results[model_name] = result
        
        progress_bar.progress(1.0)
        status_text.text("✅ 분석 완료!")
        
        # 결과 표시
        st.markdown("""
        <div class="model-comparison-box">
            <h3>🤖 모델 분석 결과</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 성능 메트릭
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_time = sum([r['response_time'] for r in results.values()]) / len(results)
            st.metric("평균 응답시간", f"{avg_time:.2f}초")
        
        with col2:
            success_count = sum([1 for r in results.values() if r['success']])
            st.metric("성공률", f"{success_count}/{len(results)}")
        
        with col3:
            total_models = len(selected_models)
            st.metric("분석 모델 수", total_models)
        
        # 응답 시간 차트
        if len(results) > 1:
            st.plotly_chart(create_response_time_chart(results), use_container_width=True)
        
        # 각 모델의 상세 응답
        for model_name, result in results.items():
            with st.expander(f"🤖 {model_name} 상세 응답 ({result['response_time']:.2f}초)", expanded=True):
                if result['success']:
                    st.success("✅ 응답 성공")
                    st.markdown("**답변:**")
                    st.write(result['answer'])
                else:
                    st.error("❌ 응답 실패")
                    st.write(result['answer'])
        
        # 결과 비교 분석
        if len(results) > 1 and all(r['success'] for r in results.values()):
            st.subheader("📊 모델 비교 분석")
            
            # 응답 길이 비교
            response_lengths = {model: len(result['answer']) for model, result in results.items()}
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**응답 길이 비교**")
                for model, length in response_lengths.items():
                    st.write(f"- {model}: {length:,} 글자")
            
            with col2:
                st.write("**응답 시간 비교**")
                for model, result in results.items():
                    st.write(f"- {model}: {result['response_time']:.2f}초")
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>⚖️ 법률 AI 분석 시스템 v3.0 | 대법원 판례 기반 RAG | Streamlit Framework</p>
        <p>🔬 Powered by LangChain • OpenAI • Anthropic • LangSmith</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()