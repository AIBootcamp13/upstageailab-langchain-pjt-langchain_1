#!/usr/bin/env python3
"""
법률 도메인 Streamlit 웹 인터페이스 v6.0 (단순화 버전)
JSON 판례 데이터를 읽어서 법률 질문에 직접 답변하는 시스템
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

# OpenAI 및 Anthropic 라이브러리 직접 사용
try:
    from openai import OpenAI
except ImportError:
    st.error("OpenAI 라이브러리를 설치하세요: pip install openai")

try:
    import anthropic
except ImportError:
    st.error("Anthropic 라이브러리를 설치하세요: pip install anthropic")

# 페이지 설정
st.set_page_config(
    page_title="⚖️ 법률 AI 분석 시스템 v6.0",
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

@st.cache_data
def load_law_documents():
    """법률 문서 로드 (캐싱)"""
    law_documents = []
    law_data_dir = Path("data/law")
    
    if not law_data_dir.exists():
        return [], "❌ data/law 디렉토리를 찾을 수 없습니다."
    
    json_files = list(law_data_dir.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
            
            # JSON 데이터를 텍스트로 변환
            document_text = format_legal_case(case_data)
            
            law_documents.append({
                'content': document_text,
                'metadata': {
                    'source': str(json_file),
                    'case_number': case_data.get('사건번호', ''),
                    'case_name': case_data.get('사건명', ''),
                    'court': case_data.get('법원명', ''),
                    'date': case_data.get('선고일자', ''),
                    'case_type': case_data.get('사건종류명', '')
                }
            })
            
        except Exception as e:
            st.error(f"파일 로드 오류 {json_file}: {e}")
            continue
    
    return law_documents, f"✅ 법률 판례 {len(law_documents)}건 로드 완료"

def format_legal_case(case_data: dict) -> str:
    """법률 사건 데이터를 텍스트로 포맷팅"""
    return f"""
==== 법률 판례 정보 ====
사건번호: {case_data.get('사건번호', 'N/A')}
사건명: {case_data.get('사건명', 'N/A')}
법원명: {case_data.get('법원명', 'N/A')}
선고일자: {case_data.get('선고일자', 'N/A')}
사건종류: {case_data.get('사건종류명', 'N/A')}

==== 판시사항 ====
{case_data.get('판시사항', 'N/A')}

==== 판결요지 ====
{case_data.get('판결요지', 'N/A')}

==== 참조조문 ====
{case_data.get('참조조문', 'N/A')}

==== 판례내용 ====
{case_data.get('판례내용', 'N/A')[:2000]}...
"""

@st.cache_resource
def initialize_ai_clients():
    """AI 클라이언트 초기화 (캐싱)"""
    load_dotenv()
    
    openai_client = None
    anthropic_client = None
    
    if os.getenv("OPENAI_API_KEY"):
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if os.getenv("ANTHROPIC_API_KEY"):
        anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    return openai_client, anthropic_client

def search_relevant_cases(question: str, law_documents: list, top_k: int = 3) -> list:
    """질문과 관련된 판례 검색 (간단한 키워드 매칭)"""
    if not law_documents:
        return []
    
    # 간단한 키워드 기반 검색
    question_keywords = question.lower().split()
    relevant_cases = []
    
    for doc in law_documents:
        content = doc['content'].lower()
        score = sum(1 for keyword in question_keywords if keyword in content)
        
        if score > 0:
            relevant_cases.append((doc, score))
    
    # 점수순 정렬하여 상위 k개 반환
    relevant_cases.sort(key=lambda x: x[1], reverse=True)
    return [case[0] for case in relevant_cases[:top_k]]

def get_gpt_response(question: str, context: str, temperature: float, openai_client) -> dict:
    """GPT-4o 응답 생성"""
    if not openai_client:
        return {
            'success': False,
            'answer': "OpenAI API 키가 설정되지 않았습니다.",
            'response_time': 0
        }
    
    start_time = time.time()
    
    try:
        system_prompt = """당신은 대한민국의 전문 법률 AI입니다. 주어진 법률 판례 정보를 바탕으로 질문에 대해 정확하고 신뢰할 수 있는 답변을 제공하세요. 
답변할 때는 관련 판례의 사건번호와 주요 내용을 인용하여 근거를 명확히 제시하세요."""
        
        user_prompt = f"""
법률 질문: {question}

관련 판례 정보:
{context}

위의 판례 정보를 참고하여 질문에 대해 상세히 답변해주세요.
"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        response_time = time.time() - start_time
        
        return {
            'success': True,
            'answer': answer,
            'response_time': response_time
        }
        
    except Exception as e:
        response_time = time.time() - start_time
        return {
            'success': False,
            'answer': f"GPT-4o 오류: {str(e)}",
            'response_time': response_time
        }

def get_claude_response(question: str, context: str, temperature: float, anthropic_client) -> dict:
    """Claude-3.5-Haiku 응답 생성"""
    if not anthropic_client:
        return {
            'success': False,
            'answer': "Anthropic API 키가 설정되지 않았습니다.",
            'response_time': 0
        }
    
    start_time = time.time()
    
    try:
        system_prompt = """당신은 대한민국의 전문 법률 AI입니다. 주어진 법률 판례 정보를 바탕으로 질문에 대해 정확하고 신뢰할 수 있는 답변을 제공하세요. 
답변할 때는 관련 판례의 사건번호와 주요 내용을 인용하여 근거를 명확히 제시하세요."""
        
        user_prompt = f"""
법률 질문: {question}

관련 판례 정보:
{context}

위의 판례 정보를 참고하여 질문에 대해 상세히 답변해주세요.
"""
        
        response = anthropic_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1000,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        answer = response.content[0].text
        response_time = time.time() - start_time
        
        return {
            'success': True,
            'answer': answer,
            'response_time': response_time
        }
        
    except Exception as e:
        response_time = time.time() - start_time
        return {
            'success': False,
            'answer': f"Claude-3.5-Haiku 오류: {str(e)}",
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
        <h1>⚖️ 법률 AI 분석 시스템 v6.0</h1>
        <p>17개 대법원 판례 기반 • GPT-4o vs Claude-3.5-Haiku 비교 • 단순화 버전</p>
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
        - 17개 대법원 판례 JSON 데이터
        - 민사/형사 사건 포함
        - 키워드 기반 검색
        """)
    
    # 데이터 및 클라이언트 로드
    law_documents, load_status = load_law_documents()
    openai_client, anthropic_client = initialize_ai_clients()
    
    if not law_documents:
        st.error(load_status)
        st.stop()
    
    st.success(load_status)
    
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
            "부당해고 구제신청의 요건과 절차는 어떻게 되나요?",
            "근로자의 업무상 재해 인정 기준은 무엇인가요?"
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
        
        # 판례 정보 요약
        case_types = [doc['metadata'].get('case_type', 'Unknown') for doc in law_documents]
        case_type_counts = {case_type: case_types.count(case_type) for case_type in set(case_types)}
        
        st.metric("로드된 판례", f"{len(law_documents)}건")
        
        st.write("**사건 유형별 분포**")
        for case_type, count in case_type_counts.items():
            st.write(f"- {case_type}: {count}건")
        
        st.write("**API 연결 상태**")
        st.write(f"- OpenAI: {'✅' if openai_client else '❌'}")
        st.write(f"- Anthropic: {'✅' if anthropic_client else '❌'}")
    
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
        
        # 관련 판례 검색
        relevant_cases = search_relevant_cases(question, law_documents, top_k=3)
        
        if not relevant_cases:
            context = "관련 판례를 찾을 수 없습니다. 일반적인 법률 지식으로 답변하겠습니다."
            st.info("관련 판례를 찾지 못했습니다.")
        else:
            context = "\n\n".join([case['content'] for case in relevant_cases])
            st.success(f"관련 판례 {len(relevant_cases)}건을 찾았습니다.")
            
            # 관련 판례 표시
            with st.expander("🔍 검색된 관련 판례"):
                for i, case in enumerate(relevant_cases):
                    metadata = case['metadata']
                    st.write(f"**{i+1}. {metadata['case_number']}** - {metadata['case_name']} ({metadata['date']})")
        
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 각 모델 실행
        for i, model_name in enumerate(selected_models):
            status_text.text(f"🤖 {model_name} 분석 중...")
            progress_bar.progress((i) / len(selected_models))
            
            if model_name == "GPT-4o":
                result = get_gpt_response(question, context, temperature, openai_client)
            elif model_name == "Claude-3.5-Haiku":
                result = get_claude_response(question, context, temperature, anthropic_client)
            
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
        <p>⚖️ 법률 AI 분석 시스템 v6.0 | 17개 대법원 판례 기반 | Streamlit Framework</p>
        <p>🔬 Powered by OpenAI • Anthropic • 키워드 검색 시스템</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()