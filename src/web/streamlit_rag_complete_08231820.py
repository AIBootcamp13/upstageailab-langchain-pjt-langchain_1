#!/usr/bin/env python3
"""
RAG 성능 개선 비교 Streamlit 인터페이스 v08231820
완벽한 RAG 성능 비교 시스템의 웹 인터페이스
실시간 분석, 진행률 표시, 상세 차트, LangSmith 추적 통합
"""

import os
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리를 Python path에 추가
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.rag_improvement_complete_08231820 import RAGImprovementComparator, save_results_multiple_formats
from src.utils.version_manager import VersionManager
from src.utils.langsmith_simple import LangSmithSimple
from src.utils.path_utils import ensure_directory_exists

# 페이지 설정
st.set_page_config(
    page_title="🧠 RAG 성능 개선 완벽 분석 v08231820",
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링 (업그레이드)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .analysis-box {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        padding: 1.5rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .improvement-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem;
    }
    .score-excellent { background: linear-gradient(135deg, #4facfe, #00f2fe); color: white; }
    .score-good { background: linear-gradient(135deg, #43e97b, #38f9d7); color: white; }
    .score-fair { background: linear-gradient(135deg, #fa709a, #fee140); color: white; }
    .score-poor { background: linear-gradient(135deg, #ff9a9e, #fecfef); color: #333; }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid #f0f0f0;
    }
    .metric-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """시스템 초기화 (캐싱)"""
    load_dotenv()
    
    # 버전 관리자 초기화
    version_manager = VersionManager()
    
    # LangSmith 설정
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'langsmith': {
            'enabled': True,
            'project_name': 'streamlit-rag-complete-v08231820',
            'session_name': f'streamlit-session_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    })
    
    langsmith_manager = LangSmithSimple(cfg, version_manager)
    
    # 비교기 초기화
    comparator = RAGImprovementComparator(version_manager, langsmith_manager)
    
    return comparator, version_manager

def get_score_class(score):
    """점수에 따른 CSS 클래스 반환"""
    if score >= 80:
        return "score-excellent"
    elif score >= 60:
        return "score-good"
    elif score >= 40:
        return "score-fair"
    else:
        return "score-poor"

def create_improvement_chart(results):
    """개선 점수 비교 차트 생성"""
    data = []
    
    for q_id, q_data in results.get('questions', {}).items():
        for model in ['GPT-4o', 'Claude-3.5']:
            if model in q_data.get('improvements', {}):
                improvement = q_data['improvements'][model]
                data.append({
                    'Question': f"Q{q_id[-1]}",
                    'Model': model,
                    'Score': improvement['overall_score'],
                    'Question_Full': q_data['question'][:50] + '...'
                })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df, 
        x='Question', 
        y='Score', 
        color='Model',
        title='RAG 개선 점수 비교 (질문별)',
        hover_data=['Question_Full'],
        color_discrete_map={
            'GPT-4o': '#3498db',
            'Claude-3.5': '#e74c3c'
        }
    )
    
    fig.update_layout(
        yaxis_title="개선 점수 (0-100)",
        xaxis_title="질문",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_response_time_chart(results):
    """응답 시간 변화 차트 생성"""
    data = []
    
    for q_id, q_data in results.get('questions', {}).items():
        for model in ['GPT-4o', 'Claude-3.5']:
            if model in q_data.get('responses', {}):
                responses = q_data['responses'][model]
                pure_time = responses.get('pure', {}).get('response_time', 0)
                rag_time = responses.get('rag', {}).get('response_time', 0)
                
                data.extend([
                    {
                        'Question': f"Q{q_id[-1]}",
                        'Model': model,
                        'Type': 'Pure LLM',
                        'Response_Time': pure_time
                    },
                    {
                        'Question': f"Q{q_id[-1]}",
                        'Model': model,
                        'Type': 'RAG Applied',
                        'Response_Time': rag_time
                    }
                ])
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df, 
        x='Question', 
        y='Response_Time', 
        color='Type',
        facet_col='Model',
        title='응답 시간 비교 (순수 LLM vs RAG)',
        color_discrete_map={
            'Pure LLM': '#95a5a6',
            'RAG Applied': '#2ecc71'
        }
    )
    
    fig.update_layout(height=400)
    fig.update_yaxes(title="응답 시간 (초)")
    
    return fig

def create_metrics_radar_chart(results):
    """모델별 종합 성능 레이더 차트"""
    summary = results.get('summary', {})
    model_averages = summary.get('model_averages', {})
    
    if len(model_averages) < 2:
        return None
    
    models = list(model_averages.keys())
    
    # 메트릭 정규화 (0-100 스케일)
    metrics = ['개선점수', '효율성', '정확성', '속도', '활용도']
    
    model_data = {}
    for model in models:
        avg_data = model_averages[model]
        model_data[model] = [
            avg_data.get('avg_improvement_score', 0),  # 개선점수
            max(0, 100 - abs(avg_data.get('avg_time_increase', 0)) * 10),  # 효율성 (시간증가 패널티)
            min(100, avg_data.get('avg_improvement_score', 0) * 1.2),  # 정확성
            max(0, 100 - avg_data.get('avg_time_increase', 0) * 20),  # 속도
            min(100, avg_data.get('avg_cases_used', 0) * 25)  # 활용도 (판례사용)
        ]
    
    fig = go.Figure()
    
    colors = ['#3498db', '#e74c3c']
    for i, (model, values) in enumerate(model_data.items()):
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # 닫힌 다각형을 위해
            theta=metrics + [metrics[0]],
            fill='toself',
            name=model,
            line_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title="모델별 종합 성능 비교",
        height=500
    )
    
    return fig

def run_analysis_with_progress(comparator, questions, temperature):
    """진행률 표시와 함께 분석 실행"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    current_step = 0
    total_steps = len(questions) * 2 * 2  # questions * models * (pure+rag)
    
    def progress_callback(progress):
        nonlocal current_step
        current_step = int(progress * total_steps)
        progress_bar.progress(progress)
        
        if progress < 1.0:
            status_text.text(f"분석 진행 중... ({current_step}/{total_steps})")
        else:
            status_text.text("✅ 분석 완료!")
    
    # 분석 실행
    results = comparator.compare_models(questions, temperature, progress_callback)
    
    time.sleep(1)  # 완료 메시지 표시 시간
    progress_bar.empty()
    status_text.empty()
    
    return results

def main():
    """메인 애플리케이션"""
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🧠 RAG 성능 개선 완벽 분석 시스템</h1>
        <h3>v08231820 • LangSmith 추적 • 실시간 시각화</h3>
        <p>순수 LLM vs RAG 적용 성능 비교 • GPT-4o • Claude-3.5 • 17개 대법원 판례</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 분석 설정")
        
        # 질문 선택
        available_questions = [
            "취업규칙을 근로자에게 불리하게 변경할 때 사용자가 지켜야 할 법적 요건은 무엇인가요?",
            "퇴직금 지급 기일을 연장하는 합의를 했더라도 연장된 기일까지 지급하지 않으면 형사처벌을 받나요?",
            "부당해고 구제신청의 요건과 절차는 어떻게 되나요?",
            "근로자의 업무상 재해 인정 기준과 절차는 어떻게 되나요?",
            "사업주가 근로계약을 해지할 때 지켜야 할 법적 절차는 무엇인가요?"
        ]
        
        selected_questions = st.multiselect(
            "분석할 질문 선택",
            available_questions,
            default=available_questions[:3],
            help="최대 5개 질문까지 선택 가능"
        )
        
        # 온도 설정
        temperature = st.slider(
            "Temperature (창의성 조절)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="낮을수록 일관된 답변, 높을수록 창의적 답변"
        )
        
        # 고급 설정
        with st.expander("🔧 고급 설정"):
            show_raw_responses = st.checkbox("원본 응답 표시", value=False)
            auto_save = st.checkbox("결과 자동 저장", value=True)
            real_time_charts = st.checkbox("실시간 차트 업데이트", value=True)
        
        st.markdown("---")
        st.markdown("""
        **📊 분석 정보**
        - 17개 대법원 판례 데이터
        - LangSmith 전체 추적
        - JSON/CSV/Markdown 다중 출력
        - 실시간 성능 시각화
        """)
    
    # 시스템 초기화
    comparator, version_manager = initialize_system()
    
    st.success("✅ RAG 성능 분석 시스템 초기화 완료")
    
    # 메인 인터페이스
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 선택된 질문들")
        if selected_questions:
            for i, question in enumerate(selected_questions, 1):
                st.write(f"**Q{i}**: {question}")
        else:
            st.warning("분석할 질문을 선택해주세요.")
    
    with col2:
        st.subheader("⚡ 빠른 실행")
        
        analyze_button = st.button(
            "🚀 RAG 성능 분석 시작",
            type="primary",
            use_container_width=True,
            disabled=not selected_questions
        )
        
        if st.button("📊 샘플 결과 보기", use_container_width=True):
            st.info("샘플 분석 결과를 표시합니다...")
    
    # 분석 실행
    if analyze_button and selected_questions:
        st.markdown("""
        <div class="analysis-box">
            <h3>🔍 RAG 성능 분석 실행 중</h3>
            <p>순수 LLM과 RAG 적용 모델을 비교 분석합니다...</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # 실제 분석 실행
            results = run_analysis_with_progress(comparator, selected_questions, temperature)
            
            # 세션 상태에 결과 저장
            st.session_state['analysis_results'] = results
            st.session_state['analysis_timestamp'] = datetime.now()
            
            st.success("🎉 분석 완료!")
            
        except Exception as e:
            st.error(f"❌ 분석 중 오류 발생: {str(e)}")
            st.stop()
    
    # 결과 표시
    if 'analysis_results' in st.session_state:
        results = st.session_state['analysis_results']
        
        st.markdown("---")
        st.header("📊 분석 결과")
        
        # 요약 메트릭
        summary = results.get('summary', {})
        model_averages = summary.get('model_averages', {})
        
        if model_averages:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                gpt_score = model_averages.get('GPT-4o', {}).get('avg_improvement_score', 0)
                score_class = get_score_class(gpt_score)
                st.markdown(f"""
                <div class="improvement-score {score_class}">
                    GPT-4o<br>{gpt_score:.1f}/100
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                claude_score = model_averages.get('Claude-3.5', {}).get('avg_improvement_score', 0)
                score_class = get_score_class(claude_score)
                st.markdown(f"""
                <div class="improvement-score {score_class}">
                    Claude-3.5<br>{claude_score:.1f}/100
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                total_questions = len(results.get('questions', {}))
                st.metric("분석 질문 수", total_questions)
            
            with col4:
                total_cases = results.get('metadata', {}).get('total_cases', 0)
                st.metric("참조 판례 수", f"{total_cases}건")
        
        # 상세 차트
        tab1, tab2, tab3, tab4 = st.tabs(["📈 개선 점수", "⏱️ 응답 시간", "🎯 종합 성능", "📋 상세 결과"])
        
        with tab1:
            st.subheader("RAG 개선 점수 비교")
            improvement_chart = create_improvement_chart(results)
            if improvement_chart:
                st.plotly_chart(improvement_chart, use_container_width=True)
            else:
                st.warning("차트 데이터가 없습니다.")
        
        with tab2:
            st.subheader("응답 시간 변화 분석")
            time_chart = create_response_time_chart(results)
            if time_chart:
                st.plotly_chart(time_chart, use_container_width=True)
            else:
                st.warning("차트 데이터가 없습니다.")
        
        with tab3:
            st.subheader("모델별 종합 성능")
            radar_chart = create_metrics_radar_chart(results)
            if radar_chart:
                st.plotly_chart(radar_chart, use_container_width=True)
            
            # 성능 비교 요약
            perf_comp = summary.get('performance_comparison', {})
            if perf_comp:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("더 나은 개선", perf_comp.get('better_improvement', 'N/A'))
                
                with col2:
                    st.metric("더 빠른 처리", perf_comp.get('faster_processing', 'N/A'))
                
                with col3:
                    st.metric("점수 차이", f"{perf_comp.get('score_difference', 0):.1f}점")
        
        with tab4:
            st.subheader("질문별 상세 결과")
            
            for q_id, q_data in results.get('questions', {}).items():
                with st.expander(f"Q{q_id[-1]}. {q_data['question'][:80]}...", expanded=False):
                    
                    col1, col2 = st.columns(2)
                    
                    for i, model in enumerate(['GPT-4o', 'Claude-3.5']):
                        col = col1 if i == 0 else col2
                        
                        if model in q_data.get('improvements', {}):
                            improvement = q_data['improvements'][model]
                            responses = q_data['responses'][model]
                            
                            with col:
                                st.markdown(f"#### {model}")
                                st.metric("개선 점수", f"{improvement['overall_score']:.1f}/100")
                                st.write(f"**분석**: {improvement['analysis']}")
                                st.write(f"**시간 변화**: {improvement['response_time_change']:+.2f}초")
                                st.write(f"**사용 판례**: {responses['rag'].get('case_count', 0)}건")
                                
                                if show_raw_responses:
                                    with st.expander(f"{model} 원본 응답"):
                                        st.write("**순수 LLM:**")
                                        st.code(responses['pure']['answer'][:300] + "...")
                                        st.write("**RAG 적용:**")
                                        st.code(responses['rag']['answer'][:300] + "...")
        
        # 자동 저장
        if auto_save:
            output_dir = ensure_directory_exists("results/rag_improvement_complete")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            try:
                json_path, csv_path, report_path = save_results_multiple_formats(
                    results, Path(output_dir), timestamp
                )
                
                st.success(f"✅ 결과가 자동 저장되었습니다!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        "📄 JSON 다운로드",
                        data=json.dumps(results, ensure_ascii=False, indent=2),
                        file_name=f"rag_results_{timestamp}.json",
                        mime="application/json"
                    )
                
                with col2:
                    if csv_path.exists():
                        with open(csv_path, 'r', encoding='utf-8') as f:
                            st.download_button(
                                "📊 CSV 다운로드",
                                data=f.read(),
                                file_name=f"rag_summary_{timestamp}.csv",
                                mime="text/csv"
                            )
                
                with col3:
                    if report_path.exists():
                        with open(report_path, 'r', encoding='utf-8') as f:
                            st.download_button(
                                "📋 보고서 다운로드",
                                data=f.read(),
                                file_name=f"rag_report_{timestamp}.md",
                                mime="text/markdown"
                            )
                
            except Exception as e:
                st.warning(f"자동 저장 실패: {e}")
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>🧠 RAG 성능 개선 완벽 분석 시스템 v08231820</strong></p>
        <p>🔬 Powered by LangChain • OpenAI • Anthropic • LangSmith • Streamlit</p>
        <p>⚖️ 17개 대법원 판례 기반 RAG 성능 검증 시스템</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()