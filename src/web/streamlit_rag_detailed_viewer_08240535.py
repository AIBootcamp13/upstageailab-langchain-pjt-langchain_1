#!/usr/bin/env python3
"""
Streamlit RAG 상세 결과 뷰어 v08240535  
질문별 LLM 모델, RAG 전후 답변, 상세 점수를 선택해서 볼 수 있는 고급 뷰어
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="RAG 상세 결과 뷰어 v08240535",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_analysis_results(json_file_path: str = None) -> Optional[Dict]:
    """분석 결과 JSON 파일 로드 (캐시됨)"""
    
    if json_file_path is None:
        # 기본 경로에서 가장 최근 파일 찾기
        results_dir = Path("results/rag_improvement_v08240535")
        if results_dir.exists():
            json_files = list(results_dir.glob("rag_improvement_v08240535_*.json"))
            if json_files:
                json_file_path = str(max(json_files, key=lambda x: x.stat().st_mtime))
            else:
                return None
        else:
            return None
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"결과 파일 로드 오류: {e}")
        return None

def get_question_options(analysis_results: Dict) -> List[str]:
    """질문 선택 옵션 반환"""
    if not analysis_results:
        return []
    
    questions = analysis_results.get('questions', {})
    question_options = []
    
    for q_id, q_data in questions.items():
        question_text = q_data.get('question', '질문 없음')
        question_options.append(f"{q_id.upper()}: {question_text}")
    
    return question_options

def display_question_analysis(analysis_results: Dict, selected_question: str, selected_model: str):
    """선택된 질문과 모델의 상세 분석 표시"""
    
    # 질문 ID 추출
    q_id = selected_question.split(':')[0].lower()
    
    questions = analysis_results.get('questions', {})
    if q_id not in questions:
        st.error("해당 질문을 찾을 수 없습니다.")
        return
    
    q_data = questions[q_id]
    question_text = q_data.get('question', '질문 없음')
    
    # 질문 표시
    st.markdown(f"### 📝 선택된 질문")
    st.info(question_text)
    
    # 모델 응답 데이터
    model_responses = q_data.get('responses', {}).get(selected_model, {})
    if not model_responses:
        st.error(f"{selected_model} 데이터를 찾을 수 없습니다.")
        return
    
    # 개선도 분석
    improvements = q_data.get('improvements', {}).get(selected_model, {})
    improvement_score = improvements.get('overall_score', 0)
    
    # 메트릭 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("개선 점수", f"{improvement_score:.1f}/100")
    
    with col2:
        case_count = model_responses.get('rag', {}).get('case_count', 0)
        st.metric("사용된 판례", f"{case_count}건")
    
    with col3:
        time_change = improvements.get('response_time_change', 0)
        st.metric("응답시간 변화", f"{time_change:+.2f}초")
    
    with col4:
        length_change = improvements.get('length_change', 0)
        st.metric("답변 길이 변화", f"{length_change:+d}글자")
    
    # RAG 전후 답변 비교
    st.markdown("## ⚖️ RAG 전후 답변 비교")
    
    col_pure, col_rag = st.columns(2)
    
    with col_pure:
        st.markdown(f"### 순수 {selected_model} 응답")
        
        pure_response = model_responses.get('pure', {})
        pure_answer = pure_response.get('answer', '응답 없음')
        
        # 순수 응답 메타데이터
        with st.expander("응답 정보"):
            st.write(f"**답변 길이**: {pure_response.get('answer_length', 0)}글자")
            st.write(f"**단어 수**: {pure_response.get('word_count', 0)}개")
            st.write(f"**응답 시간**: {pure_response.get('response_time', 0):.2f}초")
            st.write(f"**상태**: {pure_response.get('status', 'unknown')}")
        
        st.text_area(
            "순수 LLM 답변",
            value=pure_answer,
            height=300,
            key=f"pure_{q_id}_{selected_model}"
        )
    
    with col_rag:
        st.markdown(f"### RAG 기반 {selected_model} 응답")
        
        rag_response = model_responses.get('rag', {})
        rag_answer = rag_response.get('answer', '응답 없음')
        relevant_cases = rag_response.get('relevant_cases', [])
        
        # RAG 응답 메타데이터
        with st.expander("응답 정보 및 활용 판례"):
            st.write(f"**답변 길이**: {rag_response.get('answer_length', 0)}글자")
            st.write(f"**단어 수**: {rag_response.get('word_count', 0)}개")
            st.write(f"**응답 시간**: {rag_response.get('response_time', 0):.2f}초")
            st.write(f"**사용된 판례**: {rag_response.get('case_count', 0)}건")
            st.write(f"**상태**: {rag_response.get('status', 'unknown')}")
            
            if relevant_cases:
                st.write("**활용된 판례 번호**:")
                for case in relevant_cases:
                    st.write(f"- {case}")
        
        st.text_area(
            "RAG 기반 답변",
            value=rag_answer,
            height=300,
            key=f"rag_{q_id}_{selected_model}"
        )
    
    # 상세 점수 분석
    st.markdown("## 📈 상세 점수 분석")
    
    # 점수 구성 차트
    score_data = {
        '평가 항목': ['구체성 개선', '근거 강화', '길이 증가', '판례 활용'],
        '점수': [
            improvements.get('specificity_improvement', 0) * 20,  # 사건번호 × 20점
            improvements.get('evidence_improvement', 0) * 5,     # 법률키워드 × 5점
            min(improvements.get('length_change', 0), 500) / 10, # 길이 변화 / 10
            case_count * 5                                       # 판례수 × 5점
        ]
    }
    
    df_scores = pd.DataFrame(score_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 점수 구성 바 차트
        fig_bar = px.bar(
            df_scores,
            x='평가 항목',
            y='점수',
            title=f"{selected_model} RAG 개선 점수 구성",
            color='점수',
            color_continuous_scale='viridis'
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # 상세 지표 표
        st.markdown("### 📊 상세 지표")
        
        detail_data = {
            '항목': [
                '구체성 개선 (사건번호)',
                '근거 강화 (법률 키워드)',
                '답변 길이 변화',
                '단어 수 변화',
                '응답 시간 변화',
                '법률 키워드 밀도',
                '사용된 판례 수'
            ],
            '값': [
                f"{improvements.get('specificity_improvement', 0)}개",
                f"{improvements.get('evidence_improvement', 0)}개",
                f"{improvements.get('length_change', 0):+d}글자",
                f"{improvements.get('word_count_change', 0):+d}개",
                f"{improvements.get('response_time_change', 0):+.2f}초",
                f"{improvements.get('legal_keyword_density', 0):.2f}/1000글자",
                f"{case_count}건"
            ]
        }
        
        df_details = pd.DataFrame(detail_data)
        st.dataframe(df_details, use_container_width=True)
    
    # 분석 요약
    st.markdown("### 🔍 분석 요약")
    analysis_text = improvements.get('analysis', '분석 없음')
    st.success(f"**종합 분석**: {analysis_text}")

def display_overall_summary(analysis_results: Dict):
    """전체 요약 통계 표시"""
    
    summary = analysis_results.get('summary', {})
    
    st.markdown("## 📊 전체 분석 요약")
    
    # 기본 정보
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 질문 수", f"{analysis_results.get('total_questions', 0)}개")
    
    with col2:
        total_time = analysis_results.get('total_processing_time', 0)
        st.metric("총 처리 시간", f"{total_time:.1f}초")
    
    with col3:
        q_stats = summary.get('question_statistics', {})
        total_evals = q_stats.get('total_evaluations', 0)
        st.metric("총 평가 수", f"{total_evals}회")
    
    with col4:
        avg_score = q_stats.get('overall_avg_score', 0)
        st.metric("전체 평균 점수", f"{avg_score:.1f}/100")
    
    # 모델별 성능 비교
    st.markdown("### 🏆 모델별 성능 비교")
    
    model_averages = summary.get('model_averages', {})
    
    if model_averages:
        # 성능 비교 차트 데이터
        models = list(model_averages.keys())
        avg_scores = [data.get('avg_improvement_score', 0) for data in model_averages.values()]
        best_scores = [data.get('best_score', 0) for data in model_averages.values()]
        worst_scores = [data.get('worst_score', 0) for data in model_averages.values()]
        avg_cases = [data.get('avg_cases_used', 0) for data in model_averages.values()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 평균 점수 비교
            fig_comparison = go.Figure()
            
            fig_comparison.add_trace(go.Bar(
                name='평균 점수',
                x=models,
                y=avg_scores,
                marker_color='lightblue',
                text=[f"{score:.1f}" for score in avg_scores],
                textposition='auto'
            ))
            
            fig_comparison.add_trace(go.Scatter(
                name='최고 점수',
                x=models,
                y=best_scores,
                mode='markers+text',
                marker=dict(color='green', size=12),
                text=[f"최고: {score:.1f}" for score in best_scores],
                textposition='top center'
            ))
            
            fig_comparison.add_trace(go.Scatter(
                name='최저 점수',
                x=models,
                y=worst_scores,
                mode='markers+text',
                marker=dict(color='red', size=12),
                text=[f"최저: {score:.1f}" for score in worst_scores],
                textposition='bottom center'
            ))
            
            fig_comparison.update_layout(
                title="모델별 RAG 개선 점수 비교",
                xaxis_title="모델",
                yaxis_title="점수",
                height=400
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        with col2:
            # 판례 활용도 비교
            fig_cases = px.bar(
                x=models,
                y=avg_cases,
                title="모델별 평균 판례 활용도",
                labels={'x': '모델', 'y': '평균 판례 수'},
                text=[f"{cases:.1f}건" for cases in avg_cases]
            )
            fig_cases.update_traces(textposition='inside')
            fig_cases.update_layout(height=400)
            st.plotly_chart(fig_cases, use_container_width=True)
        
        # 모델별 상세 통계 테이블
        st.markdown("### 📋 모델별 상세 통계")
        
        model_stats_data = []
        for model, data in model_averages.items():
            model_stats_data.append({
                '모델': model,
                '평균 점수': f"{data.get('avg_improvement_score', 0):.1f}",
                '최고 점수': f"{data.get('best_score', 0):.1f}",
                '최저 점수': f"{data.get('worst_score', 0):.1f}",
                '평균 판례': f"{data.get('avg_cases_used', 0):.1f}건",
                '평균 시간 증가': f"{data.get('avg_time_increase', 0):+.2f}초",
                '평균 길이 증가': f"{data.get('avg_length_increase', 0):+.0f}글자"
            })
        
        df_model_stats = pd.DataFrame(model_stats_data)
        st.dataframe(df_model_stats, use_container_width=True)

def display_question_performance_chart(analysis_results: Dict):
    """질문별 성능 차트 표시"""
    
    st.markdown("## 📈 질문별 성능 분포")
    
    questions = analysis_results.get('questions', {})
    
    # 질문별 점수 데이터 수집
    chart_data = []
    
    for q_id, q_data in questions.items():
        question_short = q_data.get('question', '')[:30] + "..."
        
        for model in ['GPT-4o', 'Claude-3.5']:
            if model in q_data.get('improvements', {}):
                improvement = q_data['improvements'][model]
                score = improvement.get('overall_score', 0)
                case_count = q_data['responses'][model]['rag'].get('case_count', 0)
                
                chart_data.append({
                    '질문ID': q_id.upper(),
                    '질문': question_short,
                    '모델': model,
                    '개선점수': score,
                    '판례수': case_count,
                    '분석': improvement.get('analysis', '')[:50] + "..."
                })
    
    if chart_data:
        df_chart = pd.DataFrame(chart_data)
        
        # 점수 분포 히스토그램
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                df_chart,
                x='개선점수',
                color='모델',
                title="RAG 개선 점수 분포",
                nbins=20,
                marginal="box"
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # 모델별 점수 박스플롯
            fig_box = px.box(
                df_chart,
                x='모델',
                y='개선점수',
                title="모델별 점수 분포",
                points="all"
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # 상위/하위 성능 질문
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 상위 성능 질문 (Top 10)")
            top_questions = df_chart.nlargest(10, '개선점수')
            st.dataframe(
                top_questions[['질문ID', '모델', '개선점수', '판례수']],
                use_container_width=True
            )
        
        with col2:
            st.markdown("### ⬇️ 개선 필요 질문 (Bottom 10)")
            bottom_questions = df_chart.nsmallest(10, '개선점수')
            st.dataframe(
                bottom_questions[['질문ID', '모델', '개선점수', '판례수']],
                use_container_width=True
            )

def main():
    """메인 함수"""
    
    st.title("🔍 RAG 상세 결과 뷰어 v08240535")
    st.markdown("질문별 LLM 모델의 RAG 전후 답변과 상세 점수를 비교 분석하는 고급 뷰어")
    
    # 데이터 로드
    analysis_results = load_analysis_results()
    
    if not analysis_results:
        st.error("❌ 분석 결과 데이터를 찾을 수 없습니다. 먼저 RAG 분석을 실행해주세요.")
        return
    
    # 사이드바
    with st.sidebar:
        st.markdown("## 🎯 분석 조건 선택")
        
        # 질문 선택
        question_options = get_question_options(analysis_results)
        selected_question = st.selectbox(
            "📝 분석할 질문 선택",
            options=question_options,
            index=0 if question_options else None
        )
        
        # 모델 선택
        available_models = analysis_results.get('models', [])
        selected_model = st.selectbox(
            "🤖 분석할 모델 선택",
            options=available_models,
            index=0 if available_models else None
        )
        
        st.markdown("---")
        
        # 데이터 정보
        st.markdown("## ℹ️ 데이터 정보")
        st.info(f"""
**버전**: {analysis_results.get('version', 'N/A')}  
**분석 시간**: {analysis_results.get('timestamp', 'N/A')}  
**총 질문**: {analysis_results.get('total_questions', 0)}개  
**분석 모델**: {', '.join(analysis_results.get('models', []))}  
        """)
        
        # 새로고침 버튼
        if st.button("🔄 데이터 새로고침"):
            st.cache_data.clear()
            st.rerun()
    
    # 메인 콘텐츠
    tab1, tab2, tab3 = st.tabs(["🔍 질문별 상세 분석", "📊 전체 요약", "📈 성능 분포"])
    
    with tab1:
        if selected_question and selected_model:
            display_question_analysis(analysis_results, selected_question, selected_model)
        else:
            st.warning("질문과 모델을 선택해주세요.")
    
    with tab2:
        display_overall_summary(analysis_results)
    
    with tab3:
        display_question_performance_chart(analysis_results)

if __name__ == "__main__":
    main()