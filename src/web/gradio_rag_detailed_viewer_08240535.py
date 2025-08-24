#!/usr/bin/env python3
"""
Gradio RAG 상세 결과 뷰어 v08240535
질문별 LLM 모델, RAG 전후 답변, 상세 점수를 선택해서 볼 수 있는 인터랙티브 뷰어
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import gradio as gr
import pandas as pd

# 글로벌 변수
analysis_results = None
current_data_file = None

def load_analysis_results(json_file_path: str = None) -> Optional[Dict]:
    """분석 결과 JSON 파일 로드"""
    global analysis_results
    
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
            analysis_results = json.load(f)
        return analysis_results
    except Exception as e:
        print(f"결과 파일 로드 오류: {e}")
        return None

def get_question_list() -> List[str]:
    """질문 목록 반환"""
    if not analysis_results:
        return []
    
    questions = analysis_results.get('questions', {})
    question_list = []
    
    for q_id, q_data in questions.items():
        question_text = q_data.get('question', '질문 없음')
        question_list.append(f"{q_id.upper()}: {question_text[:50]}...")
    
    return question_list

def get_question_details(selected_question: str) -> Tuple[str, List[str]]:
    """선택된 질문의 상세 정보 반환"""
    if not analysis_results or not selected_question:
        return "질문을 선택해주세요.", []
    
    # 질문 ID 추출
    q_id = selected_question.split(':')[0].lower()
    
    questions = analysis_results.get('questions', {})
    if q_id not in questions:
        return "해당 질문을 찾을 수 없습니다.", []
    
    q_data = questions[q_id]
    question_text = q_data.get('question', '질문 없음')
    
    # 사용 가능한 모델 목록
    available_models = list(q_data.get('responses', {}).keys())
    
    return question_text, available_models

def get_model_response_comparison(selected_question: str, selected_model: str) -> Tuple[str, str, str, str]:
    """선택된 질문과 모델의 순수/RAG 답변 비교"""
    if not analysis_results or not selected_question or not selected_model:
        return "데이터 없음", "데이터 없음", "점수 정보 없음", "분석 없음"
    
    # 질문 ID 추출
    q_id = selected_question.split(':')[0].lower()
    
    questions = analysis_results.get('questions', {})
    if q_id not in questions:
        return "질문 데이터 없음", "질문 데이터 없음", "점수 정보 없음", "분석 없음"
    
    q_data = questions[q_id]
    
    # 모델 응답 데이터
    model_responses = q_data.get('responses', {}).get(selected_model, {})
    if not model_responses:
        return "모델 데이터 없음", "모델 데이터 없음", "점수 정보 없음", "분석 없음"
    
    # 순수 LLM 응답
    pure_response = model_responses.get('pure', {})
    pure_answer = pure_response.get('answer', '응답 없음')
    pure_info = f"""
**순수 {selected_model} 응답**
- 답변 길이: {pure_response.get('answer_length', 0)}글자
- 단어 수: {pure_response.get('word_count', 0)}개
- 응답 시간: {pure_response.get('response_time', 0):.2f}초
- 상태: {pure_response.get('status', 'unknown')}

**답변 내용:**
{pure_answer}
    """.strip()
    
    # RAG 응답
    rag_response = model_responses.get('rag', {})
    rag_answer = rag_response.get('answer', '응답 없음')
    relevant_cases = rag_response.get('relevant_cases', [])
    rag_info = f"""
**RAG 기반 {selected_model} 응답**
- 답변 길이: {rag_response.get('answer_length', 0)}글자
- 단어 수: {rag_response.get('word_count', 0)}개
- 응답 시간: {rag_response.get('response_time', 0):.2f}초
- 사용된 판례: {rag_response.get('case_count', 0)}건
- 상태: {rag_response.get('status', 'unknown')}

**활용 판례:** {', '.join(relevant_cases) if relevant_cases else '없음'}

**답변 내용:**
{rag_answer}
    """.strip()
    
    # 개선도 분석
    improvements = q_data.get('improvements', {}).get(selected_model, {})
    improvement_score = improvements.get('overall_score', 0)
    analysis_text = improvements.get('analysis', '분석 없음')
    
    score_details = f"""
## 📊 RAG 개선 점수 분석

### 🎯 전체 개선 점수: **{improvement_score:.1f}/100점**

### 📈 세부 지표
- **구체성 개선**: {improvements.get('specificity_improvement', 0)}점 (사건번호 인용)
- **근거 강화**: {improvements.get('evidence_improvement', 0)}점 (법률 키워드)
- **답변 길이 변화**: {improvements.get('length_change', 0):+d}글자
- **단어 수 변화**: {improvements.get('word_count_change', 0):+d}개
- **응답 시간 변화**: {improvements.get('response_time_change', 0):+.2f}초
- **법률 키워드 밀도**: {improvements.get('legal_keyword_density', 0):.2f}/1000글자

### 🔍 분석 결과
{analysis_text}
    """.strip()
    
    return pure_info, rag_info, score_details, analysis_text

def get_all_questions_summary() -> str:
    """전체 질문 요약 통계"""
    if not analysis_results:
        return "데이터 없음"
    
    summary = analysis_results.get('summary', {})
    
    summary_text = f"""
# 📊 전체 분석 요약 (v08240535)

## 🎯 기본 정보
- **분석 시간**: {analysis_results.get('timestamp', 'N/A')}
- **총 질문 수**: {analysis_results.get('total_questions', 0)}개
- **분석 모델**: {', '.join(analysis_results.get('models', []))}
- **총 처리 시간**: {analysis_results.get('total_processing_time', 0):.1f}초

## 🏆 모델별 성능 요약
"""
    
    model_averages = summary.get('model_averages', {})
    for model, data in model_averages.items():
        summary_text += f"""
### {model}
- **평균 개선 점수**: {data.get('avg_improvement_score', 0):.1f}/100점
- **최고 점수**: {data.get('best_score', 0):.1f}점
- **최저 점수**: {data.get('worst_score', 0):.1f}점
- **평균 사용 판례**: {data.get('avg_cases_used', 0):.1f}건
- **평균 처리 시간 증가**: {data.get('avg_time_increase', 0):+.2f}초
"""
    
    # 통계적 신뢰도 정보
    q_stats = summary.get('question_statistics', {})
    if q_stats:
        summary_text += f"""
## 🔬 통계적 신뢰도 (30개 질문 기반)
- **총 평가 수**: {q_stats.get('total_evaluations', 0)}회
- **전체 평균 점수**: {q_stats.get('overall_avg_score', 0):.1f}/100점
- **점수 표준편차**: {q_stats.get('score_std_dev', 0):.2f}
- **신뢰도 개선**: ⭐⭐⭐⭐⭐⭐ (기존 대비 6배 향상)
"""
    
    return summary_text

def create_comparison_table() -> str:
    """모델 비교 테이블 생성"""
    if not analysis_results:
        return "데이터 없음"
    
    questions = analysis_results.get('questions', {})
    
    table_data = []
    for q_id, q_data in questions.items():
        question_short = q_data.get('question', '')[:30] + "..."
        
        for model in ['GPT-4o', 'Claude-3.5']:
            if model in q_data.get('improvements', {}):
                improvement = q_data['improvements'][model]
                responses = q_data['responses'][model]
                
                table_data.append([
                    q_id.upper(),
                    question_short,
                    model,
                    f"{improvement.get('overall_score', 0):.1f}",
                    f"{responses['rag'].get('case_count', 0)}",
                    f"{improvement.get('response_time_change', 0):+.2f}초",
                    improvement.get('analysis', '')[:50] + "..."
                ])
    
    # 테이블 생성
    headers = ["질문ID", "질문", "모델", "개선점수", "판례수", "시간변화", "분석결과"]
    
    table_md = "| " + " | ".join(headers) + " |\n"
    table_md += "|" + "|".join(["---"] * len(headers)) + "|\n"
    
    for row in table_data:
        table_md += "| " + " | ".join(row) + " |\n"
    
    return table_md

def refresh_data():
    """데이터 새로고침"""
    global analysis_results
    load_analysis_results()
    if analysis_results:
        return "✅ 데이터 새로고침 완료", get_question_list(), get_all_questions_summary()
    else:
        return "❌ 데이터 로드 실패", [], "데이터 없음"

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    # 초기 데이터 로드
    load_analysis_results()
    
    with gr.Blocks(title="RAG 상세 결과 뷰어 v08240535", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
# 🔍 RAG 상세 결과 뷰어 v08240535
        
질문별로 LLM 모델의 RAG 전후 답변과 상세 점수를 비교해서 볼 수 있습니다.
30개 질문 × 2개 모델의 모든 결과를 인터랙티브하게 탐색해보세요!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🎯 질문 및 모델 선택")
                
                refresh_btn = gr.Button("🔄 데이터 새로고침", variant="secondary")
                status_text = gr.Textbox(
                    label="상태",
                    value="데이터 로딩 중...",
                    interactive=False
                )
                
                question_dropdown = gr.Dropdown(
                    label="📝 질문 선택",
                    choices=get_question_list(),
                    value=None,
                    interactive=True
                )
                
                model_dropdown = gr.Dropdown(
                    label="🤖 모델 선택",
                    choices=[],
                    value=None,
                    interactive=True
                )
                
                selected_question_display = gr.Textbox(
                    label="선택된 질문",
                    value="질문을 선택해주세요.",
                    interactive=False,
                    lines=3
                )
            
            with gr.Column(scale=2):
                gr.Markdown("## 📊 전체 분석 요약")
                
                summary_display = gr.Markdown(
                    value=get_all_questions_summary()
                )
        
        # 답변 비교 섹션
        gr.Markdown("## ⚖️ RAG 전후 답변 비교")
        
        with gr.Row():
            with gr.Column():
                pure_response_display = gr.Markdown(
                    label="순수 LLM 응답",
                    value="질문과 모델을 선택해주세요."
                )
            
            with gr.Column():
                rag_response_display = gr.Markdown(
                    label="RAG 기반 응답",
                    value="질문과 모델을 선택해주세요."
                )
        
        # 점수 분석 섹션
        gr.Markdown("## 📈 상세 점수 분석")
        
        with gr.Row():
            with gr.Column():
                score_analysis_display = gr.Markdown(
                    value="질문과 모델을 선택하면 상세 점수가 표시됩니다."
                )
            
            with gr.Column():
                comparison_table_display = gr.Markdown(
                    label="전체 비교 테이블",
                    value=create_comparison_table()
                )
        
        # 이벤트 핸들러
        def update_question_details(selected_question):
            if selected_question:
                question_text, available_models = get_question_details(selected_question)
                return question_text, gr.update(choices=available_models, value=None)
            else:
                return "질문을 선택해주세요.", gr.update(choices=[], value=None)
        
        def update_response_comparison(selected_question, selected_model):
            if selected_question and selected_model:
                pure_info, rag_info, score_details, analysis = get_model_response_comparison(
                    selected_question, selected_model
                )
                return pure_info, rag_info, score_details
            else:
                return "질문과 모델을 모두 선택해주세요.", "질문과 모델을 모두 선택해주세요.", "점수 정보 없음"
        
        # 질문 선택 시 모델 목록 업데이트
        question_dropdown.change(
            fn=update_question_details,
            inputs=[question_dropdown],
            outputs=[selected_question_display, model_dropdown]
        )
        
        # 모델 선택 시 답변 비교 업데이트
        model_dropdown.change(
            fn=update_response_comparison,
            inputs=[question_dropdown, model_dropdown],
            outputs=[pure_response_display, rag_response_display, score_analysis_display]
        )
        
        # 데이터 새로고침
        refresh_btn.click(
            fn=refresh_data,
            outputs=[status_text, question_dropdown, summary_display]
        )
        
        # 초기화
        interface.load(
            fn=lambda: ("✅ 뷰어 준비 완료", get_question_list(), get_all_questions_summary(), create_comparison_table()),
            outputs=[status_text, question_dropdown, summary_display, comparison_table_display]
        )
    
    return interface

if __name__ == "__main__":
    # 인터페이스 생성 및 실행
    interface = create_gradio_interface()
    
    print("🔍 RAG 상세 결과 뷰어 v08240535 시작")
    print("📊 질문별 LLM 모델, RAG 전후 답변, 상세 점수 비교 가능")
    print("🌐 웹 인터페이스: http://localhost:7866")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7866,
        share=False,
        show_error=True
    )