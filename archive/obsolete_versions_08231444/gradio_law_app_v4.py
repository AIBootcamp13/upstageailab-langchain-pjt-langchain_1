#!/usr/bin/env python3
"""
법률 도메인 Gradio 웹 인터페이스 v4.0  
JSON 판례 데이터 기반 RAG 시스템으로 법률 관련 질문 답변 및 모델 비교
"""

import os
import json
import time
import gradio as gr
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
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

# 글로벌 변수
law_retriever = None
law_cfg = None  
law_langsmith_manager = None

def initialize_law_system():
    """법률 시스템 초기화"""
    global law_retriever, law_cfg, law_langsmith_manager
    
    load_dotenv()
    
    # 설정 생성
    law_cfg = OmegaConf.create({
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
            'project_name': 'law-gradio-v4',
            'session_name': f'law-gradio-session_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    })
    
    try:
        law_retriever = create_law_rag_pipeline(law_cfg)
        law_langsmith_manager = LangSmithManager(law_cfg.langsmith)
        return "✅ 법률 판례 데이터베이스 로드 완료"
    except Exception as e:
        return f"❌ 초기화 실패: {str(e)}"

def get_law_case_info():
    """로드된 법률 판례 정보 조회"""
    try:
        law_loader = LawDocumentLoader()
        documents = law_loader.load_legal_documents()
        
        info = f"📚 로드된 판례: {len(documents)}건\n\n"
        
        if documents:
            # 사건 유형별 분포
            case_types = [doc['metadata'].get('case_type', 'Unknown') for doc in documents]
            case_type_counts = {case_type: case_types.count(case_type) for case_type in set(case_types)}
            
            info += "**사건 유형별 분포**\n"
            for case_type, count in case_type_counts.items():
                info += f"- {case_type}: {count}건\n"
            
            info += "\n**최근 판례**\n"
            for i, doc in enumerate(documents[:3]):
                metadata = doc['metadata']
                info += f"{i+1}. {metadata.get('case_number', 'N/A')} - {metadata.get('case_name', 'N/A')} ({metadata.get('date', 'N/A')})\n"
        
        return info
        
    except Exception as e:
        return f"판례 정보 조회 오류: {str(e)}"

def analyze_law_question(question, model1_enabled, model2_enabled, temperature):
    """법률 질문 분석"""
    global law_retriever, law_cfg, law_langsmith_manager
    
    if not question.strip():
        return "질문을 입력해주세요.", "", "", ""
    
    if not law_retriever:
        return "시스템이 초기화되지 않았습니다.", "", "", ""
    
    if not model1_enabled and not model2_enabled:
        return "최소 하나의 모델을 선택해주세요.", "", "", ""
    
    # 모델 설정
    models_config = {
        'GPT-4o': {
            'provider': 'openai',
            'model_name': 'gpt-4o',
            'temperature': temperature
        },
        'Claude-3.5-Haiku': {
            'provider': 'anthropic', 
            'model_name': 'claude-3-5-haiku-20241022',
            'temperature': temperature
        }
    }
    
    results = {}
    
    # 선택된 모델들 실행
    selected_models = []
    if model1_enabled:
        selected_models.append('GPT-4o')
    if model2_enabled:
        selected_models.append('Claude-3.5-Haiku')
    
    for model_name in selected_models:
        model_config = models_config[model_name]
        
        start_time = time.time()
        
        try:
            # LLM 생성
            llm_cfg = OmegaConf.create({
                'llm': {
                    'provider': model_config['provider'],
                    'model_name': model_config['model_name'],
                    'temperature': model_config['temperature']
                }
            })
            
            llm = create_llm(llm_cfg)
            qa_chain = create_qa_chain(law_retriever, llm, get_qa_prompt())
            
            # LangSmith 추적
            if law_langsmith_manager and law_langsmith_manager.enabled:
                run_id = law_langsmith_manager.start_run(
                    name=f"Law_Gradio_{model_name}",
                    inputs={"question": question}
                )
            
            # 질문 실행
            response = qa_chain.invoke({"question": question})
            answer = response['answer'] if isinstance(response, dict) else str(response)
            
            response_time = time.time() - start_time
            
            results[model_name] = {
                'answer': answer,
                'response_time': response_time,
                'success': True
            }
            
            if law_langsmith_manager and law_langsmith_manager.enabled:
                law_langsmith_manager.end_run(run_id, outputs={"answer": answer})
                
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"오류: {str(e)}"
            
            results[model_name] = {
                'answer': error_msg,
                'response_time': response_time,
                'success': False
            }
            
            if law_langsmith_manager and law_langsmith_manager.enabled:
                law_langsmith_manager.end_run(run_id, outputs={"error": error_msg})
    
    # 결과 포맷팅
    summary = f"📊 **분석 완료** (총 {len(results)}개 모델)\n\n"
    
    if len(results) > 1:
        avg_time = sum([r['response_time'] for r in results.values()]) / len(results)
        success_count = sum([1 for r in results.values() if r['success']])
        summary += f"- 평균 응답시간: {avg_time:.2f}초\n"
        summary += f"- 성공률: {success_count}/{len(results)}\n\n"
    
    # 각 모델 결과
    gpt_result = ""
    claude_result = ""
    
    if 'GPT-4o' in results:
        result = results['GPT-4o']
        status = "✅ 성공" if result['success'] else "❌ 실패"
        gpt_result = f"{status} ({result['response_time']:.2f}초)\n\n{result['answer']}"
    
    if 'Claude-3.5-Haiku' in results:
        result = results['Claude-3.5-Haiku']
        status = "✅ 성공" if result['success'] else "❌ 실패" 
        claude_result = f"{status} ({result['response_time']:.2f}초)\n\n{result['answer']}"
    
    # 비교 차트 생성
    chart_html = ""
    if len(results) > 1:
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
            height=400,
            template="plotly_white"
        )
        
        chart_html = fig.to_html(include_plotlyjs='cdn')
    
    return summary, gpt_result, claude_result, chart_html

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    # 초기화 메시지
    init_message = initialize_law_system()
    case_info = get_law_case_info()
    
    with gr.Blocks(title="⚖️ 법률 AI 분석 시스템 v4.0 (Gradio)", theme=gr.themes.Soft()) as interface:
        
        # 헤더
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #2c3e50, #3498db); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>⚖️ 법률 AI 분석 시스템 v4.0</h1>
            <p>대법원 판례 기반 RAG 시스템 • GPT-4o vs Claude-3.5-Haiku 비교 • Gradio Framework</p>
        </div>
        """)
        
        # 시스템 상태
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 시스템 상태")
                system_status = gr.Textbox(
                    label="초기화 상태",
                    value=init_message,
                    interactive=False,
                    lines=2
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### 📚 판례 정보")
                case_info_display = gr.Textbox(
                    label="로드된 판례 정보",
                    value=case_info,
                    interactive=False,
                    lines=8
                )
        
        gr.Markdown("---")
        
        # 메인 인터페이스
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📝 법률 질문 입력")
                
                # 샘플 질문 드롭다운
                sample_questions = gr.Dropdown(
                    label="샘플 질문 선택",
                    choices=[
                        "직접 입력",
                        "취업규칙을 근로자에게 불리하게 변경할 때 사용자가 지켜야 할 법적 요건은 무엇인가요?",
                        "퇴직금 지급 기일을 연장하는 합의를 했더라도 연장된 기일까지 지급하지 않으면 형사처벌을 받나요?",
                        "근로기준법에서 규정하는 퇴직금 지급 의무에 대해 설명해주세요.",
                        "사용자가 취업규칙 변경 시 근로자의 동의를 얻지 못했을 때의 법적 효과는?",
                        "퇴직급여보장법 위반 시 어떤 형사처벌을 받게 되나요?"
                    ],
                    value="직접 입력"
                )
                
                # 질문 입력창
                question_input = gr.Textbox(
                    label="법률 질문",
                    placeholder="법률 관련 질문을 입력하세요...",
                    lines=4
                )
                
                # 샘플 질문 선택 시 자동 입력
                def update_question(selected):
                    if selected == "직접 입력":
                        return ""
                    return selected
                
                sample_questions.change(update_question, sample_questions, question_input)
                
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 모델 설정")
                
                model1_enabled = gr.Checkbox(
                    label="🤖 GPT-4o 활성화",
                    value=True
                )
                
                model2_enabled = gr.Checkbox(
                    label="🤖 Claude-3.5-Haiku 활성화", 
                    value=True
                )
                
                temperature = gr.Slider(
                    label="Temperature (창의성 조절)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.1,
                    info="낮을수록 일관된 답변, 높을수록 창의적 답변"
                )
                
                analyze_btn = gr.Button(
                    "🔍 법률 분석 시작",
                    variant="primary",
                    size="lg"
                )
        
        gr.Markdown("---")
        
        # 결과 출력
        gr.Markdown("### 📊 분석 결과")
        
        # 요약 결과
        summary_output = gr.Markdown(label="분석 요약")
        
        # 모델별 상세 결과
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 🤖 GPT-4o 결과")
                gpt_output = gr.Textbox(
                    label="GPT-4o 상세 응답",
                    lines=8,
                    interactive=False
                )
                
            with gr.Column():
                gr.Markdown("#### 🤖 Claude-3.5-Haiku 결과")
                claude_output = gr.Textbox(
                    label="Claude-3.5-Haiku 상세 응답", 
                    lines=8,
                    interactive=False
                )
        
        # 비교 차트
        gr.Markdown("#### 📈 성능 비교 차트")
        chart_output = gr.HTML()
        
        # 분석 버튼 클릭 이벤트
        analyze_btn.click(
            analyze_law_question,
            inputs=[question_input, model1_enabled, model2_enabled, temperature],
            outputs=[summary_output, gpt_output, claude_output, chart_output]
        )
        
        # 푸터
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <p><strong>⚖️ 법률 AI 분석 시스템 v4.0</strong></p>
            <p>🔬 Powered by LangChain • OpenAI • Anthropic • LangSmith • Gradio</p>
            <p>📚 대법원 판례 데이터 기반 검색증강생성(RAG) 시스템</p>
        </div>
        """)
    
    return interface

def main():
    """메인 실행 함수"""
    
    print("⚖️ 법률 AI 분석 시스템 v4.0 (Gradio) 시작 중...")
    
    # Gradio 인터페이스 생성 및 실행
    interface = create_gradio_interface()
    
    # 인터페이스 실행
    interface.launch(
        server_name="0.0.0.0",
        server_port=7862,  # 기존 Gradio 앱과 다른 포트 사용
        share=False,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()