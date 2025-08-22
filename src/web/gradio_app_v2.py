import gradio as gr
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

# 전역 변수로 RAG 파이프라인 저장
PIPELINE_INITIALIZED = False
RAG_COMPONENTS = {}

def initialize_rag_pipeline():
    """RAG 파이프라인 초기화"""
    global PIPELINE_INITIALIZED, RAG_COMPONENTS
    
    if PIPELINE_INITIALIZED:
        return RAG_COMPONENTS
    
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
            "project_name": "langchain-rag-project-gradio",
            "session_name": "gradio-session",
            "tags": ["gradio", "web", "comparison", "v2"]
        }
    })
    
    try:
        # 문서 로딩 및 전처리
        documents = load_documents(cfg)
        split_documents_list = split_documents(cfg, documents)
        embeddings = get_embedding_model(cfg)
        vectorstore = get_vector_store(cfg, split_documents_list, embeddings)
        
        # Retriever 생성
        if cfg.retriever.type == "bm25" or cfg.retriever.type == "ensemble":
            retriever = get_retriever(cfg, vectorstore, documents=split_documents_list)
        else:
            retriever = get_retriever(cfg, vectorstore)
        
        prompt = get_qa_prompt()
        
        # 버전 관리 시스템 초기화
        version_manager = VersionManager()
        langsmith = LangSmithSimple(cfg, version_manager)
        
        RAG_COMPONENTS = {
            "cfg": cfg,
            "retriever": retriever,
            "prompt": prompt,
            "version_manager": version_manager,
            "langsmith": langsmith,
            "doc_count": len(documents),
            "chunk_count": len(split_documents_list)
        }
        
        PIPELINE_INITIALIZED = True
        return RAG_COMPONENTS
        
    except Exception as e:
        return {"error": f"RAG 파이프라인 초기화 실패: {str(e)}"}

def run_single_model_test(question, model_name, provider, model_id, temperature):
    """단일 모델 테스트 실행"""
    try:
        components = initialize_rag_pipeline()
        if "error" in components:
            return {"error": components["error"]}
        
        retriever = components["retriever"]
        prompt = components["prompt"]
        
        # 모델 설정
        model_cfg = OmegaConf.create({
            "llm": {
                "provider": provider,
                "model_name": model_id,
                "temperature": temperature
            }
        })
        
        start_time = time.time()
        llm = get_llm(model_cfg)
        qa_chain = get_qa_chain(llm, retriever, prompt)
        response = qa_chain.invoke(question)
        execution_time = time.time() - start_time
        
        result = {
            "model": model_name,
            "response": str(response),
            "execution_time": execution_time,
            "response_length": len(str(response)),
            "word_count": len(str(response).split()),
            "success": True,
            "timestamp": datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result
        
    except Exception as e:
        return {
            "model": model_name,
            "response": f"오류 발생: {str(e)}",
            "execution_time": 0,
            "response_length": 0,
            "word_count": 0,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S")
        }

def compare_models(question, models_to_compare, temperature):
    """모델 비교 실행"""
    if not question.strip():
        return "❌ 질문을 입력해주세요!", None, None
    
    if len(models_to_compare) < 1:
        return "❌ 최소 1개 모델을 선택해주세요!", None, None
    
    # 모델 정보 매핑
    model_info = {
        "GPT-4o": {"provider": "openai", "model_id": "gpt-4o"},
        "Claude-3.5-Haiku": {"provider": "anthropic", "model_id": "claude-3-5-haiku-20241022"},
    }
    
    results = []
    result_text = f"# 🤖 모델 비교 결과\n\n"
    result_text += f"**📅 실행시간:** {datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    result_text += f"**❓ 질문:** {question}\n\n"
    result_text += f"**🌡️ Temperature:** {temperature}\n\n"
    result_text += "---\n\n"
    
    # 각 모델별로 테스트 실행
    for model_name in models_to_compare:
        if model_name in model_info:
            model_data = model_info[model_name]
            
            result_text += f"## 🤖 {model_name} 테스트 중...\n"
            
            result = run_single_model_test(
                question, 
                model_name, 
                model_data["provider"], 
                model_data["model_id"], 
                temperature
            )
            
            results.append(result)
            
            if result["success"]:
                result_text += f"### ✅ 성공\n"
                result_text += f"- **⏱️ 실행시간:** {result['execution_time']:.2f}초\n"
                result_text += f"- **📏 응답길이:** {result['response_length']}자\n" 
                result_text += f"- **📊 단어수:** {result['word_count']}개\n"
                result_text += f"- **📝 응답:** {result['response']}\n\n"
            else:
                result_text += f"### ❌ 실패\n"
                result_text += f"- **오류:** {result.get('error', '알 수 없는 오류')}\n\n"
    
    # 성능 비교 분석
    successful_results = [r for r in results if r["success"]]
    
    if len(successful_results) >= 2:
        result_text += "---\n\n## 🏆 성능 비교 분석\n\n"
        
        # 속도 비교
        fastest = min(successful_results, key=lambda x: x["execution_time"])
        slowest = max(successful_results, key=lambda x: x["execution_time"])
        
        if fastest != slowest:
            speed_diff = ((slowest["execution_time"] - fastest["execution_time"]) / fastest["execution_time"]) * 100
            result_text += f"### ⚡ 속도 비교\n"
            result_text += f"- **우승자:** {fastest['model']} ({fastest['execution_time']:.2f}초)\n"
            result_text += f"- **차이:** {speed_diff:.1f}% 더 빠름\n\n"
        
        # 응답 길이 비교
        longest = max(successful_results, key=lambda x: x["response_length"])
        shortest = min(successful_results, key=lambda x: x["response_length"])
        
        if longest != shortest:
            length_diff = ((longest["response_length"] - shortest["response_length"]) / shortest["response_length"]) * 100
            result_text += f"### 📏 응답 길이 비교\n"
            result_text += f"- **가장 상세:** {longest['model']} ({longest['response_length']}자)\n"
            result_text += f"- **차이:** {length_diff:.1f}% 더 길음\n\n"
    
    # 차트 생성
    chart_time = None
    chart_length = None
    
    if len(successful_results) >= 1:
        # 실행 시간 차트
        fig_time = px.bar(
            x=[r['model'] for r in successful_results],
            y=[r['execution_time'] for r in successful_results],
            title="⏱️ 모델별 응답 시간 비교 (초)",
            color=[r['model'] for r in successful_results],
            color_discrete_sequence=px.colors.qualitative.Pastel1
        )
        fig_time.update_layout(
            showlegend=False,
            height=400,
            font=dict(size=12),
            title_font_size=16
        )
        chart_time = fig_time
        
        # 응답 길이 차트
        fig_length = px.bar(
            x=[r['model'] for r in successful_results],
            y=[r['response_length'] for r in successful_results],
            title="📏 모델별 응답 길이 비교 (글자수)",
            color=[r['model'] for r in successful_results],
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_length.update_layout(
            showlegend=False,
            height=400,
            font=dict(size=12),
            title_font_size=16
        )
        chart_length = fig_length
    
    return result_text, chart_time, chart_length

def get_pipeline_info():
    """파이프라인 정보 반환"""
    components = initialize_rag_pipeline()
    if "error" in components:
        return f"❌ {components['error']}"
    
    info = f"""
    # 🚀 RAG 파이프라인 상태
    
    ## ✅ 시스템 준비 완료
    
    ### 📊 문서 정보
    - **문서 페이지 수:** {components['doc_count']}페이지
    - **텍스트 청크 수:** {components['chunk_count']}개
    - **임베딩 모델:** text-embedding-ada-002 (OpenAI)
    - **벡터 데이터베이스:** FAISS
    - **검색 방식:** Ensemble (BM25 + Vector Search)
    
    ### 🔧 시스템 구성
    - **LangSmith 추적:** 활성화
    - **버전 관리:** 한국시각 기반 자동 백업
    - **지원 모델:** GPT-4o, Claude-3.5-Haiku
    
    ### 💡 사용법
    1. 왼쪽에서 질문을 입력하세요
    2. 비교할 모델들을 선택하세요  
    3. Temperature를 조정하세요
    4. '🚀 모델 비교 시작' 버튼을 클릭하세요
    """
    
    return info

# Gradio 인터페이스 생성
def create_gradio_interface():
    with gr.Blocks(
        title="🤖 LLM 모델 비교 시스템 v2.0 (Gradio)",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            color: #2563eb;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        
        # 헤더
        gr.Markdown(
            """
            # 🤖 LLM 모델 비교 시스템 v2.0
            ### Gradio로 구현된 고급 RAG 모델 성능 분석 플랫폼
            
            ---
            """, 
            elem_classes="main-header"
        )
        
        with gr.Row():
            # 왼쪽 패널 - 입력 및 설정
            with gr.Column(scale=1):
                gr.Markdown("## 📝 질문 및 설정")
                
                # 샘플 질문 드롭다운
                sample_questions = [
                    "직접 입력",
                    "미국 바이든 대통령이 몇년 몇월 몇일에 연방정부 차원에서 안전하고 신뢰할 수 있는 AI 개발과 사용을 보장하기 위한 행정명령을 발표했나요?",
                    "AI 안전성 정상회의에 참가한 28개국들이 AI 안전 보장을 위한 협력 방안을 담은 블레츨리 선언을 발표한 나라는 어디인가요?",
                    "구글이 앤스로픽에 투자한 금액은 총 얼마인가요?",
                    "삼성전자가 자체 개발한 생성 AI 모델의 이름은 무엇인가요?",
                    "갈릴레오의 LLM 환각 지수 평가에서 가장 우수한 성능을 보인 모델은 무엇인가요?"
                ]
                
                question_dropdown = gr.Dropdown(
                    choices=sample_questions,
                    value="직접 입력",
                    label="📋 샘플 질문 선택",
                    info="미리 준비된 질문을 선택하거나 '직접 입력'을 선택하세요"
                )
                
                question_input = gr.Textbox(
                    label="💬 질문 입력",
                    placeholder="RAG 시스템에 물어볼 질문을 입력하세요...",
                    lines=4,
                    value=""
                )
                
                # 질문 드롭다운 변경 시 텍스트박스 업데이트
                def update_question(selected):
                    if selected == "직접 입력":
                        return gr.update(value="", interactive=True)
                    else:
                        return gr.update(value=selected, interactive=False)
                
                question_dropdown.change(
                    update_question,
                    inputs=[question_dropdown],
                    outputs=[question_input]
                )
                
                # 모델 선택
                model_selection = gr.CheckboxGroup(
                    choices=["GPT-4o", "Claude-3.5-Haiku"],
                    value=["GPT-4o", "Claude-3.5-Haiku"],
                    label="🤖 비교 모델 선택",
                    info="비교할 모델들을 선택하세요"
                )
                
                # Temperature 설정
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="🌡️ Temperature",
                    info="응답의 창의성을 조절합니다"
                )
                
                # 실행 버튼
                compare_btn = gr.Button(
                    "🚀 모델 비교 시작",
                    variant="primary",
                    size="lg"
                )
            
            # 오른쪽 패널 - 시스템 정보
            with gr.Column(scale=1):
                gr.Markdown("## ℹ️ 시스템 정보")
                system_info = gr.Markdown(get_pipeline_info())
        
        # 결과 표시 영역
        gr.Markdown("---")
        gr.Markdown("## 📊 비교 결과")
        
        with gr.Row():
            # 텍스트 결과
            with gr.Column(scale=2):
                result_output = gr.Markdown(
                    "결과가 여기에 표시됩니다...",
                    label="📋 상세 결과"
                )
            
            # 차트 영역
            with gr.Column(scale=1):
                chart_time = gr.Plot(
                    label="⏱️ 응답 시간 차트"
                )
                chart_length = gr.Plot(
                    label="📏 응답 길이 차트"
                )
        
        # 버튼 클릭 이벤트
        compare_btn.click(
            compare_models,
            inputs=[question_input, model_selection, temperature],
            outputs=[result_output, chart_time, chart_length]
        )
        
        # 푸터
        gr.Markdown(
            """
            ---
            
            **🔧 시스템 정보**
            - 버전: v2.0 (Gradio)
            - 지원 모델: GPT-4o, Claude-3.5-Haiku
            - RAG 엔진: LangChain + FAISS + BM25
            - 추적 시스템: LangSmith
            
            **💡 팁**: 다양한 질문과 Temperature 설정으로 모델 특성을 비교해보세요!
            """,
            elem_classes="footer"
        )
    
    return demo

if __name__ == "__main__":
    # Gradio 앱 실행
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True,
        show_error=True,
        debug=True
    )