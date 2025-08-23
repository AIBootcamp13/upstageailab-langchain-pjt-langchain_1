#!/usr/bin/env python3
"""
법률 도메인 Gradio 웹 인터페이스 v5.0 (단순화 버전)
JSON 판례 데이터를 읽어서 법률 질문에 직접 답변하는 시스템
"""

import os
import json
import time
import gradio as gr
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# OpenAI 라이브러리 직접 사용
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI 라이브러리를 설치하세요: pip install openai")

try:
    import anthropic
except ImportError:
    print("Anthropic 라이브러리를 설치하세요: pip install anthropic")

# 글로벌 변수
law_documents = []
openai_client = None
anthropic_client = None

def initialize_law_system():
    """법률 시스템 초기화"""
    global law_documents, openai_client, anthropic_client
    
    load_dotenv()
    
    # 클라이언트 초기화
    if os.getenv("OPENAI_API_KEY"):
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if os.getenv("ANTHROPIC_API_KEY"):
        anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # 법률 문서 로드
    law_data_dir = Path("data/law")
    
    if not law_data_dir.exists():
        return "❌ data/law 디렉토리를 찾을 수 없습니다."
    
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
            print(f"파일 로드 오류 {json_file}: {e}")
            continue
    
    return f"✅ 법률 판례 {len(law_documents)}건 로드 완료"

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

def get_law_case_info():
    """로드된 법률 판례 정보 조회"""
    global law_documents
    
    if not law_documents:
        return "판례 데이터가 로드되지 않았습니다."
    
    info = f"📚 로드된 판례: {len(law_documents)}건\n\n"
    
    # 사건 유형별 분포
    case_types = [doc['metadata'].get('case_type', 'Unknown') for doc in law_documents]
    case_type_counts = {case_type: case_types.count(case_type) for case_type in set(case_types)}
    
    info += "**사건 유형별 분포**\n"
    for case_type, count in case_type_counts.items():
        info += f"- {case_type}: {count}건\n"
    
    info += "\n**최근 판례 샘플**\n"
    for i, doc in enumerate(law_documents[:5]):
        metadata = doc['metadata']
        info += f"{i+1}. {metadata.get('case_number', 'N/A')} - {metadata.get('case_name', 'N/A')} ({metadata.get('date', 'N/A')})\n"
    
    return info

def search_relevant_cases(question: str, top_k: int = 3) -> list:
    """질문과 관련된 판례 검색 (간단한 키워드 매칭)"""
    global law_documents
    
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

def get_gpt_response(question: str, context: str, temperature: float) -> dict:
    """GPT-4o 응답 생성"""
    global openai_client
    
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

def get_claude_response(question: str, context: str, temperature: float) -> dict:
    """Claude-3.5-Haiku 응답 생성"""
    global anthropic_client
    
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

def analyze_law_question(question, model1_enabled, model2_enabled, temperature):
    """법률 질문 분석"""
    global law_documents
    
    if not question.strip():
        return "질문을 입력해주세요.", "", "", ""
    
    if not law_documents:
        return "법률 판례 데이터가 로드되지 않았습니다.", "", "", ""
    
    if not model1_enabled and not model2_enabled:
        return "최소 하나의 모델을 선택해주세요.", "", "", ""
    
    # 관련 판례 검색
    relevant_cases = search_relevant_cases(question, top_k=3)
    
    if not relevant_cases:
        context = "관련 판례를 찾을 수 없습니다. 일반적인 법률 지식으로 답변하겠습니다."
    else:
        context = "\n\n".join([case['content'] for case in relevant_cases])
    
    results = {}
    
    # GPT-4o 실행
    if model1_enabled:
        results['GPT-4o'] = get_gpt_response(question, context, temperature)
    
    # Claude-3.5-Haiku 실행
    if model2_enabled:
        results['Claude-3.5-Haiku'] = get_claude_response(question, context, temperature)
    
    # 결과 포맷팅
    summary = f"📊 **분석 완료** (총 {len(results)}개 모델, 관련 판례 {len(relevant_cases)}건)\n\n"
    
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
    
    # 비교 차트 생성 (Gradio Plot 사용)
    chart_plot = None
    if len(results) > 1:
        import matplotlib.pyplot as plt
        
        models = list(results.keys())
        times = [results[model]['response_time'] for model in models]
        colors = ['#3498db', '#e74c3c']
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(models, times, color=colors[:len(models)])
        
        # 막대 위에 시간 표시
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{time_val:.2f}s', ha='center', va='bottom')
        
        ax.set_title('모델별 응답 시간 비교', fontsize=14, fontweight='bold')
        ax.set_xlabel('모델', fontsize=12)
        ax.set_ylabel('응답 시간 (초)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 한글 폰트 설정 (선택사항)
        try:
            plt.rcParams['font.family'] = 'DejaVu Sans'
        except:
            pass
            
        plt.tight_layout()
        chart_plot = fig
        plt.close()
    
    return summary, gpt_result, claude_result, chart_plot

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    # 초기화
    init_message = initialize_law_system()
    case_info = get_law_case_info()
    
    with gr.Blocks(title="⚖️ 법률 AI 분석 시스템 v5.0 (Gradio Simple)", theme=gr.themes.Soft()) as interface:
        
        # 헤더
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #2c3e50, #3498db); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>⚖️ 법률 AI 분석 시스템 v5.0</h1>
            <p>17개 대법원 판례 기반 • GPT-4o vs Claude-3.5-Haiku 비교 • 단순화 버전</p>
        </div>
        """)
        
        # 시스템 상태
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 시스템 상태")
                gr.Textbox(
                    label="초기화 상태",
                    value=init_message,
                    interactive=False,
                    lines=2
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### 📚 판례 정보")
                gr.Textbox(
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
                        "부당해고 구제신청의 요건과 절차는 어떻게 되나요?",
                        "근로자의 업무상 재해 인정 기준은 무엇인가요?"
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
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.1
                )
                
                analyze_btn = gr.Button(
                    "🔍 법률 분석 시작",
                    variant="primary",
                    size="lg"
                )
        
        gr.Markdown("---")
        
        # 결과 출력
        gr.Markdown("### 📊 분석 결과")
        
        summary_output = gr.Markdown(label="분석 요약")
        
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
        
        gr.Markdown("#### 📈 성능 비교 차트")
        chart_output = gr.Plot()
        
        # 분석 버튼 클릭 이벤트
        analyze_btn.click(
            analyze_law_question,
            inputs=[question_input, model1_enabled, model2_enabled, temperature],
            outputs=[summary_output, gpt_output, claude_output, chart_output]
        )
        
        # 푸터
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <p><strong>⚖️ 법률 AI 분석 시스템 v5.0 (Simple)</strong></p>
            <p>🔬 Powered by OpenAI • Anthropic • Gradio</p>
            <p>📚 17개 대법원 판례 키워드 기반 검색 시스템</p>
        </div>
        """)
    
    return interface

def main():
    """메인 실행 함수"""
    
    print("⚖️ 법률 AI 분석 시스템 v5.0 (Gradio Simple) 시작 중...")
    
    # Gradio 인터페이스 생성 및 실행
    interface = create_gradio_interface()
    
    # 인터페이스 실행
    interface.launch(
        server_name="0.0.0.0",
        server_port=7863,  # 다른 포트 사용
        share=False,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()