#!/usr/bin/env python3
"""
RAG 성능 개선 비교 Gradio 인터페이스 v08231820
완벽한 RAG 성능 비교 시스템의 Gradio 웹 인터페이스
실시간 분석, 진행률 표시, 대화형 차트, LangSmith 추적 통합
"""

import os
import json
import time
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 시스템에서 사용 가능한 한글 폰트 찾기
def setup_korean_font():
    """한글 폰트 설정"""
    font_candidates = [
        'NanumGothic', 'NanumBarunGothic', 'Malgun Gothic', 
        'Arial Unicode MS', 'AppleGothic', 'Gulim'
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams['font.family'] = [font]
            return font
    
    # 폰트를 찾지 못한 경우 기본 설정
    plt.rcParams['font.family'] = ['DejaVu Sans']
    return 'DejaVu Sans'

# 한글 폰트 초기화
current_font = setup_korean_font()
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리를 Python path에 추가
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.rag_improvement_complete_08231820 import RAGImprovementComparator, save_results_multiple_formats
from src.utils.version_manager import VersionManager
from src.utils.langsmith_simple import LangSmithSimple
from src.utils.path_utils import ensure_directory_exists

# 글로벌 변수
rag_comparator = None
version_manager = None
analysis_results = None

def initialize_system():
    """시스템 초기화"""
    global rag_comparator, version_manager
    
    load_dotenv()
    
    # 버전 관리자 초기화
    version_manager = VersionManager()
    version_manager.logger.info("Gradio RAG 완벽 분석 시스템 초기화")
    
    # LangSmith 설정
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'langsmith': {
            'enabled': True,
            'project_name': 'gradio-rag-complete-v08231820',
            'session_name': f'gradio-session_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    })
    
    langsmith_manager = LangSmithSimple(cfg, version_manager)
    
    # 비교기 초기화
    rag_comparator = RAGImprovementComparator(version_manager, langsmith_manager)
    
    return "✅ RAG 성능 분석 시스템 초기화 완료 (v08231820 - LangSmith 추적 활성화)"

def get_system_info():
    """시스템 정보 조회"""
    if not rag_comparator:
        return "❌ 시스템이 초기화되지 않았습니다."
    
    # 판례 로드
    rag_comparator.case_loader.load_cases()
    cases_count = len(rag_comparator.case_loader.cases)
    
    # API 클라이언트 상태 확인
    openai_status = "✅ 연결됨" if rag_comparator.openai_client else "❌ 연결 실패"
    anthropic_status = "✅ 연결됨" if rag_comparator.anthropic_client else "❌ 연결 실패"
    
    info = f"""
📊 **시스템 상태**

- **로드된 판례**: {cases_count}건
- **OpenAI 연결**: {openai_status}
- **Anthropic 연결**: {anthropic_status}
- **LangSmith**: {"✅ 활성화" if rag_comparator.langsmith_manager else "❌ 비활성화"}

📈 **분석 기능**
- 순수 LLM vs RAG 성능 비교
- GPT-4o와 Claude-3.5 동시 분석
- 실시간 진행률 표시
- 다중 형식 결과 저장 (JSON/CSV/MD)
"""
    
    return info

def run_rag_analysis(questions_text, temperature, progress=gr.Progress()):
    """RAG 성능 분석 실행"""
    global analysis_results
    
    if not rag_comparator:
        return "❌ 시스템이 초기화되지 않았습니다. 먼저 시스템을 초기화해주세요.", None, None, None
    
    if not questions_text.strip():
        return "❌ 질문을 입력해주세요. 기본 질문이 제공되어 있으니 그대로 사용하거나 직접 수정하세요.", None, None, None
    
    # 질문 파싱
    questions = [q.strip() for q in questions_text.strip().split('\n') if q.strip()]
    
    if not questions:
        return "❌ 유효한 질문이 없습니다.", None, None, None
    
    if len(questions) > 5:
        return "❌ 최대 5개 질문까지 분석 가능합니다.", None, None, None
    
    try:
        progress(0, "분석 준비 중...")
        
        # 진행률 콜백 함수
        def progress_callback(p):
            if p < 1.0:
                progress(p, f"분석 진행 중... ({p*100:.1f}%)")
            else:
                progress(1.0, "✅ 분석 완료!")
        
        # 실제 분석 실행
        results = rag_comparator.compare_models(questions, temperature, progress_callback)
        analysis_results = results
        
        # 결과 요약 생성
        summary = generate_analysis_summary(results)
        
        # 차트 생성
        improvement_chart = create_improvement_chart(results)
        response_time_chart = create_response_time_chart(results)
        performance_radar = create_performance_radar(results)
        
        return summary, improvement_chart, response_time_chart, performance_radar
        
    except Exception as e:
        error_msg = f"❌ 분석 중 오류 발생: {str(e)}"
        if version_manager:
            version_manager.logger.error(f"Gradio 분석 오류: {e}")
        return error_msg, None, None, None

def generate_analysis_summary(results):
    """분석 결과 요약 생성"""
    summary = results.get('summary', {})
    model_averages = summary.get('model_averages', {})
    questions_count = len(results.get('questions', {}))
    
    report = f"""
# 🧠 RAG 성능 개선 분석 결과

**분석 완료 시각**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**분석 질문 수**: {questions_count}개
**참조 판례 수**: {results.get('metadata', {}).get('total_cases', 0)}건

## 📋 평가 기준 및 방법론

### 🎯 RAG 개선 점수 계산 방법
RAG 개선 점수는 다음 4가지 요소를 종합하여 0-100점으로 산출됩니다:

1. **판례 인용 개선도 (40점)**: RAG 적용 시 판례 번호 인용 증가량
   - 순수 LLM: 판례 번호 언급 부족 시 감점
   - RAG 적용: 정확한 판례 번호 인용 시 가점

2. **법률 키워드 밀도 (30점)**: 전문적인 법률 용어 사용 빈도
   - 법조문, 판례, 조항, 근로기준법, 민법, 상법 등의 키워드 분석
   - RAG를 통해 더 정확하고 전문적인 용어 사용 평가

3. **답변 내용 충실성 (20점)**: 답변 길이 및 상세성 개선
   - 순수 LLM 대비 RAG 적용 시 답변 길이 증가율
   - 최대 50% 증가까지 만점, 그 이상은 감점

4. **처리 시간 효율성 (10점)**: 응답 속도 고려
   - 시간 증가가 적을수록 높은 점수
   - 과도한 시간 증가 시 효율성 감점

### 📈 성능 등급 기준
- 🏆 **우수** (80-100점): RAG 적용 효과가 매우 뛰어남
- 👍 **양호** (60-79점): RAG 적용 효과가 양호함
- ⚠️ **보통** (40-59점): RAG 적용 효과가 보통 수준
- ❌ **부족** (0-39점): RAG 적용 효과가 부족함

## 📊 모델별 성능 요약

"""
    
    if model_averages:
        for model, avg_data in model_averages.items():
            improvement = avg_data.get('avg_improvement_score', 0)
            time_change = avg_data.get('avg_time_increase', 0)
            
            # 성능 등급 결정
            if improvement >= 80:
                grade = "🏆 우수"
                color = "🟢"
            elif improvement >= 60:
                grade = "👍 양호"
                color = "🟡"
            elif improvement >= 40:
                grade = "⚠️ 보통"
                color = "🟠"
            else:
                grade = "❌ 부족"
                color = "🔴"
            
            report += f"""
### {color} {model} {grade}
- **평균 개선 점수**: {improvement:.1f}/100점
- **처리 시간 변화**: {time_change:+.2f}초
- **평균 활용 판례**: {avg_data.get('avg_cases_used', 0):.1f}건
- **답변 길이 증가**: {avg_data.get('avg_length_increase', 0):+.0f}글자
"""
    
    # 성능 비교
    perf_comp = summary.get('performance_comparison', {})
    if perf_comp:
        report += f"""
## 🏁 모델간 비교

- **더 나은 RAG 개선**: {perf_comp.get('better_improvement', 'N/A')}
- **더 빠른 처리 속도**: {perf_comp.get('faster_processing', 'N/A')}
- **성능 점수 차이**: {perf_comp.get('score_difference', 0):.1f}점

## 💡 주요 발견사항

"""
        
        # 주요 인사이트 생성
        if 'GPT-4o' in model_averages and 'Claude-3.5' in model_averages:
            gpt_score = model_averages['GPT-4o'].get('avg_improvement_score', 0)
            claude_score = model_averages['Claude-3.5'].get('avg_improvement_score', 0)
            
            if abs(gpt_score - claude_score) < 5:
                report += "- 두 모델의 RAG 개선 효과가 비슷합니다.\n"
            elif gpt_score > claude_score:
                report += "- GPT-4o가 RAG 적용에서 더 우수한 성능을 보입니다.\n"
            else:
                report += "- Claude-3.5가 RAG 적용에서 더 우수한 성능을 보입니다.\n"
            
            gpt_time = model_averages['GPT-4o'].get('avg_time_increase', 0)
            claude_time = model_averages['Claude-3.5'].get('avg_time_increase', 0)
            
            if gpt_time < claude_time:
                report += "- GPT-4o가 더 효율적인 처리 속도를 보입니다.\n"
            else:
                report += "- Claude-3.5가 더 효율적인 처리 속도를 보입니다.\n"
    
    # 상세 질문별 평가 내역 추가 (개선 전후 답변 포함)
    questions_data = results.get('questions', {})
    if questions_data:
        report += f"""

## 📝 질문별 상세 분석 및 개선 전후 비교

"""
        for q_id, q_data in questions_data.items():
            q_number = q_id[-1] if len(q_id) > 0 else "1"
            q_text = q_data.get('question', '질문 내용을 불러올 수 없습니다')
            
            report += f"""
---

### 📋 질문 {q_number}

**질문**: {q_text}

"""
            
            # 모델별로 개선 전후 분석
            analysis_data = q_data.get('analysis', {})
            improvements = q_data.get('improvements', {})
            
            for model in ['GPT-4o', 'Claude-3.5']:
                if model in analysis_data and model in improvements:
                    model_analysis = analysis_data[model]
                    model_improvement = improvements[model]
                    
                    if isinstance(model_improvement, dict):
                        # 순수 LLM 응답
                        pure_response = model_analysis.get('pure_response', {})
                        pure_answer = pure_response.get('response', '답변을 불러올 수 없습니다')
                        pure_time = pure_response.get('response_time', 0)
                        
                        # RAG 적용 응답
                        rag_response = model_analysis.get('rag_response', {})
                        rag_answer = rag_response.get('response', '답변을 불러올 수 없습니다')
                        rag_time = rag_response.get('response_time', 0)
                        used_cases = rag_response.get('cases_used', 0)
                        
                        # 평가 점수들
                        overall_score = model_improvement.get('overall_score', 0)
                        case_citation = model_improvement.get('case_citation_score', 0)
                        keyword_score = model_improvement.get('keyword_density_score', 0)
                        length_score = model_improvement.get('length_score', 0)
                        time_score = model_improvement.get('time_efficiency_score', 0)
                        
                        report += f"""
#### 🤖 {model} 모델 분석

##### 📝 개선 전 (순수 LLM) 답변:
```
{pure_answer[:500]}{"..." if len(pure_answer) > 500 else ""}
```
- **응답 시간**: {pure_time:.2f}초
- **답변 길이**: {len(pure_answer)}글자

##### 🚀 개선 후 (RAG 적용) 답변:
```
{rag_answer[:500]}{"..." if len(rag_answer) > 500 else ""}
```
- **응답 시간**: {rag_time:.2f}초
- **답변 길이**: {len(rag_answer)}글자
- **활용 판례**: {used_cases}건

##### 📊 세부 평가 점수:
| 평가 기준 | 점수 | 만점 | 설명 |
|-----------|------|------|------|
| 📚 **판례 인용 개선도** | **{case_citation:.1f}점** | 40점 | RAG 적용 시 판례 번호 인용 증가량 |
| 🔑 **법률 키워드 밀도** | **{keyword_score:.1f}점** | 30점 | 전문적인 법률 용어 사용 빈도 |
| 📄 **답변 내용 충실성** | **{length_score:.1f}점** | 20점 | 답변 길이 및 상세성 개선 |
| ⚡ **처리 시간 효율성** | **{time_score:.1f}점** | 10점 | 응답 속도 대비 효율성 |
| 🎯 **종합 개선 점수** | **{overall_score:.1f}점** | **100점** | **전체 RAG 개선 효과** |

---

"""

        # 최종 모델별 평균 개선 점수 비교
        model_averages = summary.get('model_averages', {})
        if model_averages:
            report += f"""

## 🏆 최종 LLM별 성능 비교 및 평균 개선 점수

"""
            models_scores = []
            for model, avg_data in model_averages.items():
                avg_score = avg_data.get('avg_improvement_score', 0)
                avg_time = avg_data.get('avg_time_increase', 0)
                avg_cases = avg_data.get('avg_cases_used', 0)
                avg_length = avg_data.get('avg_length_increase', 0)
                
                models_scores.append((model, avg_score))
                
                # 성능 등급 결정
                if avg_score >= 80:
                    grade = "🏆 우수"
                    grade_color = "🟢"
                elif avg_score >= 60:
                    grade = "👍 양호"
                    grade_color = "🟡"
                elif avg_score >= 40:
                    grade = "⚠️ 보통"
                    grade_color = "🟠"
                else:
                    grade = "❌ 부족"
                    grade_color = "🔴"
                
                report += f"""
### {grade_color} {model} {grade}

| 항목 | 값 | 설명 |
|------|----|----- |
| 🎯 **평균 개선 점수** | **{avg_score:.1f}/100점** | **전체 질문 평균 RAG 개선 효과** |
| 🕐 **평균 시간 증가** | {avg_time:+.2f}초 | RAG 적용으로 인한 응답 시간 변화 |
| 📚 **평균 활용 판례** | {avg_cases:.1f}건 | 질문당 평균 사용된 판례 수 |
| 📄 **평균 답변 증가** | {avg_length:+.0f}글자 | 순수 LLM 대비 답변 길이 변화 |

"""
            
            # 승부 결과
            if len(models_scores) >= 2:
                models_scores.sort(key=lambda x: x[1], reverse=True)
                winner = models_scores[0]
                runner_up = models_scores[1]
                score_diff = winner[1] - runner_up[1]
                
                report += f"""
### 🥇 최종 순위 및 승부 결과

1. 🥇 **1위: {winner[0]}** - {winner[1]:.1f}점
2. 🥈 **2위: {runner_up[0]}** - {runner_up[1]:.1f}점

**점수 차이**: {score_diff:.1f}점

"""
                if score_diff > 10:
                    report += f"**결론**: {winner[0]}가 RAG 적용에서 **확실히 우수한** 성능을 보입니다.\n\n"
                elif score_diff > 5:
                    report += f"**결론**: {winner[0]}가 RAG 적용에서 **약간 더 나은** 성능을 보입니다.\n\n"
                else:
                    report += f"**결론**: 두 모델의 RAG 적용 성능이 **비슷한 수준**입니다.\n\n"
    
    return report

def create_improvement_chart(results):
    """RAG 개선 점수 차트 생성"""
    if not results or not results.get('questions'):
        return None
    
    data = []
    questions = []
    models = ['GPT-4o', 'Claude-3.5']
    
    for q_id, q_data in results['questions'].items():
        questions.append(f"Q{q_id[-1]}")
        for model in models:
            if model in q_data.get('improvements', {}):
                improvement = q_data['improvements'][model]
                data.append(improvement['overall_score'])
            else:
                data.append(0)
    
    if not data:
        return None
    
    # 데이터 재구성
    gpt_scores = data[::2]  # 짝수 인덱스
    claude_scores = data[1::2]  # 홀수 인덱스
    
    x = np.arange(len(questions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, gpt_scores, width, label='GPT-4o', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, claude_scores, width, label='Claude-3.5', color='#e74c3c', alpha=0.8)
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}', ha='center', va='bottom')
    
    ax.set_xlabel('Questions')
    ax.set_ylabel('RAG Improvement Score (0-100)')
    ax.set_title('RAG Performance Improvement Score by Question')
    ax.set_xticks(x)
    ax.set_xticklabels(questions)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_response_time_chart(results):
    """응답 시간 비교 차트 생성"""
    if not results or not results.get('questions'):
        return None
    
    pure_times = {'GPT-4o': [], 'Claude-3.5': []}
    rag_times = {'GPT-4o': [], 'Claude-3.5': []}
    questions = []
    
    for q_id, q_data in results['questions'].items():
        questions.append(f"Q{q_id[-1]}")
        
        for model in ['GPT-4o', 'Claude-3.5']:
            if model in q_data.get('responses', {}):
                responses = q_data['responses'][model]
                pure_times[model].append(responses.get('pure', {}).get('response_time', 0))
                rag_times[model].append(responses.get('rag', {}).get('response_time', 0))
            else:
                pure_times[model].append(0)
                rag_times[model].append(0)
    
    if not any(pure_times.values()):
        return None
    
    x = np.arange(len(questions))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 각 모델의 순수/RAG 시간
    ax.bar(x - 1.5*width, pure_times['GPT-4o'], width, label='GPT-4o (Pure)', color='lightblue', alpha=0.7)
    ax.bar(x - 0.5*width, rag_times['GPT-4o'], width, label='GPT-4o (RAG)', color='#3498db')
    ax.bar(x + 0.5*width, pure_times['Claude-3.5'], width, label='Claude-3.5 (Pure)', color='lightcoral', alpha=0.7)
    ax.bar(x + 1.5*width, rag_times['Claude-3.5'], width, label='Claude-3.5 (RAG)', color='#e74c3c')
    
    ax.set_xlabel('Questions')
    ax.set_ylabel('Response Time (seconds)')
    ax.set_title('Pure LLM vs RAG-Applied Response Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(questions)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_performance_radar(results):
    """성능 레이더 차트 생성"""
    if not results or not results.get('summary', {}).get('model_averages'):
        return None
    
    model_averages = results['summary']['model_averages']
    models = list(model_averages.keys())
    
    if len(models) < 2:
        return None
    
    # 성능 메트릭 (정규화)
    metrics = ['Improvement', 'Efficiency', 'Accuracy', 'Speed', 'Utilization']
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 닫힌 다각형
    
    colors = ['#3498db', '#e74c3c']
    
    for i, model in enumerate(models):
        avg_data = model_averages[model]
        
        # 메트릭 값 정규화 (0-100)
        values = [
            avg_data.get('avg_improvement_score', 0),  # 개선점수
            max(0, 100 - abs(avg_data.get('avg_time_increase', 0)) * 10),  # 효율성
            min(100, avg_data.get('avg_improvement_score', 0) * 1.1),  # 정확성
            max(0, 100 - avg_data.get('avg_time_increase', 0) * 15),  # 속도
            min(100, avg_data.get('avg_cases_used', 0) * 20)  # 활용도
        ]
        
        values = np.concatenate((values, [values[0]]))  # 닫힌 다각형
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.set_title('모델별 종합 성능 비교 (레이더 차트)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    return fig

def save_analysis_results():
    """분석 결과 저장"""
    global analysis_results
    
    if not analysis_results:
        return "❌ 저장할 분석 결과가 없습니다. 먼저 분석을 실행해주세요."
    
    try:
        output_dir = ensure_directory_exists("results/rag_improvement_complete")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_path, csv_path, report_path = save_results_multiple_formats(
            analysis_results, Path(output_dir), timestamp
        )
        
        success_msg = f"""
✅ **분석 결과 저장 완료!**

📄 **JSON 데이터**: `{json_path.name}`
📊 **CSV 요약**: `{csv_path.name}`  
📋 **분석 보고서**: `{report_path.name}`

💾 **저장 위치**: `{output_dir}`
"""
        
        if version_manager:
            version_manager.logger.info(f"Gradio 분석 결과 저장 완료: {json_path}")
        
        return success_msg
        
    except Exception as e:
        error_msg = f"❌ 저장 중 오류 발생: {str(e)}"
        if version_manager:
            version_manager.logger.error(f"Gradio 결과 저장 오류: {e}")
        return error_msg

def get_detailed_results():
    """상세 결과 조회"""
    global analysis_results
    
    if not analysis_results:
        return "❌ 조회할 분석 결과가 없습니다."
    
    detailed_report = "# 📋 상세 분석 결과\n\n"
    
    for q_id, q_data in analysis_results.get('questions', {}).items():
        detailed_report += f"## {q_id.upper()}. {q_data['question']}\n\n"
        
        for model in ['GPT-4o', 'Claude-3.5']:
            if model in q_data.get('improvements', {}):
                improvement = q_data['improvements'][model]
                responses = q_data['responses'][model]
                
                detailed_report += f"### {model} 상세 분석\n"
                detailed_report += f"- **개선 점수**: {improvement['overall_score']:.1f}/100\n"
                detailed_report += f"- **분석 내용**: {improvement['analysis']}\n"
                detailed_report += f"- **응답시간 변화**: {improvement['response_time_change']:+.2f}초\n"
                detailed_report += f"- **사용 판례 수**: {responses['rag'].get('case_count', 0)}건\n"
                detailed_report += f"- **참조 판례**: {', '.join(responses['rag'].get('cases_used', []))}\n\n"
                
                detailed_report += f"**순수 {model} 답변**:\n"
                detailed_report += f"```\n{responses['pure']['answer'][:200]}...\n```\n\n"
                
                detailed_report += f"**RAG 적용 {model} 답변**:\n"
                detailed_report += f"```\n{responses['rag']['answer'][:200]}...\n```\n\n"
                
                detailed_report += "---\n\n"
    
    return detailed_report

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    # 시스템 초기화
    init_status = initialize_system()
    system_info = get_system_info()
    
    with gr.Blocks(
        title="🧠 RAG 성능 개선 완벽 분석 시스템 v08231820",
        theme=gr.themes.Soft()
    ) as interface:
        
        # 헤더
        gr.HTML("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-radius: 15px; margin-bottom: 20px;">
            <h1>🧠 RAG 성능 개선 완벽 분석 시스템</h1>
            <h3>v08231820 • LangSmith 추적 • 실시간 시각화 • Gradio Framework</h3>
            <p>순수 LLM vs RAG 적용 성능 비교 • GPT-4o • Claude-3.5 • 17개 대법원 판례</p>
        </div>
        """)
        
        # 시스템 상태
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 시스템 상태")
                gr.Textbox(
                    label="초기화 상태",
                    value=init_status,
                    interactive=False,
                    lines=2
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### ℹ️ 시스템 정보")
                gr.Markdown(system_info)
        
        gr.Markdown("---")
        
        # 메인 분석 인터페이스
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📝 분석 질문 입력")
                
                questions_input = gr.Textbox(
                    label="질문 목록 (줄바꿈으로 구분)",
                    value="""취업규칙을 근로자에게 불리하게 변경할 때 사용자가 지켜야 할 법적 요건은 무엇인가요?
퇴직금 지급 기일을 연장하는 합의를 했더라도 연장된 기일까지 지급하지 않으면 형사처벌을 받나요?
부당해고 구제신청의 요건과 절차는 어떻게 되나요?""",
                    placeholder="질문을 한 줄씩 입력하세요",
                    lines=6
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 분석 설정")
                
                temperature = gr.Slider(
                    label="Temperature (창의성 조절)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.05
                )
                
                analyze_btn = gr.Button(
                    "🚀 RAG 성능 분석 시작",
                    variant="primary",
                    size="lg"
                )
                
                save_btn = gr.Button(
                    "💾 결과 저장",
                    variant="secondary",
                    size="lg"
                )
        
        gr.Markdown("---")
        
        # 결과 출력
        gr.Markdown("### 📊 분석 결과")
        
        # 요약 결과
        summary_output = gr.Markdown(label="분석 요약")
        
        # 차트 탭
        with gr.Tabs():
            with gr.TabItem("📈 개선 점수"):
                improvement_chart = gr.Plot(label="RAG 개선 점수 비교")
            
            with gr.TabItem("⏱️ 응답 시간"):
                response_time_chart = gr.Plot(label="응답 시간 비교")
            
            with gr.TabItem("🎯 종합 성능"):
                performance_radar = gr.Plot(label="모델별 종합 성능")
            
            with gr.TabItem("📋 상세 결과"):
                detailed_results = gr.Markdown(label="상세 분석 결과")
                
                detail_btn = gr.Button("🔍 상세 결과 조회")
        
        # 저장 결과 표시
        save_status = gr.Markdown(label="저장 상태")
        
        # 이벤트 핸들러
        analyze_btn.click(
            run_rag_analysis,
            inputs=[questions_input, temperature],
            outputs=[summary_output, improvement_chart, response_time_chart, performance_radar]
        )
        
        save_btn.click(
            save_analysis_results,
            outputs=[save_status]
        )
        
        detail_btn.click(
            get_detailed_results,
            outputs=[detailed_results]
        )
        
        # 푸터
        gr.HTML("""
        <div style="text-align: center; margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <p><strong>🧠 RAG 성능 개선 완벽 분석 시스템 v08231820</strong></p>
            <p>🔬 Powered by LangChain • OpenAI • Anthropic • LangSmith • Gradio</p>
            <p>⚖️ 17개 대법원 판례 기반 RAG 성능 검증 • 실시간 분석 • 다중 형식 출력</p>
            <p>🎯 순수 LLM 대비 RAG 적용 효과 정량적 측정 및 시각화</p>
        </div>
        """)
    
    return interface

def main():
    """메인 실행 함수"""
    
    print("🧠 RAG 성능 개선 완벽 분석 시스템 v08231820 (Gradio) 시작 중...")
    
    # Gradio 인터페이스 생성 및 실행
    interface = create_gradio_interface()
    
    # 인터페이스 실행
    interface.launch(
        server_name="0.0.0.0",
        server_port=7864,  # 기존 포트와 구분
        share=False,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()