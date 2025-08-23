#!/usr/bin/env python3
"""
법률 정확성 기반 RAG 비교 Gradio 인터페이스 v08240001
- 법조문 인용 정확성 최우선 (50점)
- 완전 투명한 점수 산출 과정
- 반자동화: AI 분석 + 사람 검증 필요
- 개선 전후 답변 및 세부 평가 점수 표시
"""

import os
import json
import time
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

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

# 프로젝트 루트 디렉토리를 Python path에 추가
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.rag_improvement_legal_accuracy_08240001 import RAGLegalAccuracyComparator, save_results_multiple_formats
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
    
    try:
        # 환경 변수 로드
        load_dotenv()
        
        # 버전 관리자 초기화
        version_manager = VersionManager()
        version_manager.logger.info("법률 정확성 Gradio 시스템 초기화")
        
        # LangSmith 설정
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            'langsmith': {
                'enabled': True,
                'project_name': 'gradio-legal-accuracy-v08240001',
                'session_name': f'gradio-session_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            }
        })
        
        langsmith_manager = LangSmithSimple(cfg, version_manager)
        
        # RAG 비교기 초기화
        rag_comparator = RAGLegalAccuracyComparator(version_manager, langsmith_manager)
        
        return "✅ 법률 정확성 기반 RAG 시스템이 성공적으로 초기화되었습니다.", gr.Column(visible=True)
        
    except Exception as e:
        error_msg = f"❌ 시스템 초기화 실패: {str(e)}"
        if version_manager:
            version_manager.logger.error(f"Gradio 초기화 오류: {e}")
        return error_msg, gr.Column(visible=False)

def run_legal_accuracy_analysis(questions_text, temperature, progress=gr.Progress()):
    """법률 정확성 기반 RAG 분석 실행"""
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
        progress(0, "법률 정확성 분석 준비 중...")
        
        # 진행률 콜백 함수
        def progress_callback(p):
            if p < 1.0:
                progress(p, f"법률 정확성 분석 진행 중... ({p*100:.1f}%)")
            else:
                progress(1.0, "✅ 법률 정확성 분석 완료!")
        
        # 실제 분석 실행
        results = rag_comparator.compare_models(questions, temperature, progress_callback)
        analysis_results = results
        
        # 결과 요약 생성
        summary = generate_legal_accuracy_summary(results)
        
        # 차트 생성
        improvement_chart = create_legal_accuracy_chart(results)
        response_time_chart = create_response_time_chart(results)
        performance_radar = create_performance_radar(results)
        
        return summary, improvement_chart, response_time_chart, performance_radar
        
    except Exception as e:
        error_msg = f"❌ 분석 중 오류 발생: {str(e)}"
        if version_manager:
            version_manager.logger.error(f"Gradio 분석 오류: {e}")
        return error_msg, None, None, None

def generate_legal_accuracy_summary(results):
    """법률 정확성 분석 결과 요약 생성"""
    summary = results.get('summary', {})
    model_averages = summary.get('model_averages', {})
    questions_count = len(results.get('questions', {}))
    
    report = f"""
# ⚖️ 법률 정확성 기반 RAG 성능 분석 결과

**분석 완료 시각**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**분석 버전**: v08240001 (법률 정확성 중심)
**분석 질문 수**: {questions_count}개
**참조 판례 수**: {results.get('metadata', {}).get('total_cases', 0)}건

## 📋 새로운 평가 기준 (법률 전문성 중심)

### 🎯 법률 정확성 평가 방법론
이 분석은 **법조문 인용 정확성을 최우선**으로 하는 객관적 평가 시스템을 사용합니다:

| 평가 영역 | 배점 | 평가 내용 | 중요도 |
|-----------|------|-----------|--------|
| 📚 **법조문 인용 정확성** | **50점** | 정확한 조문 인용(30점) + 적용 타당성(20점) | **최우선** |
| ⚖️ **판례 적절성** | **25점** | 사안 관련성(15점) + 판시사항 정확성(10점) | 높음 |
| 🧠 **법리 논리성** | **15점** | 전제→추론→결론의 논리적 구조 | 보통 |
| 🎯 **실무 적용성** | **10점** | 구체적이고 실행 가능한 해결방안 | 보통 |

### 🔍 평가 방식의 특징
- **반자동화**: AI 분석 + 사람의 최종 검증 필요
- **완전 투명**: 모든 점수 산출 과정 상세 표시
- **정밀 평가**: 속도보다 정확성 우선

## 📊 모델별 법률 정확성 성능 요약

"""
    
    if model_averages:
        for model, avg_data in model_averages.items():
            improvement = avg_data.get('avg_improvement_score', 0)
            time_change = avg_data.get('avg_time_increase', 0)
            
            # 법률 전문성 등급 결정
            if improvement >= 80:
                grade = "🏆 법률 전문가 수준"
                color = "🟢"
            elif improvement >= 65:
                grade = "⚖️ 법률 실무 수준" 
                color = "🟡"
            elif improvement >= 50:
                grade = "📚 법률 기초 수준"
                color = "🟠"
            else:
                grade = "❌ 법률 지식 부족"
                color = "🔴"
            
            report += f"""
### {color} {model} - {grade}
- **법률 정확성 점수**: {improvement:.1f}/100점
- **처리 시간 변화**: {time_change:+.2f}초
- **평균 활용 판례**: {avg_data.get('avg_cases_used', 0):.1f}건
- **답변 길이 증가**: {avg_data.get('avg_length_increase', 0):+.0f}글자
"""
    
    # 성능 비교
    perf_comp = summary.get('performance_comparison', {})
    if perf_comp:
        report += f"""
## 🏁 모델간 법률 정확성 비교

- **더 정확한 법률 분석**: {perf_comp.get('better_improvement', 'N/A')}
- **더 빠른 처리 속도**: {perf_comp.get('faster_processing', 'N/A')}
- **법률 정확성 점수 차이**: {perf_comp.get('score_difference', 0):.1f}점

## 💡 주요 발견사항

"""
        
        # 주요 인사이트 생성
        if 'GPT-4o' in model_averages and 'Claude-3.5' in model_averages:
            gpt_score = model_averages['GPT-4o'].get('avg_improvement_score', 0)
            claude_score = model_averages['Claude-3.5'].get('avg_improvement_score', 0)
            
            if abs(gpt_score - claude_score) < 3:
                report += "- 두 모델의 법률 정확성 수준이 비슷합니다.\n"
            elif gpt_score > claude_score:
                report += "- GPT-4o가 법률 분석에서 더 정확한 성능을 보입니다.\n"
            else:
                report += "- Claude-3.5가 법률 분석에서 더 정확한 성능을 보입니다.\n"
            
            # 실용적 권고사항
            if max(gpt_score, claude_score) >= 75:
                report += "- 두 모델 모두 실제 법률 자문에 활용 가능한 수준입니다.\n"
            elif max(gpt_score, claude_score) >= 60:
                report += "- 기초적인 법률 정보 제공에는 활용 가능하나, 전문가 검토가 필요합니다.\n"
            else:
                report += "- 현재 수준으로는 법률 전문 업무에 신중하게 사용해야 합니다.\n"
    
    # 상세 질문별 분석 및 개선 전후 비교
    questions_data = results.get('questions', {})
    if questions_data:
        report += f"""

## 📝 질문별 상세 법률 정확성 분석

"""
        for q_id, q_data in questions_data.items():
            q_number = q_id[-1] if len(q_id) > 0 else "1"
            q_text = q_data.get('question', '질문 내용을 불러올 수 없습니다')
            
            report += f"""
---

### ⚖️ 질문 {q_number}

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
                        pure_answer = pure_response.get('answer', '답변을 불러올 수 없습니다')
                        pure_time = pure_response.get('response_time', 0)
                        
                        # RAG 적용 응답
                        rag_response = model_analysis.get('rag_response', {})
                        rag_answer = rag_response.get('answer', '답변을 불러올 수 없습니다')
                        rag_time = rag_response.get('response_time', 0)
                        used_cases = rag_response.get('cases_used', 0)
                        
                        # 법률 정확성 점수들
                        legal_score = model_improvement.get('legal_accuracy_score')
                        overall_score = model_improvement.get('overall_improvement', 0)
                        
                        if legal_score:
                            statute_citation = legal_score.statute_citation_accuracy
                            statute_application = legal_score.statute_application_validity
                            precedent_relevance = legal_score.precedent_relevance
                            precedent_accuracy = legal_score.precedent_accuracy
                            legal_reasoning = legal_score.legal_reasoning_logic
                            practical_applicability = legal_score.practical_applicability
                        else:
                            # 호환성을 위한 기본값
                            statute_citation = model_improvement.get('case_citation_score', 0)
                            statute_application = 0
                            precedent_relevance = model_improvement.get('keyword_density_score', 0)
                            precedent_accuracy = 0
                            legal_reasoning = model_improvement.get('length_score', 0)
                            practical_applicability = model_improvement.get('time_efficiency_score', 0)
                        
                        report += f"""
#### 🤖 {model} 모델 법률 정확성 분석

##### 📝 개선 전 (순수 LLM) 답변:
```
{pure_answer[:400]}{"..." if len(pure_answer) > 400 else ""}
```
- **응답 시간**: {pure_time:.2f}초
- **답변 길이**: {len(pure_answer)}글자

##### 🚀 개선 후 (RAG + 판례 적용) 답변:
```
{rag_answer[:400]}{"..." if len(rag_answer) > 400 else ""}
```
- **응답 시간**: {rag_time:.2f}초
- **답변 길이**: {len(rag_answer)}글자
- **활용 판례**: {used_cases}건

##### ⚖️ 법률 정확성 세부 평가 점수:

| 평가 기준 | 점수 | 만점 | 달성률 | 중요도 |
|-----------|------|------|--------|--------|
| 📚 **법조문 인용 정확성** | **{statute_citation:.1f}점** | 30점 | {statute_citation/30*100:.1f}% | 최우선 |
| ⚖️ **조문 적용 타당성** | **{statute_application:.1f}점** | 20점 | {statute_application/20*100 if statute_application > 0 else 0:.1f}% | 최우선 |
| 🏛️ **판례 사안 관련성** | **{precedent_relevance:.1f}점** | 15점 | {precedent_relevance/15*100:.1f}% | 높음 |
| 📖 **판시사항 정확성** | **{precedent_accuracy:.1f}점** | 10점 | {precedent_accuracy/10*100 if precedent_accuracy > 0 else 0:.1f}% | 높음 |
| 🧠 **법리 논리적 구조** | **{legal_reasoning:.1f}점** | 15점 | {legal_reasoning/15*100:.1f}% | 보통 |
| 🎯 **실무 적용 가능성** | **{practical_applicability:.1f}점** | 10점 | {practical_applicability/10*100:.1f}% | 보통 |
| **🏆 종합 법률 정확성** | **{overall_score:.1f}점** | **100점** | **{overall_score:.1f}%** | |

##### 👤 사람 검증 필요사항:
- 법조문 인용의 법리적 정확성 재검토
- 판례 적용의 사안별 적절성 확인
- 실무적 조언의 적법성 및 실현가능성 검증

---

"""

        # 최종 모델별 평균 법률 정확성 비교
        model_averages = summary.get('model_averages', {})
        if model_averages:
            report += f"""

## 🏆 최종 법률 정확성 순위 및 종합 평가

"""
            models_scores = []
            for model, avg_data in model_averages.items():
                avg_score = avg_data.get('avg_improvement_score', 0)
                avg_time = avg_data.get('avg_time_increase', 0)
                avg_cases = avg_data.get('avg_cases_used', 0)
                avg_length = avg_data.get('avg_length_increase', 0)
                
                models_scores.append((model, avg_score))
                
                # 법률 전문성 등급 결정
                if avg_score >= 80:
                    grade = "🏆 법률 전문가 수준"
                    grade_color = "🟢"
                    recommendation = "실제 법률 자문에 활용 가능"
                elif avg_score >= 65:
                    grade = "⚖️ 법률 실무 수준"
                    grade_color = "🟡"  
                    recommendation = "기본적인 법률 업무 지원 가능, 전문가 검토 권장"
                elif avg_score >= 50:
                    grade = "📚 법률 기초 수준"
                    grade_color = "🟠"
                    recommendation = "일반적인 법률 정보 제공만 권장, 반드시 전문가 검토 필요"
                else:
                    grade = "❌ 법률 지식 부족"
                    grade_color = "🔴"
                    recommendation = "법률 전문 업무에 부적합, 개선 필요"
                
                report += f"""
### {grade_color} {model} - {grade}

| 항목 | 값 | 설명 |
|------|----|----- |
| ⚖️ **평균 법률 정확성** | **{avg_score:.1f}/100점** | **전체 질문 평균 법률 분석 정확도** |
| 🕐 **평균 시간 증가** | {avg_time:+.2f}초 | RAG 적용으로 인한 응답 시간 변화 |
| 📚 **평균 활용 판례** | {avg_cases:.1f}건 | 질문당 평균 사용된 판례 수 |
| 📄 **평균 답변 증가** | {avg_length:+.0f}글자 | 순수 LLM 대비 답변 길이 변화 |
| 💡 **활용 권장사항** | {recommendation} | 실무 적용 가능성 평가 |

"""
            
            # 승부 결과
            if len(models_scores) >= 2:
                models_scores.sort(key=lambda x: x[1], reverse=True)
                winner = models_scores[0]
                runner_up = models_scores[1]
                score_diff = winner[1] - runner_up[1]
                
                report += f"""
### 🥇 최종 법률 정확성 순위

1. 🥇 **1위: {winner[0]}** - {winner[1]:.1f}점
2. 🥈 **2위: {runner_up[0]}** - {runner_up[1]:.1f}점

**법률 정확성 점수 차이**: {score_diff:.1f}점

"""
                if score_diff > 8:
                    report += f"**결론**: {winner[0]}가 법률 분석에서 **확실히 더 정확하고 신뢰할 수 있는** 성능을 보입니다.\n\n"
                elif score_diff > 3:
                    report += f"**결론**: {winner[0]}가 법률 분석에서 **약간 더 나은** 성능을 보입니다.\n\n"
                else:
                    report += f"**결론**: 두 모델의 법률 분석 정확성이 **비슷한 수준**입니다.\n\n"

    # 법률 전문가 검증 필요 안내
    report += f"""
## ⚠️ 중요한 주의사항

### 🔍 사람 검증 필수 영역
이 분석은 AI 기반 1차 평가이며, 다음 영역은 **반드시 법률 전문가의 검증**이 필요합니다:

1. **법조문 해석의 정확성**: AI가 제시한 법조문 적용이 실제 법리에 부합하는지
2. **판례 적용의 적절성**: 인용된 판례가 해당 사안에 실제로 적용 가능한지  
3. **법률 조언의 실무성**: 제시된 해결방안이 실제 법률 실무에서 유효한지
4. **위험 요소 평가**: 제안된 방법의 법적 리스크 및 부작용

### 📋 활용 권고사항
- 70점 이상: 초기 법률 검토용으로 활용 가능, 전문가 최종 검증 필요
- 50-69점: 법률 정보 수집용으로만 활용, 의사결정에는 부적합
- 50점 미만: 법률 전문 업무에 사용 금지

---
*법률 정확성 기반 RAG 평가 시스템 v08240001 - 정밀하고 투명한 법률 AI 평가*
"""
    
    return report

def create_legal_accuracy_chart(results):
    """법률 정확성 점수 차트 생성"""
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
                data.append(improvement['overall_improvement'])
            else:
                data.append(0)
    
    if not data:
        return None
    
    # 데이터 재구성
    gpt_scores = data[::2]
    claude_scores = data[1::2]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(questions))
    width = 0.35
    
    # 바 차트
    bars1 = ax.bar(x - width/2, gpt_scores, width, label='GPT-4o', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, claude_scores, width, label='Claude-3.5', color='#e74c3c', alpha=0.8)
    
    # 점수 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            if bar.get_height() > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}', ha='center', va='bottom')
    
    ax.set_xlabel('Questions')
    ax.set_ylabel('Legal Accuracy Score (0-100)')
    ax.set_title('Legal Accuracy Score Comparison by Question')
    ax.set_xticks(x)
    ax.set_xticklabels(questions)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    return fig

def create_response_time_chart(results):
    """응답 시간 비교 차트 생성"""
    if not results or not results.get('questions'):
        return None
    
    questions = []
    pure_times = {'GPT-4o': [], 'Claude-3.5': []}
    rag_times = {'GPT-4o': [], 'Claude-3.5': []}
    
    for q_id, q_data in results['questions'].items():
        questions.append(f"Q{q_id[-1]}")
        
        for model in ['GPT-4o', 'Claude-3.5']:
            if model in q_data.get('analysis', {}):
                analysis = q_data['analysis'][model]
                pure_time = analysis.get('pure_response', {}).get('response_time', 0)
                rag_time = analysis.get('rag_response', {}).get('response_time', 0)
                pure_times[model].append(pure_time)
                rag_times[model].append(rag_time)
            else:
                pure_times[model].append(0)
                rag_times[model].append(0)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(questions))
    width = 0.2
    
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
    metrics = ['Legal Accuracy', 'Efficiency', 'Precedent Use', 'Speed', 'Comprehensiveness']
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 닫힌 다각형
    
    colors = ['#3498db', '#e74c3c']
    
    for i, model in enumerate(models):
        avg_data = model_averages[model]
        
        # 메트릭 값 정규화 (0-100)
        values = [
            avg_data.get('avg_improvement_score', 0),  # 법률 정확성
            max(0, 100 - abs(avg_data.get('avg_time_increase', 0)) * 10),  # 효율성
            min(100, avg_data.get('avg_cases_used', 0) * 25),  # 판례 활용도
            max(0, 100 - avg_data.get('avg_time_increase', 0) * 15),  # 속도
            min(100, avg_data.get('avg_length_increase', 0) / 10)  # 포괄성
        ]
        
        values += values[:1]  # 닫힌 다각형을 위해 첫 값을 마지막에 추가
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax.set_title('Legal Analysis Performance Comparison', pad=20)
    
    plt.tight_layout()
    return fig

def save_analysis_results():
    """분석 결과 저장"""
    global analysis_results
    
    if not analysis_results:
        return "❌ 저장할 분석 결과가 없습니다. 먼저 분석을 실행해주세요."
    
    try:
        json_path, md_path = save_results_multiple_formats(analysis_results, "results/legal_accuracy_rag")
        return f"✅ 결과 저장 완료!\n📊 JSON: {json_path}\n📋 상세보고서: {md_path}"
    except Exception as e:
        return f"❌ 저장 실패: {str(e)}"

# Gradio 인터페이스 구성
with gr.Blocks(
    title="법률 정확성 기반 RAG 성능 비교 v08240001",
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("""
    # ⚖️ 법률 정확성 기반 RAG 성능 비교 시스템 v08240001
    
    **새로운 평가 방식**: 법조문 인용 정확성 최우선 (50점) + 판례 적절성 (25점) + 법리 논리성 (15점) + 실무 적용성 (10점)
    
    **평가 특징**: 
    - 🔍 **완전 투명한 점수 산출**: 모든 평가 과정 상세 공개
    - 👤 **반자동화 평가**: AI 분석 + 사람 검증 필요
    - ⚖️ **법률 전문성 중심**: 법조문과 판례의 정확한 활용 평가
    """)
    
    # 시스템 초기화
    with gr.Row():
        with gr.Column():
            init_btn = gr.Button("🚀 시스템 초기화", variant="primary")
            init_status = gr.Textbox(label="초기화 상태", value="⏳ 시스템 초기화 버튼을 클릭하세요", interactive=False)
    
    # 초기화 상태에 따라 인터페이스 표시
    with gr.Column(visible=False) as main_interface:
        
        gr.Markdown("---")
        
        # 메인 분석 인터페이스
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📝 법률 질문 입력")
                
                questions_input = gr.Textbox(
                    label="질문 목록 (줄바꿈으로 구분)",
                    value="""취업규칙을 근로자에게 불리하게 변경할 때 사용자가 지켜야 할 법적 요건은 무엇인가요?
퇴직금 지급 기일을 연장하는 합의를 했더라도 연장된 기일까지 지급하지 않으면 형사처벌을 받나요?
부당해고 구제신청의 요건과 절차는 어떻게 되나요?""",
                    placeholder="법률 질문을 한 줄씩 입력하세요",
                    lines=6
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 분석 설정")
                
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.1,
                    label="Temperature (창의성)",
                    info="0.1 (정확성 우선) ~ 1.0 (창의성 우선)"
                )
                
                gr.Markdown("### 🚀 실행")
                analyze_btn = gr.Button(
                    "⚖️ 법률 정확성 분석 시작",
                    variant="primary",
                    size="lg"
                )
        
        # 결과 표시 영역
        gr.Markdown("---")
        gr.Markdown("## 📊 분석 결과")
        
        with gr.Tabs() as result_tabs:
            with gr.TabItem("📋 상세 분석 결과"):
                analysis_output = gr.Markdown(label="법률 정확성 분석 보고서")
            
            with gr.TabItem("📊 법률 정확성 차트"):
                legal_accuracy_chart = gr.Plot(label="법률 정확성 점수 비교")
            
            with gr.TabItem("⏱️ 응답 시간"):
                response_time_chart = gr.Plot(label="응답 시간 비교")
            
            with gr.TabItem("🎯 종합 성능"):
                performance_radar = gr.Plot(label="종합 성능 레이더 차트")
        
        # 결과 저장
        gr.Markdown("---")
        with gr.Row():
            save_btn = gr.Button("💾 결과 저장", variant="secondary")
            save_output = gr.Textbox(label="저장 결과", interactive=False)
        
        # 이벤트 바인딩
        analyze_btn.click(
            fn=run_legal_accuracy_analysis,
            inputs=[questions_input, temperature],
            outputs=[analysis_output, legal_accuracy_chart, response_time_chart, performance_radar]
        )
        
        save_btn.click(
            fn=save_analysis_results,
            outputs=[save_output]
        )
    
    # 초기화 버튼 이벤트
    init_btn.click(
        fn=initialize_system,
        outputs=[init_status, main_interface]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=False,
        debug=True
    )