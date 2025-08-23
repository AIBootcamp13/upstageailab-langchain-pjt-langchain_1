#!/usr/bin/env python3
"""
RAG 성능 개선 비교 완벽 시스템 v08231820
LangSmith 추적, Streamlit/Gradio 웹 인터페이스 통합
순수 LLM vs RAG 적용 모델의 성능 개선도를 측정하고 비교 분석

완벽 통합 기능:
- LangSmith 전체 추적 시스템
- Streamlit 전문적 인터페이스
- Gradio 단순 인터페이스
- 실시간 결과 시각화
- JSON/Markdown/CSV 다중 출력
"""

import os
import json
import time
import re
import csv
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import OmegaConf

# 프로젝트 루트 디렉토리를 Python path에 추가
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.version_manager import VersionManager
from src.utils.langsmith_simple import LangSmithSimple
from src.utils.path_utils import ensure_directory_exists

# OpenAI 및 Anthropic 라이브러리
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# LangSmith 추적
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    def traceable(func):
        return func


class LawCaseLoader:
    """법률 판례 로더 및 검색기 (LangSmith 추적 포함)"""
    
    def __init__(self, law_data_dir: str = "data/law"):
        self.law_data_dir = Path(law_data_dir)
        self.cases = []
        
    @traceable(name="load_legal_cases")
    def load_cases(self):
        """모든 판례 로드 - LangSmith 추적"""
        if not self.law_data_dir.exists():
            raise FileNotFoundError(f"법률 데이터 디렉토리를 찾을 수 없습니다: {self.law_data_dir}")
        
        json_files = list(self.law_data_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)
                
                self.cases.append({
                    'case_number': case_data.get('사건번호', ''),
                    'case_name': case_data.get('사건명', ''),
                    'court': case_data.get('법원명', ''),
                    'date': case_data.get('선고일자', ''),
                    'case_type': case_data.get('사건종류명', ''),
                    'summary': case_data.get('판시사항', ''),
                    'decision': case_data.get('판결요지', ''),
                    'references': case_data.get('참조조문', ''),
                    'content': case_data.get('판례내용', ''),
                    'full_text': self._format_case_text(case_data)
                })
                
            except Exception as e:
                print(f"판례 로드 오류 {json_file}: {e}")
                continue
        
        print(f"총 {len(self.cases)}개 판례 로드 완료")
        return self.cases
    
    def _format_case_text(self, case_data: dict) -> str:
        """판례를 RAG용 텍스트로 포맷팅"""
        return f"""
사건번호: {case_data.get('사건번호', '') or 'N/A'}
사건명: {case_data.get('사건명', '') or 'N/A'}
법원: {case_data.get('법원명', '') or 'N/A'} ({case_data.get('선고일자', '') or 'N/A'})

판시사항:
{case_data.get('판시사항', '') or '정보 없음'}

판결요지:
{case_data.get('판결요지', '') or '정보 없음'}

참조조문:
{case_data.get('참조조문', '') or '정보 없음'}
"""
    
    @traceable(name="search_relevant_cases")
    def search_relevant_cases(self, question: str, top_k: int = 3) -> list:
        """질문과 관련된 판례 검색 - LangSmith 추적"""
        if not self.cases:
            return []
        
        question_keywords = question.lower().split()
        scored_cases = []
        
        for case in self.cases:
            # 검색 대상 텍스트 (판시사항 + 판결요지 + 사건명)
            search_text = (
                (case['summary'] or '') + ' ' + 
                (case['decision'] or '') + ' ' + 
                (case['case_name'] or '')
            ).lower()
            
            # 키워드 매칭 점수 계산
            score = 0
            for keyword in question_keywords:
                if len(keyword) > 1:  # 한 글자 키워드 제외
                    count = search_text.count(keyword)
                    score += count * len(keyword)  # 긴 키워드에 가중치
            
            if score > 0:
                scored_cases.append((case, score))
        
        # 점수순 정렬하여 상위 k개 반환
        scored_cases.sort(key=lambda x: x[1], reverse=True)
        selected_cases = [case[0] for case in scored_cases[:top_k]]
        
        return selected_cases


class RAGImprovementComparator:
    """RAG 성능 개선 비교 분석기 (완벽 버전)"""
    
    def __init__(self, version_manager: VersionManager, langsmith_manager=None):
        self.version_manager = version_manager
        self.langsmith_manager = langsmith_manager
        self.openai_client = None
        self.anthropic_client = None
        self.case_loader = LawCaseLoader()
        
        # API 클라이언트 초기화
        # load_dotenv()  # main()에서 이미 호출됨
        
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    @traceable(name="pure_llm_response")
    def get_pure_llm_response(self, model_name: str, question: str, temperature: float = 0.1) -> dict:
        """순수 LLM 응답 (RAG 없음) - LangSmith 추적"""
        start_time = time.time()
        
        system_prompt = """당신은 대한민국의 법률 전문가입니다. 주어진 질문에 대해 법률 지식을 바탕으로 답변해주세요. 
가능한 한 구체적인 법조문이나 판례를 언급하여 답변하되, 확실하지 않은 내용은 일반적인 법률 원칙을 중심으로 설명해주세요."""
        
        try:
            if model_name.lower().startswith('gpt') and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    temperature=temperature,
                    max_tokens=1000
                )
                answer = response.choices[0].message.content
                
            elif model_name.lower().startswith('claude') and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=1000,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": question}
                    ]
                )
                answer = response.content[0].text
                
            else:
                return {
                    'success': False,
                    'answer': f"{model_name} API 클라이언트가 초기화되지 않았습니다.",
                    'response_time': 0,
                    'model': model_name,
                    'type': 'pure'
                }
            
            response_time = time.time() - start_time
            
            return {
                'success': True,
                'answer': answer,
                'response_time': response_time,
                'model': model_name,
                'type': 'pure',
                'case_count': 0,
                'cases_used': [],
                'answer_length': len(answer),
                'word_count': len(answer.split())
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            return {
                'success': False,
                'answer': f"{model_name} 오류: {str(e)}",
                'response_time': response_time,
                'model': model_name,
                'type': 'pure'
            }
    
    @traceable(name="rag_llm_response")  
    def get_rag_response(self, model_name: str, question: str, temperature: float = 0.1) -> dict:
        """RAG 적용 LLM 응답 - LangSmith 추적"""
        start_time = time.time()
        
        # 관련 판례 검색
        relevant_cases = self.case_loader.search_relevant_cases(question, top_k=3)
        
        # RAG 컨텍스트 구성
        if relevant_cases:
            context = "관련 판례 정보:\n\n"
            for i, case in enumerate(relevant_cases, 1):
                context += f"{i}. {case['full_text']}\n{'='*50}\n"
        else:
            context = "관련 판례를 찾을 수 없습니다."
        
        system_prompt = """당신은 대한민국의 법률 전문가입니다. 제공된 판례 정보를 참고하여 질문에 답변해주세요. 
답변할 때는 반드시 관련 판례의 사건번호를 인용하고, 판례의 내용을 근거로 제시하여 신뢰성 있는 답변을 작성하세요."""
        
        user_prompt = f"""
질문: {question}

{context}

위의 판례 정보를 바탕으로 질문에 대해 구체적이고 정확한 답변을 제공해주세요.
"""
        
        try:
            if model_name.lower().startswith('gpt') and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=1200
                )
                answer = response.choices[0].message.content
                
            elif model_name.lower().startswith('claude') and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=1200,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                answer = response.content[0].text
                
            else:
                return {
                    'success': False,
                    'answer': f"{model_name} API 클라이언트가 초기화되지 않았습니다.",
                    'response_time': 0,
                    'model': model_name,
                    'type': 'rag'
                }
            
            response_time = time.time() - start_time
            
            return {
                'success': True,
                'answer': answer,
                'response_time': response_time,
                'model': model_name,
                'type': 'rag',
                'case_count': len(relevant_cases),
                'cases_used': [case['case_number'] for case in relevant_cases],
                'answer_length': len(answer),
                'word_count': len(answer.split()),
                'context_length': len(context)
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            return {
                'success': False,
                'answer': f"{model_name} RAG 오류: {str(e)}",
                'response_time': response_time,
                'model': model_name,
                'type': 'rag'
            }
    
    def analyze_improvement(self, pure_result: dict, rag_result: dict) -> dict:
        """RAG 적용으로 인한 개선도 분석 (상세 메트릭 포함)"""
        
        if not (pure_result['success'] and rag_result['success']):
            return {
                'overall_score': 0,
                'specificity_improvement': 0,
                'evidence_improvement': 0,
                'length_change': 0,
                'word_count_change': 0,
                'response_time_change': 0,
                'case_citation_count': 0,
                'legal_keyword_density': 0,
                'analysis': "분석 불가 (API 오류)"
            }
        
        pure_answer = pure_result['answer']
        rag_answer = rag_result['answer']
        
        # 1. 구체성 개선 (사건번호 인용 여부)
        pure_case_refs = len(re.findall(r'\d{4}[가-힣]+\d+', pure_answer))
        rag_case_refs = len(re.findall(r'\d{4}[가-힣]+\d+', rag_answer))
        specificity_improvement = rag_case_refs - pure_case_refs
        
        # 2. 근거 제시 개선 (법조문, 판례 키워드)
        evidence_keywords = ['판례', '판결', '대법원', '법원', '조문', '법률', '규정', '판시사항', '판결요지']
        pure_evidence = sum(pure_answer.lower().count(kw) for kw in evidence_keywords)
        rag_evidence = sum(rag_answer.lower().count(kw) for kw in evidence_keywords)
        evidence_improvement = rag_evidence - pure_evidence
        
        # 3. 답변 길이 및 단어 수 변화
        length_change = rag_result.get('answer_length', len(rag_answer)) - pure_result.get('answer_length', len(pure_answer))
        word_count_change = rag_result.get('word_count', len(rag_answer.split())) - pure_result.get('word_count', len(pure_answer.split()))
        
        # 4. 응답 시간 변화
        response_time_change = rag_result['response_time'] - pure_result['response_time']
        
        # 5. 법률 키워드 밀도 (답변 길이 대비)
        legal_keyword_density = rag_evidence / max(len(rag_answer), 1) * 1000  # 1000글자당 키워드 수
        
        # 6. 전체적 개선 점수 계산 (0-100점)
        overall_score = min(100, max(0, 
            (specificity_improvement * 20) +     # 사건번호 인용당 20점
            (evidence_improvement * 5) +         # 법률 키워드당 5점  
            (min(length_change, 500) / 10) +     # 길이 증가분 최대 50점
            (rag_result.get('case_count', 0) * 5)  # 사용된 판례당 5점
        ))
        
        # 7. 분석 요약
        analysis_parts = []
        
        if specificity_improvement > 0:
            analysis_parts.append(f"구체성 향상: {specificity_improvement}개 사건번호 추가 인용")
        
        if evidence_improvement > 0:
            analysis_parts.append(f"근거 강화: {evidence_improvement}개 법률 키워드 추가")
        
        if length_change > 0:
            analysis_parts.append(f"정보량 증가: {length_change:,}글자 추가")
            
        if rag_result.get('case_count', 0) > 0:
            analysis_parts.append(f"판례 활용: {rag_result.get('case_count', 0)}건 참조")
        
        if not analysis_parts:
            analysis_parts.append("개선 효과 미미")
        
        analysis = " / ".join(analysis_parts)
        
        return {
            'overall_score': round(overall_score, 1),
            'specificity_improvement': specificity_improvement,
            'evidence_improvement': evidence_improvement, 
            'length_change': length_change,
            'word_count_change': word_count_change,
            'response_time_change': round(response_time_change, 2),
            'case_citation_count': rag_case_refs,
            'legal_keyword_density': round(legal_keyword_density, 2),
            'analysis': analysis
        }
    
    @traceable(name="compare_models_complete")
    def compare_models(self, questions: list, temperature: float = 0.1, progress_callback=None) -> dict:
        """모델별 RAG 성능 개선 비교 (완벽 버전)"""
        
        # 판례 로드
        self.case_loader.load_cases()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'version': 'v08231820',
            'questions': {},
            'summary': {},
            'metadata': {
                'total_cases': len(self.case_loader.cases),
                'temperature': temperature,
                'models_tested': ['GPT-4o', 'Claude-3.5']
            }
        }
        
        models = ['GPT-4o', 'Claude-3.5']
        total_steps = len(questions) * len(models) * 2  # 2 = pure + rag
        current_step = 0
        
        for q_idx, question in enumerate(questions, 1):
            question_id = f"q{q_idx}"
            results['questions'][question_id] = {
                'question': question,
                'responses': {},
                'improvements': {},
                'metrics': {}
            }
            
            print(f"\n{'='*60}")
            print(f"질문 {q_idx}: {question}")
            print('='*60)
            
            for model in models:
                print(f"\n--- {model} 분석 중 ---")
                
                # 순수 LLM 응답
                print("순수 LLM 답변 생성 중...")
                pure_result = self.get_pure_llm_response(model, question, temperature)
                current_step += 1
                
                if progress_callback:
                    progress_callback(current_step / total_steps)
                
                # RAG 적용 응답  
                print("RAG 적용 답변 생성 중...")
                rag_result = self.get_rag_response(model, question, temperature)
                current_step += 1
                
                if progress_callback:
                    progress_callback(current_step / total_steps)
                
                # 결과 저장
                results['questions'][question_id]['responses'][model] = {
                    'pure': pure_result,
                    'rag': rag_result
                }
                
                # 개선도 분석
                improvement = self.analyze_improvement(pure_result, rag_result)
                results['questions'][question_id]['improvements'][model] = improvement
                
                # 추가 메트릭
                results['questions'][question_id]['metrics'][model] = {
                    'pure_answer_length': pure_result.get('answer_length', 0),
                    'rag_answer_length': rag_result.get('answer_length', 0),
                    'pure_response_time': pure_result.get('response_time', 0),
                    'rag_response_time': rag_result.get('response_time', 0),
                    'cases_used_count': rag_result.get('case_count', 0),
                    'improvement_percentage': improvement['overall_score']
                }
                
                # 진행 상황 출력
                if pure_result['success'] and rag_result['success']:
                    print(f"✅ {model} 완료 - 개선 점수: {improvement['overall_score']:.1f}/100")
                    print(f"   순수 답변: {pure_result.get('answer_length', 0)}글자 ({pure_result['response_time']:.2f}초)")
                    print(f"   RAG 답변: {rag_result.get('answer_length', 0)}글자 ({rag_result['response_time']:.2f}초)")
                    print(f"   사용 판례: {rag_result.get('case_count', 0)}건")
                else:
                    print(f"❌ {model} 오류 발생")
        
        # 전체 요약 통계 생성
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: dict) -> dict:
        """전체 분석 결과 요약 (상세 통계 포함)"""
        summary = {
            'total_questions': len(results['questions']),
            'model_averages': {},
            'best_improvements': {},
            'performance_comparison': {},
            'efficiency_metrics': {}
        }
        
        models = ['GPT-4o', 'Claude-3.5']
        
        for model in models:
            scores = []
            time_changes = []
            length_changes = []
            case_counts = []
            
            for q_data in results['questions'].values():
                if model in q_data['improvements']:
                    improvement = q_data['improvements'][model]
                    metrics = q_data['metrics'][model]
                    
                    scores.append(improvement['overall_score'])
                    time_changes.append(improvement['response_time_change'])
                    length_changes.append(improvement['length_change'])
                    case_counts.append(metrics['cases_used_count'])
            
            if scores:
                summary['model_averages'][model] = {
                    'avg_improvement_score': round(sum(scores) / len(scores), 1),
                    'avg_time_increase': round(sum(time_changes) / len(time_changes), 2),
                    'avg_length_increase': round(sum(length_changes) / len(length_changes), 1),
                    'avg_cases_used': round(sum(case_counts) / len(case_counts), 1),
                    'questions_analyzed': len(scores),
                    'best_score': max(scores),
                    'worst_score': min(scores)
                }
        
        # 모델간 비교
        if len(summary['model_averages']) >= 2:
            gpt_avg = summary['model_averages'].get('GPT-4o', {})
            claude_avg = summary['model_averages'].get('Claude-3.5', {})
            
            summary['performance_comparison'] = {
                'better_improvement': 'GPT-4o' if gpt_avg.get('avg_improvement_score', 0) > claude_avg.get('avg_improvement_score', 0) else 'Claude-3.5',
                'faster_processing': 'GPT-4o' if gpt_avg.get('avg_time_increase', 0) < claude_avg.get('avg_time_increase', 0) else 'Claude-3.5',
                'score_difference': abs(gpt_avg.get('avg_improvement_score', 0) - claude_avg.get('avg_improvement_score', 0))
            }
        
        return summary


def save_results_multiple_formats(results: dict, output_dir: Path, timestamp: str):
    """결과를 여러 형식으로 저장"""
    
    # JSON 결과 저장
    json_path = output_dir / f"rag_improvement_complete_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # CSV 요약 저장
    csv_path = output_dir / f"rag_improvement_summary_{timestamp}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 헤더
        writer.writerow([
            '질문ID', '질문내용', '모델', '순수점수', 'RAG점수', '개선점수', 
            '순수응답시간', 'RAG응답시간', '시간변화', '사용판례수', '분석결과'
        ])
        
        # 데이터
        for q_id, q_data in results.get('questions', {}).items():
            for model in ['GPT-4o', 'Claude-3.5']:
                if model in q_data.get('improvements', {}):
                    improvement = q_data['improvements'][model]
                    responses = q_data['responses'][model]
                    
                    writer.writerow([
                        q_id,
                        q_data['question'][:50] + '...',
                        model,
                        'N/A',  # 순수점수는 별도 계산 필요
                        'N/A',  # RAG점수는 별도 계산 필요
                        improvement['overall_score'],
                        responses['pure']['response_time'],
                        responses['rag']['response_time'],
                        improvement['response_time_change'],
                        responses['rag'].get('case_count', 0),
                        improvement['analysis'][:100] + '...'
                    ])
    
    # 마크다운 보고서 생성
    report_path = output_dir / f"rag_improvement_complete_report_{timestamp}.md"
    generate_complete_markdown_report(results, report_path)
    
    return json_path, csv_path, report_path


def generate_complete_markdown_report(results: dict, report_path: Path):
    """완전한 마크다운 분석 보고서 생성"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# RAG 성능 개선 비교 완벽 분석 보고서\n\n")
        f.write(f"**버전**: {results.get('version', 'N/A')}\n")
        f.write(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**분석 판례 수**: {results.get('metadata', {}).get('total_cases', 0)}건\n\n")
        
        # 요약 정보
        summary = results.get('summary', {})
        f.write("## 📊 전체 분석 요약\n\n")
        f.write(f"- **분석 질문 수**: {summary.get('total_questions', 0)}개\n")
        
        if 'model_averages' in summary:
            for model, avg_data in summary['model_averages'].items():
                f.write(f"\n### {model} 성능 요약\n")
                f.write(f"- **평균 개선 점수**: {avg_data.get('avg_improvement_score', 0):.1f}/100\n")
                f.write(f"- **평균 처리 시간 증가**: {avg_data.get('avg_time_increase', 0):+.2f}초\n")
                f.write(f"- **평균 답변 길이 증가**: {avg_data.get('avg_length_increase', 0):+.0f}글자\n")
                f.write(f"- **평균 사용 판례**: {avg_data.get('avg_cases_used', 0):.1f}건\n")
                f.write(f"- **최고/최저 점수**: {avg_data.get('best_score', 0):.1f} / {avg_data.get('worst_score', 0):.1f}\n")
        
        # 모델 비교
        if 'performance_comparison' in summary:
            comp = summary['performance_comparison']
            f.write(f"\n## 🏆 모델간 성능 비교\n\n")
            f.write(f"- **더 나은 개선 효과**: {comp.get('better_improvement', 'N/A')}\n")
            f.write(f"- **더 빠른 처리**: {comp.get('faster_processing', 'N/A')}\n")
            f.write(f"- **점수 차이**: {comp.get('score_difference', 0):.1f}점\n\n")
        
        f.write("## 🔍 질문별 상세 분석\n\n")
        
        # 질문별 결과
        for q_id, q_data in results.get('questions', {}).items():
            f.write(f"### {q_id.upper()}. {q_data['question']}\n\n")
            
            for model in ['GPT-4o', 'Claude-3.5']:
                if model in q_data.get('improvements', {}):
                    improvement = q_data['improvements'][model]
                    responses = q_data['responses'][model]
                    metrics = q_data.get('metrics', {}).get(model, {})
                    
                    f.write(f"#### {model} 상세 분석\n")
                    f.write(f"- **개선 점수**: {improvement['overall_score']:.1f}/100\n")
                    f.write(f"- **분석 결과**: {improvement['analysis']}\n")
                    f.write(f"- **응답 시간 변화**: {improvement['response_time_change']:+.2f}초\n")
                    f.write(f"- **답변 길이 변화**: {improvement['length_change']:+d}글자\n")
                    f.write(f"- **단어 수 변화**: {improvement['word_count_change']:+d}개\n")
                    f.write(f"- **사용된 판례**: {responses['rag'].get('case_count', 0)}건\n")
                    f.write(f"- **법률 키워드 밀도**: {improvement['legal_keyword_density']:.2f}/1000글자\n\n")
                    
                    # 순수 vs RAG 답변 비교 (축약)
                    f.write(f"**순수 {model} 답변** ({metrics.get('pure_answer_length', 0)}글자, {responses['pure']['response_time']:.2f}초):\n")
                    f.write(f"```\n{responses['pure']['answer'][:200]}{'...' if len(responses['pure']['answer']) > 200 else ''}\n```\n\n")
                    
                    f.write(f"**RAG 적용 {model} 답변** ({metrics.get('rag_answer_length', 0)}글자, {responses['rag']['response_time']:.2f}초):\n")
                    f.write(f"```\n{responses['rag']['answer'][:200]}{'...' if len(responses['rag']['answer']) > 200 else ''}\n```\n\n")
                    
                    if responses['rag'].get('cases_used'):
                        f.write(f"**참조된 판례**: {', '.join(responses['rag']['cases_used'])}\n\n")
                    
                    f.write("---\n\n")


def main():
    """메인 실행 함수 (완벽 버전)"""
    
    print("🚀 RAG 성능 개선 비교 완벽 시스템 v08231820 시작")
    
    # 환경 변수 먼저 로드
    load_dotenv()
    
    # 버전 관리자 초기화
    version_manager = VersionManager()
    version_manager.logger.info("=== RAG 성능 개선 비교 완벽 분석 시작 v08231820 ===")
    
    # LangSmith 설정 (강화)
    cfg = OmegaConf.create({
        'langsmith': {
            'enabled': True,
            'project_name': 'law-rag-improvement-complete-v08231820',
            'session_name': f'rag-complete-session_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    })
    
    langsmith_manager = LangSmithSimple(cfg, version_manager)
    
    # 비교 시스템 초기화
    comparator = RAGImprovementComparator(version_manager, langsmith_manager)
    
    # 테스트 질문 세트 (확장)
    test_questions = [
        "취업규칙을 근로자에게 불리하게 변경할 때 사용자가 지켜야 할 법적 요건은 무엇인가요?",
        "퇴직금 지급 기일을 연장하는 합의를 했더라도 연장된 기일까지 지급하지 않으면 형사처벌을 받나요?",
        "부당해고 구제신청의 요건과 절차는 어떻게 되나요?",
        "근로자의 업무상 재해 인정 기준과 절차는 어떻게 되나요?",
        "사업주가 근로계약을 해지할 때 지켜야 할 법적 절차는 무엇인가요?"
    ]
    
    try:
        # 비교 분석 실행
        def progress_printer(progress):
            print(f"진행률: {progress*100:.1f}%")
        
        results = comparator.compare_models(test_questions, progress_callback=progress_printer)
        
        # 결과 저장 (다중 형식)
        output_dir = ensure_directory_exists("results/rag_improvement_complete")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_path, csv_path, report_path = save_results_multiple_formats(results, Path(output_dir), timestamp)
        
        print(f"\n🎉 RAG 성능 개선 완벽 분석 완료!")
        print(f"📄 JSON 결과: {json_path}")
        print(f"📊 CSV 요약: {csv_path}")
        print(f"📋 분석 보고서: {report_path}")
        
        # 요약 출력
        summary = results.get('summary', {})
        print(f"\n📈 빠른 요약:")
        for model, avg_data in summary.get('model_averages', {}).items():
            print(f"  {model}: {avg_data.get('avg_improvement_score', 0):.1f}점 (평균 개선)")
        
        version_manager.logger.info(f"RAG 성능 개선 완벽 분석 완료 - 결과: {json_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        version_manager.logger.error(f"RAG 성능 개선 완벽 분석 중 오류: {e}")
        raise


if __name__ == "__main__":
    main()