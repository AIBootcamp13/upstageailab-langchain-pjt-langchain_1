#!/usr/bin/env python3
"""
RAG 성능 개선 비교 시스템 v08240001
법률 정확성 중심의 객관적 평가 시스템 통합 버전
- 법조문 인용 정확성 최우선 (50점)
- 반자동화: AI 분석 + 사람 검증
- 완전 투명한 점수 산출 과정
"""

import os
import json
import time
import re
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
from src.legal_accuracy_evaluator_08240001 import (
    LegalAccuracyEvaluator, LegalAccuracyScore, EvaluationDetail, generate_transparency_report
)

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
    def traceable(name=None):
        def decorator(func):
            return func
        return decorator

class LawCaseLoader:
    """법률 판례 로더 및 검색기 (LangSmith 추적 포함)"""
    
    def __init__(self, data_dir: str = "data/law"):
        self.data_dir = Path(data_dir)
        self.cases = []
        
    @traceable(name="load_legal_cases")
    def load_cases(self):
        """모든 판례 로드 - LangSmith 추적"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"법률 데이터 디렉토리를 찾을 수 없습니다: {self.data_dir}")
        
        json_files = list(self.data_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)
                    
                # 필수 필드 확인
                required_fields = ['case_number', 'summary', 'facts']
                if all(field in case_data for field in required_fields):
                    self.cases.append({
                        'case_number': case_data['case_number'],
                        'summary': case_data.get('summary', ''),
                        'facts': case_data.get('facts', ''),
                        'holdings': case_data.get('holdings', ''),
                        'keywords': case_data.get('keywords', []),
                        'file_path': str(json_file)
                    })
                    
            except Exception as e:
                print(f"판례 로드 중 오류 ({json_file}): {e}")
        
        print(f"총 {len(self.cases)}개 판례 로드 완료")
        return len(self.cases)
    
    @traceable(name="retrieve_relevant_cases")
    def retrieve_relevant_cases(self, question: str, top_k: int = 3):
        """질문과 관련된 판례 검색 - LangSmith 추적"""
        if not self.cases:
            self.load_cases()
            
        scored_cases = []
        question_lower = question.lower()
        
        for case in self.cases:
            score = 0
            case_text = f"{case['summary']} {case['facts']} {case['holdings']}".lower()
            
            # 키워드 매칭 점수
            common_words = set(question_lower.split()) & set(case_text.split())
            score += len(common_words)
            
            # 특정 법률 키워드 가중치
            legal_keywords = ['취업규칙', '퇴직금', '해고', '근로기준', '동의', '변경']
            for keyword in legal_keywords:
                if keyword in question_lower and keyword in case_text:
                    score += 5
            
            if score > 0:
                scored_cases.append((case, score))
        
        # 점수 순으로 정렬하여 상위 k개 반환
        scored_cases.sort(key=lambda x: x[1], reverse=True)
        return [case[0] for case in scored_cases[:top_k]]

class RAGLegalAccuracyComparator:
    """RAG 성능 개선 비교기 - 법률 정확성 중심 평가"""
    
    def __init__(self, version_manager: VersionManager, langsmith_manager=None):
        self.version_manager = version_manager
        self.langsmith_manager = langsmith_manager
        self.openai_client = None
        self.anthropic_client = None
        self.case_loader = LawCaseLoader()
        self.legal_evaluator = LegalAccuracyEvaluator(version_manager)
        
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
            if model_name == "GPT-4o" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    temperature=temperature,
                    max_tokens=2000
                )
                answer = response.choices[0].message.content
                
            elif model_name == "Claude-3.5" and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=2000,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": question}]
                )
                answer = response.content[0].text
                
            else:
                return {
                    'success': False,
                    'answer': '',
                    'response_time': 0,
                    'error': f"{model_name} 클라이언트 사용 불가"
                }
            
            end_time = time.time()
            
            return {
                'success': True,
                'answer': answer,
                'response_time': end_time - start_time,
                'answer_length': len(answer),
                'word_count': len(answer.split()),
                'model': model_name
            }
            
        except Exception as e:
            return {
                'success': False,
                'answer': '',
                'response_time': time.time() - start_time,
                'error': str(e)
            }
    
    @traceable(name="rag_enhanced_response")  
    def get_rag_response(self, model_name: str, question: str, temperature: float = 0.1) -> dict:
        """RAG 적용 LLM 응답 - LangSmith 추적"""
        start_time = time.time()
        
        try:
            # 관련 판례 검색
            relevant_cases = self.case_loader.retrieve_relevant_cases(question, top_k=3)
            
            # 판례 정보를 컨텍스트로 구성
            context = "다음은 관련 판례들입니다:\n\n"
            for i, case in enumerate(relevant_cases, 1):
                context += f"【판례 {i}】\n"
                context += f"사건번호: {case['case_number']}\n"
                context += f"사건개요: {case.get('summary', '')}\n"
                context += f"사실관계: {case.get('facts', '')}\n"
                context += f"판시사항: {case.get('holdings', '')}\n\n"
            
            enhanced_prompt = f"""{context}

위의 판례들을 참고하여 다음 질문에 대해 구체적이고 정확한 법률 답변을 제공해주세요.
반드시 관련 법조문과 판례를 정확히 인용하며, 논리적인 근거를 들어 설명해주세요.

질문: {question}"""

            system_prompt = """당신은 대한민국의 법률 전문가입니다. 제공된 판례를 바탕으로 정확한 법률 답변을 제공해주세요.
- 관련 법조문을 정확히 인용하세요
- 판례의 판시사항을 올바르게 적용하세요  
- 논리적인 구조로 답변하세요 (전제 → 추론 → 결론)
- 실무에 도움이 되는 구체적인 해결방안을 제시하세요"""
            
            if model_name == "GPT-4o" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": enhanced_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=2000
                )
                answer = response.choices[0].message.content
                
            elif model_name == "Claude-3.5" and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=2000,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": enhanced_prompt}]
                )
                answer = response.content[0].text
                
            else:
                return {
                    'success': False,
                    'answer': '',
                    'response_time': 0,
                    'cases_used': 0,
                    'error': f"{model_name} 클라이언트 사용 불가"
                }
            
            end_time = time.time()
            
            return {
                'success': True,
                'answer': answer,
                'response_time': end_time - start_time,
                'answer_length': len(answer),
                'word_count': len(answer.split()),
                'cases_used': len(relevant_cases),
                'case_data': relevant_cases,
                'model': model_name
            }
            
        except Exception as e:
            return {
                'success': False,
                'answer': '',
                'response_time': time.time() - start_time,
                'cases_used': 0,
                'error': str(e)
            }
    
    @traceable(name="legal_accuracy_analysis")
    def analyze_legal_accuracy_improvement(self, pure_result: dict, rag_result: dict, question: str) -> dict:
        """법률 정확성 기반 개선도 분석 - LangSmith 추적"""
        
        if not (pure_result['success'] and rag_result['success']):
            return {
                'legal_accuracy_score': LegalAccuracyScore(),
                'evaluation_details': [],
                'transparency_report': "분석 불가 (API 오류)",
                'overall_improvement': 0.0,
                'analysis': "분석 불가"
            }
        
        # 새로운 법률 정확성 평가 시스템 사용
        case_data = rag_result.get('case_data', [])
        legal_score, evaluation_details = self.legal_evaluator.evaluate_legal_accuracy(
            pure_result['answer'],
            rag_result['answer'], 
            question,
            case_data
        )
        
        # 투명성 보고서 생성
        transparency_report = generate_transparency_report(legal_score, evaluation_details)
        
        # 전체적인 개선도 점수 (0-100점)
        overall_improvement = legal_score.total_score()
        
        # 간단한 분석 요약
        analysis_parts = []
        if legal_score.statute_citation_accuracy > 15:
            analysis_parts.append(f"법조문 인용 우수 ({legal_score.statute_citation_accuracy:.1f}/30점)")
        if legal_score.precedent_relevance > 8:
            analysis_parts.append(f"판례 활용 양호 ({legal_score.precedent_relevance:.1f}/15점)")
        if legal_score.legal_reasoning_logic > 8:
            analysis_parts.append(f"논리 구조 체계적 ({legal_score.legal_reasoning_logic:.1f}/15점)")
        
        if not analysis_parts:
            analysis_parts.append("법률 정확성 개선 필요")
            
        analysis = " / ".join(analysis_parts)
        
        return {
            'legal_accuracy_score': legal_score,
            'evaluation_details': evaluation_details,
            'transparency_report': transparency_report,
            'overall_improvement': round(overall_improvement, 1),
            'analysis': analysis,
            
            # 기존 호환성을 위한 필드들
            'overall_score': round(overall_improvement, 1),
            'case_citation_score': legal_score.statute_citation_accuracy,
            'keyword_density_score': legal_score.precedent_relevance, 
            'length_score': legal_score.legal_reasoning_logic,
            'time_efficiency_score': legal_score.practical_applicability
        }
    
    @traceable(name="compare_models_legal_accuracy")
    def compare_models(self, questions: list, temperature: float = 0.1, progress_callback=None) -> dict:
        """모델별 RAG 성능 개선 비교 (법률 정확성 중심)"""
        
        # 판례 로드
        self.case_loader.load_cases()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'version': 'v08240001_legal_accuracy',
            'evaluation_method': '법률 정확성 중심 평가',
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
                'analysis': {},
                'improvements': {},
                'transparency_reports': {}
            }
            
            print(f"\n{'='*60}")
            print(f"질문 {q_idx}: {question[:50]}...")
            print(f"{'='*60}")
            
            for model in models:
                print(f"\n--- {model} 분석 중 ---")
                
                # 순수 LLM 응답
                current_step += 1
                if progress_callback:
                    progress_callback(current_step / total_steps)
                    
                print("순수 LLM 답변 생성 중...")
                pure_result = self.get_pure_llm_response(model, question, temperature)
                
                # RAG 적용 응답
                current_step += 1
                if progress_callback:
                    progress_callback(current_step / total_steps)
                    
                print("RAG 적용 답변 생성 중...")
                rag_result = self.get_rag_response(model, question, temperature)
                
                # 법률 정확성 기반 개선도 분석
                improvement_analysis = self.analyze_legal_accuracy_improvement(
                    pure_result, rag_result, question
                )
                
                # 결과 저장
                results['questions'][question_id]['analysis'][model] = {
                    'pure_response': pure_result,
                    'rag_response': rag_result
                }
                
                results['questions'][question_id]['improvements'][model] = improvement_analysis
                
                # 투명성 보고서 별도 저장
                results['questions'][question_id]['transparency_reports'][model] = improvement_analysis['transparency_report']
                
                # 진행 상황 출력
                overall_score = improvement_analysis['overall_improvement']
                used_cases = rag_result.get('cases_used', 0)
                pure_time = pure_result.get('response_time', 0)
                rag_time = rag_result.get('response_time', 0)
                
                print(f"✅ {model} 완료 - 법률정확성 점수: {overall_score}/100")
                print(f"   순수 답변: {len(pure_result.get('answer', ''))}글자 ({pure_time:.2f}초)")
                print(f"   RAG 답변: {len(rag_result.get('answer', ''))}글자 ({rag_time:.2f}초)")
                print(f"   사용 판례: {used_cases}건")
        
        # 전체 요약 통계 계산
        results['summary'] = self._calculate_summary_statistics(results['questions'])
        
        return results
    
    def _calculate_summary_statistics(self, questions_data: dict) -> dict:
        """전체 요약 통계 계산"""
        
        model_stats = {}
        models = ['GPT-4o', 'Claude-3.5']
        
        for model in models:
            scores = []
            time_increases = []
            cases_used = []
            length_increases = []
            
            for q_data in questions_data.values():
                if model in q_data.get('improvements', {}):
                    improvement = q_data['improvements'][model]
                    analysis = q_data['analysis'][model]
                    
                    scores.append(improvement['overall_improvement'])
                    
                    pure_time = analysis['pure_response'].get('response_time', 0)
                    rag_time = analysis['rag_response'].get('response_time', 0)
                    time_increases.append(rag_time - pure_time)
                    
                    cases_used.append(analysis['rag_response'].get('cases_used', 0))
                    
                    pure_length = len(analysis['pure_response'].get('answer', ''))
                    rag_length = len(analysis['rag_response'].get('answer', ''))
                    length_increases.append(rag_length - pure_length)
            
            if scores:
                model_stats[model] = {
                    'avg_improvement_score': sum(scores) / len(scores),
                    'avg_time_increase': sum(time_increases) / len(time_increases),
                    'avg_cases_used': sum(cases_used) / len(cases_used),
                    'avg_length_increase': sum(length_increases) / len(length_increases),
                    'question_count': len(scores)
                }
        
        # 모델간 비교
        performance_comparison = {}
        if len(model_stats) == 2:
            gpt_score = model_stats.get('GPT-4o', {}).get('avg_improvement_score', 0)
            claude_score = model_stats.get('Claude-3.5', {}).get('avg_improvement_score', 0)
            
            if gpt_score > claude_score:
                performance_comparison['better_improvement'] = 'GPT-4o'
                performance_comparison['score_difference'] = gpt_score - claude_score
            else:
                performance_comparison['better_improvement'] = 'Claude-3.5'
                performance_comparison['score_difference'] = claude_score - gpt_score
                
            gpt_time = model_stats.get('GPT-4o', {}).get('avg_time_increase', 0)
            claude_time = model_stats.get('Claude-3.5', {}).get('avg_time_increase', 0)
            
            if gpt_time < claude_time:
                performance_comparison['faster_processing'] = 'GPT-4o'
            else:
                performance_comparison['faster_processing'] = 'Claude-3.5'
        
        return {
            'model_averages': model_stats,
            'performance_comparison': performance_comparison,
            'evaluation_criteria': {
                'statute_citation_accuracy': '법조문 인용 정확성 (30점)',
                'statute_application_validity': '조문 적용 타당성 (20점)',
                'precedent_relevance': '판례 사안 관련성 (15점)',
                'precedent_accuracy': '판시사항 정확성 (10점)', 
                'legal_reasoning_logic': '법리 논리적 구조 (15점)',
                'practical_applicability': '실무 적용 가능성 (10점)'
            }
        }

def save_results_multiple_formats(results: dict, output_dir: str = "results/legal_accuracy_rag"):
    """결과를 다양한 형식으로 저장"""
    
    output_path = Path(output_dir)
    ensure_directory_exists(output_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON 저장
    json_file = output_path / f"legal_accuracy_results_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"📄 결과 저장: {json_file}")
    
    # 투명성 보고서들을 마크다운으로 저장
    md_file = output_path / f"legal_accuracy_report_{timestamp}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# 법률 정확성 기반 RAG 성능 비교 보고서\n\n")
        f.write(f"**분석 일시**: {results.get('timestamp', '')}\n")
        f.write(f"**분석 버전**: {results.get('version', '')}\n")  
        f.write(f"**평가 방식**: {results.get('evaluation_method', '')}\n\n")
        
        # 각 질문별 투명성 보고서 추가
        for q_id, q_data in results['questions'].items():
            question = q_data.get('question', '')
            f.write(f"## 질문 {q_id[-1]}: {question}\n\n")
            
            transparency_reports = q_data.get('transparency_reports', {})
            for model, report in transparency_reports.items():
                f.write(f"### {model} 평가 보고서\n\n")
                f.write(report)
                f.write("\n\n")
    
    print(f"📋 투명성 보고서 저장: {md_file}")
    
    return str(json_file), str(md_file)

def main():
    """메인 실행 함수 (법률 정확성 중심 버전)"""
    
    print("⚖️ 법률 정확성 기반 RAG 성능 비교 시스템 v08240001 시작")
    
    # 환경 변수 먼저 로드
    load_dotenv()
    
    # 버전 관리자 초기화
    version_manager = VersionManager()
    version_manager.logger.info("=== 법률 정확성 기반 RAG 분석 시작 v08240001 ===")
    
    # LangSmith 설정
    cfg = OmegaConf.create({
        'langsmith': {
            'enabled': True,
            'project_name': 'legal-accuracy-rag-v08240001',
            'session_name': f'legal-accuracy-session_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    })
    
    langsmith_manager = LangSmithSimple(cfg, version_manager)
    
    # 비교 시스템 초기화
    comparator = RAGLegalAccuracyComparator(version_manager, langsmith_manager)
    
    # 법률 전문 테스트 질문 세트
    test_questions = [
        "취업규칙을 근로자에게 불리하게 변경할 때 사용자가 지켜야 할 법적 요건은 무엇인가요?",
        "퇴직금 지급 기일을 연장하는 합의를 했더라도 연장된 기일까지 지급하지 않으면 형사처벌을 받나요?",
        "부당해고 구제신청의 요건과 절차는 어떻게 되나요?"
    ]
    
    def progress_callback(progress: float):
        print(f"진행률: {progress*100:.1f}%")
    
    try:
        print(f"\n📋 분석 대상 질문 {len(test_questions)}개")
        print("🔍 평가 방식: 법률 정확성 중심 (법조문 50% + 판례 25% + 논리 15% + 실무 10%)")
        print("🤖 반자동화: AI 분석 + 사람 검증 필요")
        print("🔍 투명성: 모든 점수 산출 과정 상세 표시")
        
        # 성능 비교 실행
        results = comparator.compare_models(
            test_questions, 
            temperature=0.1,
            progress_callback=progress_callback
        )
        
        # 결과 저장
        json_path, md_path = save_results_multiple_formats(results)
        
        print(f"\n✅ 법률 정확성 분석 완료!")
        print(f"📊 JSON 결과: {json_path}")
        print(f"📋 투명성 보고서: {md_path}")
        print(f"🔍 LangSmith 프로젝트: {cfg.langsmith.project_name}")
        
        # 요약 출력
        summary = results['summary']
        model_averages = summary.get('model_averages', {})
        
        print(f"\n🏆 모델별 평균 법률 정확성 점수:")
        for model, stats in model_averages.items():
            print(f"  {model}: {stats.get('avg_improvement_score', 0):.1f}/100점")
        
        performance_comparison = summary.get('performance_comparison', {})
        if 'better_improvement' in performance_comparison:
            winner = performance_comparison['better_improvement']
            score_diff = performance_comparison.get('score_difference', 0)
            print(f"\n🥇 법률 정확성 우위: {winner} (+{score_diff:.1f}점)")
            
    except Exception as e:
        version_manager.logger.error(f"분석 중 오류 발생: {e}")
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    main()