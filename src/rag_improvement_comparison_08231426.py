#!/usr/bin/env python3
"""
RAG 성능 개선 비교 시스템 v08231426
순수 LLM vs RAG 적용 모델의 성능 개선도를 측정하고 비교 분석

비교 대상:
1. GPT-4o (순수) vs GPT-4o (RAG)
2. Claude-3.5 (순수) vs Claude-3.5 (RAG) 
3. 모델간 RAG 개선 효과 비교
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


class LawCaseLoader:
    """법률 판례 로더 및 검색기"""
    
    def __init__(self, law_data_dir: str = "data/law"):
        self.law_data_dir = Path(law_data_dir)
        self.cases = []
        
    def load_cases(self):
        """모든 판례 로드"""
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
    
    def search_relevant_cases(self, question: str, top_k: int = 3) -> list:
        """질문과 관련된 판례 검색 (키워드 기반)"""
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
        return [case[0] for case in scored_cases[:top_k]]


class RAGImprovementComparator:
    """RAG 성능 개선 비교 분석기"""
    
    def __init__(self, version_manager: VersionManager):
        self.version_manager = version_manager
        self.openai_client = None
        self.anthropic_client = None
        self.case_loader = LawCaseLoader()
        
        # API 클라이언트 초기화
        load_dotenv()
        
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def get_pure_llm_response(self, model_name: str, question: str, temperature: float = 0.1) -> dict:
        """순수 LLM 응답 (RAG 없음)"""
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
                'cases_used': []
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
    
    def get_rag_response(self, model_name: str, question: str, temperature: float = 0.1) -> dict:
        """RAG 적용 LLM 응답"""
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
                'cases_used': [case['case_number'] for case in relevant_cases]
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
        """RAG 적용으로 인한 개선도 분석"""
        
        if not (pure_result['success'] and rag_result['success']):
            return {
                'overall_score': 0,
                'specificity_improvement': 0,
                'evidence_improvement': 0,
                'length_change': 0,
                'response_time_change': 0,
                'analysis': "분석 불가 (API 오류)"
            }
        
        pure_answer = pure_result['answer']
        rag_answer = rag_result['answer']
        
        # 1. 구체성 개선 (사건번호 인용 여부)
        pure_case_refs = len(re.findall(r'\d{4}[가-힣]+\d+', pure_answer))  # 사건번호 패턴
        rag_case_refs = len(re.findall(r'\d{4}[가-힣]+\d+', rag_answer))
        specificity_improvement = rag_case_refs - pure_case_refs
        
        # 2. 근거 제시 개선 (법조문, 판례 키워드)
        evidence_keywords = ['판례', '판결', '대법원', '법원', '조문', '법률', '규정']
        pure_evidence = sum(pure_answer.lower().count(kw) for kw in evidence_keywords)
        rag_evidence = sum(rag_answer.lower().count(kw) for kw in evidence_keywords)
        evidence_improvement = rag_evidence - pure_evidence
        
        # 3. 답변 길이 변화 (정보량 증가)
        length_change = len(rag_answer) - len(pure_answer)
        
        # 4. 응답 시간 변화
        response_time_change = rag_result['response_time'] - pure_result['response_time']
        
        # 5. 전체적 개선 점수 계산 (0-100점)
        overall_score = min(100, max(0, 
            (specificity_improvement * 20) +  # 사건번호 인용당 20점
            (evidence_improvement * 5) +       # 법률 키워드당 5점
            (min(length_change, 500) / 10)     # 길이 증가분 최대 50점
        ))
        
        # 6. 분석 요약
        analysis_parts = []
        
        if specificity_improvement > 0:
            analysis_parts.append(f"구체성 향상: {specificity_improvement}개 사건번호 추가 인용")
        
        if evidence_improvement > 0:
            analysis_parts.append(f"근거 강화: {evidence_improvement}개 법률 키워드 추가")
        
        if length_change > 0:
            analysis_parts.append(f"정보량 증가: {length_change:,}글자 추가")
        
        if not analysis_parts:
            analysis_parts.append("개선 효과 미미")
        
        analysis = " / ".join(analysis_parts)
        
        return {
            'overall_score': round(overall_score, 1),
            'specificity_improvement': specificity_improvement,
            'evidence_improvement': evidence_improvement, 
            'length_change': length_change,
            'response_time_change': round(response_time_change, 2),
            'analysis': analysis
        }
    
    def compare_models(self, questions: list, temperature: float = 0.1) -> dict:
        """모델별 RAG 성능 개선 비교"""
        
        # 판례 로드
        self.case_loader.load_cases()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'questions': {},
            'summary': {}
        }
        
        models = ['GPT-4o', 'Claude-3.5']
        
        for q_idx, question in enumerate(questions, 1):
            question_id = f"q{q_idx}"
            results['questions'][question_id] = {
                'question': question,
                'responses': {},
                'improvements': {}
            }
            
            print(f"\n{'='*60}")
            print(f"질문 {q_idx}: {question}")
            print('='*60)
            
            for model in models:
                print(f"\n--- {model} 분석 중 ---")
                
                # 순수 LLM 응답
                print("순수 LLM 답변 생성 중...")
                pure_result = self.get_pure_llm_response(model, question, temperature)
                
                # RAG 적용 응답  
                print("RAG 적용 답변 생성 중...")
                rag_result = self.get_rag_response(model, question, temperature)
                
                # 결과 저장
                results['questions'][question_id]['responses'][model] = {
                    'pure': pure_result,
                    'rag': rag_result
                }
                
                # 개선도 분석
                improvement = self.analyze_improvement(pure_result, rag_result)
                results['questions'][question_id]['improvements'][model] = improvement
                
                # 진행 상황 출력
                if pure_result['success'] and rag_result['success']:
                    print(f"✅ {model} 완료 - 개선 점수: {improvement['overall_score']:.1f}/100")
                    print(f"   순수 답변: {len(pure_result['answer'])}글자 ({pure_result['response_time']:.2f}초)")
                    print(f"   RAG 답변: {len(rag_result['answer'])}글자 ({rag_result['response_time']:.2f}초)")
                    print(f"   사용 판례: {rag_result['case_count']}건")
                else:
                    print(f"❌ {model} 오류 발생")
        
        # 전체 요약 통계 생성
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: dict) -> dict:
        """전체 분석 결과 요약"""
        summary = {
            'total_questions': len(results['questions']),
            'model_averages': {},
            'best_improvements': {},
            'performance_comparison': {}
        }
        
        models = ['GPT-4o', 'Claude-3.5']
        
        for model in models:
            scores = []
            time_changes = []
            
            for q_data in results['questions'].values():
                if model in q_data['improvements']:
                    improvement = q_data['improvements'][model]
                    scores.append(improvement['overall_score'])
                    time_changes.append(improvement['response_time_change'])
            
            if scores:
                summary['model_averages'][model] = {
                    'avg_improvement_score': round(sum(scores) / len(scores), 1),
                    'avg_time_increase': round(sum(time_changes) / len(time_changes), 2),
                    'questions_analyzed': len(scores)
                }
        
        return summary


def main():
    """메인 실행 함수"""
    
    print("🚀 RAG 성능 개선 비교 시스템 v08231426 시작")
    
    # 버전 관리자 초기화
    version_manager = VersionManager()
    version_manager.logger.info("=== RAG 성능 개선 비교 분석 시작 v08231426 ===")
    
    # LangSmith 설정
    cfg = OmegaConf.create({
        'langsmith': {
            'enabled': True,
            'project_name': 'law-rag-improvement-comparison',
            'session_name': f'rag-improvement-session_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    })
    
    langsmith_manager = LangSmithSimple(cfg, version_manager)
    
    # 비교 시스템 초기화
    comparator = RAGImprovementComparator(version_manager)
    
    # 테스트 질문 세트
    test_questions = [
        "취업규칙을 근로자에게 불리하게 변경할 때 사용자가 지켜야 할 법적 요건은 무엇인가요?",
        "퇴직금 지급 기일을 연장하는 합의를 했더라도 연장된 기일까지 지급하지 않으면 형사처벌을 받나요?",
        "부당해고 구제신청의 요건과 절차는 어떻게 되나요?"
    ]
    
    try:
        # 비교 분석 실행
        results = comparator.compare_models(test_questions)
        
        # 결과 저장
        output_dir = ensure_directory_exists("results/rag_improvement_comparison")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 결과 저장
        json_output_path = Path(output_dir) / f"rag_improvement_results_{timestamp}.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 마크다운 보고서 생성
        report_path = Path(output_dir) / f"rag_improvement_report_{timestamp}.md"
        generate_markdown_report(results, report_path)
        
        print(f"\n🎉 RAG 성능 개선 분석 완료!")
        print(f"📄 JSON 결과: {json_output_path}")
        print(f"📊 분석 보고서: {report_path}")
        
        version_manager.logger.info(f"RAG 성능 개선 분석 완료 - 결과: {json_output_path}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        version_manager.logger.error(f"RAG 성능 개선 분석 중 오류: {e}")
        raise


def generate_markdown_report(results: dict, report_path: Path):
    """마크다운 분석 보고서 생성"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# RAG 성능 개선 비교 분석 보고서\n\n")
        f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 요약 정보
        summary = results.get('summary', {})
        f.write("## 📊 분석 요약\n\n")
        f.write(f"- **분석 질문 수**: {summary.get('total_questions', 0)}개\n")
        
        if 'model_averages' in summary:
            for model, avg_data in summary['model_averages'].items():
                f.write(f"- **{model} 평균 개선 점수**: {avg_data.get('avg_improvement_score', 0):.1f}/100\n")
                f.write(f"- **{model} 평균 처리 시간 증가**: {avg_data.get('avg_time_increase', 0):.2f}초\n")
        
        f.write("\n## 🔍 질문별 상세 분석\n\n")
        
        # 질문별 결과
        for q_id, q_data in results.get('questions', {}).items():
            f.write(f"### {q_id.upper()}. {q_data['question']}\n\n")
            
            for model in ['GPT-4o', 'Claude-3.5']:
                if model in q_data.get('improvements', {}):
                    improvement = q_data['improvements'][model]
                    responses = q_data['responses'][model]
                    
                    f.write(f"#### {model} 개선 분석\n")
                    f.write(f"- **개선 점수**: {improvement['overall_score']:.1f}/100\n")
                    f.write(f"- **분석 결과**: {improvement['analysis']}\n")
                    f.write(f"- **응답 시간 변화**: +{improvement['response_time_change']:.2f}초\n")
                    f.write(f"- **사용된 판례**: {responses['rag'].get('case_count', 0)}건\n\n")
                    
                    # 순수 vs RAG 답변 비교
                    f.write(f"**순수 {model} 답변 ({len(responses['pure']['answer'])}글자):**\n")
                    f.write(f"```\n{responses['pure']['answer'][:300]}{'...' if len(responses['pure']['answer']) > 300 else ''}\n```\n\n")
                    
                    f.write(f"**RAG 적용 {model} 답변 ({len(responses['rag']['answer'])}글자):**\n")
                    f.write(f"```\n{responses['rag']['answer'][:300]}{'...' if len(responses['rag']['answer']) > 300 else ''}\n```\n\n")
                    
                    if responses['rag'].get('cases_used'):
                        f.write(f"**참조된 판례**: {', '.join(responses['rag']['cases_used'])}\n\n")
                    
                    f.write("---\n\n")


if __name__ == "__main__":
    main()