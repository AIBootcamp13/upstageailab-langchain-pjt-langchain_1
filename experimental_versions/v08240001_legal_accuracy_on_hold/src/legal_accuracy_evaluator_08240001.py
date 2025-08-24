#!/usr/bin/env python3
"""
법률 정확성 평가 시스템 v08240001
법조문 인용 정확성 중심의 객관적이고 투명한 평가 시스템
반자동화: AI 분석 + 사람의 최종 검증
"""

import os
import re
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

# 프로젝트 루트 추가
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.version_manager import VersionManager

@dataclass
class LegalAccuracyScore:
    """법률 정확성 평가 점수 구조체"""
    # 법조문 정확성 (50점)
    statute_citation_accuracy: float = 0.0    # 정확한 조문 인용 (30점)
    statute_application_validity: float = 0.0  # 조문 적용 타당성 (20점)
    
    # 판례 적절성 (25점)
    precedent_relevance: float = 0.0          # 사안 관련성 (15점)
    precedent_accuracy: float = 0.0           # 판시사항 정확성 (10점)
    
    # 법리 논리성 (15점)
    legal_reasoning_logic: float = 0.0        # 논리적 추론 구조 (15점)
    
    # 실무 적용성 (10점)
    practical_applicability: float = 0.0     # 구체적 해결방안 (10점)
    
    def total_score(self) -> float:
        """총점 계산"""
        return (self.statute_citation_accuracy + 
                self.statute_application_validity +
                self.precedent_relevance + 
                self.precedent_accuracy +
                self.legal_reasoning_logic +
                self.practical_applicability)

@dataclass 
class EvaluationDetail:
    """평가 상세 내역"""
    criterion: str                    # 평가 기준
    max_score: float                 # 만점
    actual_score: float              # 실제 점수
    evaluation_reason: str           # 평가 근거
    evidence_texts: List[str]        # 근거 텍스트들
    ai_analysis: str                 # AI 분석 내용
    human_verification: str          # 사람 검증 내용
    confidence_level: float          # 신뢰도 (0-1)

class LegalStatuteDatabase:
    """법조문 데이터베이스"""
    
    def __init__(self):
        self.labor_law_articles = {
            # 근로기준법 주요 조문
            "근로기준법 제93조": {
                "content": "사용자는 취업규칙을 작성하거나 변경할 때에는 해당 사업 또는 사업장에 근로자의 과반수로 조직된 노동조합이 있는 경우에는 그 노동조합, 근로자의 과반수로 조직된 노동조합이 없는 경우에는 근로자의 과반수의 의견을 들어야 한다.",
                "keywords": ["취업규칙", "변경", "노동조합", "과반수", "의견청취"],
                "related_concepts": ["불리한 변경", "동의", "의견청취"]
            },
            "근로기준법 제94조": {
                "content": "사용자는 취업규칙을 근로자에게 불리하게 변경하는 경우에는 해당 사업 또는 사업장에 근로자의 과반수로 조직된 노동조합이 있는 경우에는 그 노동조합, 근로자의 과반수로 조직된 노동조합이 없는 경우에는 근로자의 과반수의 동의를 받아야 한다.",
                "keywords": ["불리한 변경", "동의", "노동조합", "과반수"],
                "related_concepts": ["취업규칙", "변경", "동의"]
            },
            "근로기준법 제36조": {
                "content": "사용자는 퇴직하는 근로자에게 그 지급사유가 발생한 때부터 14일 이내에 임금, 보상금, 그 밖에 일체의 금품을 지급하여야 한다.",
                "keywords": ["퇴직금", "14일", "지급기한", "금품"],
                "related_concepts": ["퇴직급여", "지급연기", "합의"]
            },
            "근로기준법 제109조": {
                "content": "제36조를 위반한 자는 3년 이하의 징역 또는 3천만원 이하의 벌금에 처한다.",
                "keywords": ["형사처벌", "3년", "3천만원", "벌금", "징역"],
                "related_concepts": ["퇴직금", "지급의무", "처벌"]
            }
        }
        
        self.precedent_patterns = {
            # 판례 패턴 정의
            "employment_rules_change": {
                "keywords": ["취업규칙", "불리한 변경", "동의", "의견청취"],
                "legal_principles": ["집단적 동의원칙", "기득권 보호"]
            },
            "severance_pay_delay": {
                "keywords": ["퇴직금", "지급기한", "합의", "연장", "형사처벌"],
                "legal_principles": ["지급의무", "기한준수", "형사책임"]
            }
        }

class LegalAccuracyEvaluator:
    """법률 정확성 평가기"""
    
    def __init__(self, version_manager: VersionManager = None):
        self.version_manager = version_manager or VersionManager()
        self.statute_db = LegalStatuteDatabase()
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """로거 설정"""
        logger = logging.getLogger(f'LegalAccuracyEvaluator')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def evaluate_legal_accuracy(self, 
                               pure_answer: str, 
                               rag_answer: str,
                               question: str,
                               case_data: List[Dict]) -> Tuple[LegalAccuracyScore, List[EvaluationDetail]]:
        """
        법률 정확성 종합 평가
        
        Args:
            pure_answer: 순수 LLM 답변
            rag_answer: RAG 적용 답변  
            question: 원본 질문
            case_data: 참조 판례 데이터
            
        Returns:
            평가 점수 및 상세 내역
        """
        
        self.logger.info(f"법률 정확성 평가 시작")
        
        evaluation_details = []
        
        # 1. 법조문 인용 정확성 평가 (50점)
        statute_score, statute_details = self._evaluate_statute_accuracy(
            pure_answer, rag_answer, question
        )
        evaluation_details.extend(statute_details)
        
        # 2. 판례 적절성 평가 (25점)  
        precedent_score, precedent_details = self._evaluate_precedent_accuracy(
            pure_answer, rag_answer, question, case_data
        )
        evaluation_details.extend(precedent_details)
        
        # 3. 법리 논리성 평가 (15점)
        logic_score, logic_details = self._evaluate_legal_reasoning(
            pure_answer, rag_answer, question
        )
        evaluation_details.extend(logic_details)
        
        # 4. 실무 적용성 평가 (10점)
        practical_score, practical_details = self._evaluate_practical_applicability(
            pure_answer, rag_answer, question
        )
        evaluation_details.extend(practical_details)
        
        # 최종 점수 생성
        final_score = LegalAccuracyScore(
            statute_citation_accuracy=statute_score[0],
            statute_application_validity=statute_score[1], 
            precedent_relevance=precedent_score[0],
            precedent_accuracy=precedent_score[1],
            legal_reasoning_logic=logic_score,
            practical_applicability=practical_score
        )
        
        self.logger.info(f"법률 정확성 평가 완료 - 총점: {final_score.total_score():.1f}/100점")
        
        return final_score, evaluation_details
    
    def _evaluate_statute_accuracy(self, pure_answer: str, rag_answer: str, question: str) -> Tuple[Tuple[float, float], List[EvaluationDetail]]:
        """법조문 인용 정확성 평가 (50점)"""
        
        details = []
        
        # 1-1. 정확한 조문 인용 (30점)
        pure_statutes = self._extract_statute_references(pure_answer)
        rag_statutes = self._extract_statute_references(rag_answer)
        
        citation_score = 0.0
        citation_evidence = []
        ai_analysis = ""
        human_verification = "[사람 검증 필요] "
        
        # RAG에서 추가된 법조문 분석
        added_statutes = set(rag_statutes) - set(pure_statutes)
        
        if added_statutes:
            for statute in added_statutes:
                if statute in self.statute_db.labor_law_articles:
                    # 관련성 확인
                    relevance = self._check_statute_relevance(statute, question)
                    if relevance > 0.7:  # 높은 관련성
                        citation_score += 15.0  # 정확한 인용마다 15점
                        citation_evidence.append(f"✅ {statute} (관련성: {relevance:.2f})")
                        ai_analysis += f"적절한 법조문 인용: {statute}. "
                    elif relevance > 0.4:  # 보통 관련성
                        citation_score += 8.0
                        citation_evidence.append(f"🔶 {statute} (관련성: {relevance:.2f})")
                        ai_analysis += f"부분적으로 관련된 법조문: {statute}. "
                    else:
                        citation_evidence.append(f"❌ {statute} (관련성: {relevance:.2f})")
                        ai_analysis += f"관련성이 낮은 법조문: {statute}. "
                        human_verification += f"{statute} 관련성 재검토 필요. "
                else:
                    citation_evidence.append(f"❓ {statute} (DB에 없음)")
                    human_verification += f"{statute} 존재 여부 및 정확성 확인 필요. "
        
        citation_score = min(30.0, citation_score)  # 최대 30점
        
        details.append(EvaluationDetail(
            criterion="법조문 인용 정확성",
            max_score=30.0,
            actual_score=citation_score,
            evaluation_reason=f"RAG로 {len(added_statutes)}개 법조문 추가 인용",
            evidence_texts=citation_evidence,
            ai_analysis=ai_analysis,
            human_verification=human_verification,
            confidence_level=0.8 if citation_evidence else 0.3
        ))
        
        # 1-2. 조문 적용 타당성 (20점) 
        application_score = 0.0
        application_evidence = []
        application_analysis = ""
        
        if added_statutes:
            for statute in added_statutes:
                if statute in self.statute_db.labor_law_articles:
                    # 적용 방식 분석
                    application_quality = self._analyze_statute_application(statute, rag_answer, question)
                    application_score += application_quality * 10  # 최대 10점씩
                    
                    if application_quality > 0.7:
                        application_evidence.append(f"✅ {statute} 올바르게 적용됨")
                        application_analysis += f"{statute}이 적절하게 적용됨. "
                    elif application_quality > 0.4:
                        application_evidence.append(f"🔶 {statute} 부분적으로 적용됨") 
                        application_analysis += f"{statute}이 부분적으로 적용됨. "
                    else:
                        application_evidence.append(f"❌ {statute} 부적절하게 적용됨")
                        application_analysis += f"{statute}이 부적절하게 적용됨. "
        
        application_score = min(20.0, application_score)  # 최대 20점
        
        details.append(EvaluationDetail(
            criterion="법조문 적용 타당성", 
            max_score=20.0,
            actual_score=application_score,
            evaluation_reason=f"인용된 법조문의 적용 방식 분석",
            evidence_texts=application_evidence,
            ai_analysis=application_analysis,
            human_verification="[사람 검증 필요] 법조문 적용의 법리적 타당성 확인",
            confidence_level=0.7
        ))
        
        return (citation_score, application_score), details
    
    def _evaluate_precedent_accuracy(self, pure_answer: str, rag_answer: str, question: str, case_data: List[Dict]) -> Tuple[Tuple[float, float], List[EvaluationDetail]]:
        """판례 적절성 평가 (25점)"""
        
        details = []
        
        # 2-1. 사안 관련성 (15점)
        relevance_score = 0.0
        relevance_evidence = []
        
        # RAG에서 사용된 판례들
        used_cases = self._extract_case_references(rag_answer)
        
        if case_data and used_cases:
            for case_info in case_data:
                case_number = case_info.get('case_number', '')
                if any(case_number in ref for ref in used_cases):
                    # 사안 유사성 분석
                    similarity = self._calculate_case_similarity(question, case_info)
                    if similarity > 0.8:
                        relevance_score += 7.5  # 매우 관련있는 판례
                        relevance_evidence.append(f"✅ {case_number} (유사도: {similarity:.2f})")
                    elif similarity > 0.6:
                        relevance_score += 5.0   # 관련있는 판례  
                        relevance_evidence.append(f"🔶 {case_number} (유사도: {similarity:.2f})")
                    elif similarity > 0.4:
                        relevance_score += 2.5   # 부분 관련 판례
                        relevance_evidence.append(f"🔸 {case_number} (유사도: {similarity:.2f})")
                    else:
                        relevance_evidence.append(f"❌ {case_number} (유사도: {similarity:.2f})")
        
        relevance_score = min(15.0, relevance_score)
        
        details.append(EvaluationDetail(
            criterion="판례 사안 관련성",
            max_score=15.0, 
            actual_score=relevance_score,
            evaluation_reason=f"{len(used_cases)}개 판례 활용, 평균 관련성 분석",
            evidence_texts=relevance_evidence,
            ai_analysis=f"사용된 판례들의 질문과의 유사도 분석 완료",
            human_verification="[사람 검증 필요] 판례 사안의 실질적 관련성 확인",
            confidence_level=0.75
        ))
        
        # 2-2. 판시사항 정확성 (10점) 
        accuracy_score = 0.0
        accuracy_evidence = []
        
        if case_data:
            for case_info in case_data:
                # 판시사항이 올바르게 반영되었는지 확인
                holding_accuracy = self._check_holding_accuracy(rag_answer, case_info)
                accuracy_score += holding_accuracy * 5  # 최대 5점씩
                
                if holding_accuracy > 0.7:
                    accuracy_evidence.append(f"✅ {case_info.get('case_number', '')} 판시사항 정확 반영")
                elif holding_accuracy > 0.4:
                    accuracy_evidence.append(f"🔶 {case_info.get('case_number', '')} 판시사항 부분 반영")
                else:
                    accuracy_evidence.append(f"❌ {case_info.get('case_number', '')} 판시사항 왜곡")
        
        accuracy_score = min(10.0, accuracy_score)
        
        details.append(EvaluationDetail(
            criterion="판시사항 정확성",
            max_score=10.0,
            actual_score=accuracy_score, 
            evaluation_reason="판례의 핵심 판시사항 반영도 분석",
            evidence_texts=accuracy_evidence,
            ai_analysis="각 판례의 판시사항과 답변 내용 비교 분석",
            human_verification="[사람 검증 필수] 판시사항 해석의 법리적 정확성 검토",
            confidence_level=0.6
        ))
        
        return (relevance_score, accuracy_score), details
    
    def _evaluate_legal_reasoning(self, pure_answer: str, rag_answer: str, question: str) -> Tuple[float, List[EvaluationDetail]]:
        """법리 논리성 평가 (15점)"""
        
        # 논리적 구조 분석: 전제 → 추론 → 결론
        logic_score = 0.0
        logic_evidence = []
        
        # RAG 답변의 논리적 구조 분석
        has_premise = self._check_legal_premise(rag_answer)
        has_reasoning = self._check_legal_reasoning(rag_answer)  
        has_conclusion = self._check_legal_conclusion(rag_answer)
        
        if has_premise:
            logic_score += 5.0
            logic_evidence.append("✅ 법적 전제 명확히 제시")
        else:
            logic_evidence.append("❌ 법적 전제 부족")
            
        if has_reasoning:
            logic_score += 7.0
            logic_evidence.append("✅ 논리적 추론 과정 포함")
        else:
            logic_evidence.append("❌ 추론 과정 불충분")
            
        if has_conclusion:
            logic_score += 3.0
            logic_evidence.append("✅ 명확한 결론 제시")
        else:
            logic_evidence.append("❌ 결론 부족")
        
        detail = EvaluationDetail(
            criterion="법리 해석 논리성",
            max_score=15.0,
            actual_score=logic_score,
            evaluation_reason="전제→추론→결론의 논리적 구조 분석",
            evidence_texts=logic_evidence,
            ai_analysis=f"논리 구조: 전제({has_premise}) + 추론({has_reasoning}) + 결론({has_conclusion})",
            human_verification="[사람 검증 필요] 법리적 추론의 타당성 검토",
            confidence_level=0.8
        )
        
        return logic_score, [detail]
    
    def _evaluate_practical_applicability(self, pure_answer: str, rag_answer: str, question: str) -> Tuple[float, List[EvaluationDetail]]:
        """실무 적용성 평가 (10점)"""
        
        practical_score = 0.0
        practical_evidence = []
        
        # 구체적 해결방안 제시 여부
        has_specific_solution = self._check_specific_solution(rag_answer)
        has_procedural_steps = self._check_procedural_steps(rag_answer) 
        has_risk_warning = self._check_risk_warning(rag_answer)
        
        if has_specific_solution:
            practical_score += 5.0
            practical_evidence.append("✅ 구체적 해결방안 제시")
        else:
            practical_evidence.append("❌ 추상적 답변")
            
        if has_procedural_steps:
            practical_score += 3.0
            practical_evidence.append("✅ 절차적 단계 포함")
        else:
            practical_evidence.append("❌ 절차 설명 부족")
            
        if has_risk_warning:
            practical_score += 2.0
            practical_evidence.append("✅ 위험요소 경고")
        else:
            practical_evidence.append("❌ 리스크 언급 없음")
        
        detail = EvaluationDetail(
            criterion="실무 적용 가능성",
            max_score=10.0,
            actual_score=practical_score,
            evaluation_reason="구체적 해결방안 및 실행 가능성 분석",
            evidence_texts=practical_evidence,
            ai_analysis=f"실무성: 해결방안({has_specific_solution}) + 절차({has_procedural_steps}) + 경고({has_risk_warning})",
            human_verification="[사람 검증 권장] 실제 실무 환경에서의 적용 가능성 확인",
            confidence_level=0.85
        )
        
        return practical_score, [detail]
    
    # 유틸리티 메서드들
    def _extract_statute_references(self, text: str) -> List[str]:
        """법조문 참조 추출"""
        patterns = [
            r'근로기준법\s*제\s*\d+조',
            r'민법\s*제\s*\d+조', 
            r'상법\s*제\s*\d+조',
            r'노동조합법\s*제\s*\d+조'
        ]
        
        references = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)
        
        return list(set(references))  # 중복 제거
    
    def _extract_case_references(self, text: str) -> List[str]:
        """판례 번호 추출"""
        pattern = r'\d{4}[가-힣]+\d+'
        return re.findall(pattern, text)
    
    def _check_statute_relevance(self, statute: str, question: str) -> float:
        """법조문 관련성 확인"""
        if statute in self.statute_db.labor_law_articles:
            article_info = self.statute_db.labor_law_articles[statute]
            keywords = article_info['keywords']
            
            # 키워드 매칭으로 관련성 계산
            question_lower = question.lower()
            matching_keywords = sum(1 for keyword in keywords if keyword in question_lower)
            return min(1.0, matching_keywords / len(keywords))
        
        return 0.0
    
    def _analyze_statute_application(self, statute: str, answer: str, question: str) -> float:
        """법조문 적용 방식 분석"""
        if statute not in answer:
            return 0.0
            
        # 법조문이 단순 인용만 되었는지, 해석과 함께 적용되었는지 분석
        context_words = ['따라서', '그러므로', '적용하면', '해석하면', '규정에 의해']
        has_context = any(word in answer for word in context_words)
        
        # 구체적 적용 방식
        has_specific_application = len(answer.split(statute)) > 1 and len(answer.split(statute)[1]) > 50
        
        score = 0.0
        if has_context:
            score += 0.5
        if has_specific_application:
            score += 0.5
            
        return score
    
    def _calculate_case_similarity(self, question: str, case_info: Dict) -> float:
        """판례와 질문의 유사도 계산 (간단한 키워드 매칭)"""
        question_words = set(question.lower().split())
        case_summary = case_info.get('summary', '') + case_info.get('facts', '')
        case_words = set(case_summary.lower().split())
        
        if not case_words:
            return 0.0
            
        intersection = question_words.intersection(case_words)
        union = question_words.union(case_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _check_holding_accuracy(self, answer: str, case_info: Dict) -> float:
        """판시사항 정확성 확인"""
        holdings = case_info.get('holdings', '')
        if not holdings:
            return 0.5  # 판시사항 정보 없으면 중간점수
            
        # 핵심 판시사항 키워드가 답변에 포함되었는지 확인
        holding_keywords = holdings.lower().split()[:5]  # 상위 5개 키워드
        answer_lower = answer.lower()
        
        matching = sum(1 for keyword in holding_keywords if keyword in answer_lower)
        return matching / len(holding_keywords) if holding_keywords else 0.0
    
    def _check_legal_premise(self, answer: str) -> bool:
        """법적 전제 확인"""
        premise_indicators = ['법에 따르면', '규정에 의하면', '판례에 의하면', '법리상']
        return any(indicator in answer for indicator in premise_indicators)
    
    def _check_legal_reasoning(self, answer: str) -> bool:
        """논리적 추론 확인"""
        reasoning_indicators = ['따라서', '그러므로', '이에 따라', '결국', '그런데', '하지만', '다만']
        return any(indicator in answer for indicator in reasoning_indicators)
    
    def _check_legal_conclusion(self, answer: str) -> bool:
        """법적 결론 확인"""
        conclusion_indicators = ['결론적으로', '정리하면', '답변하면', '해당됩니다', '해당되지 않습니다']
        return any(indicator in answer for indicator in conclusion_indicators)
    
    def _check_specific_solution(self, answer: str) -> bool:
        """구체적 해결방안 확인"""
        solution_indicators = ['방법은', '절차는', '해야 합니다', '필요합니다', '권고합니다']
        return any(indicator in answer for indicator in solution_indicators)
    
    def _check_procedural_steps(self, answer: str) -> bool:
        """절차적 단계 확인"""
        step_indicators = ['1단계', '첫째', '둘째', '먼저', '다음', '마지막으로', '단계']
        return any(indicator in answer for indicator in step_indicators)
    
    def _check_risk_warning(self, answer: str) -> bool:
        """위험요소 경고 확인"""
        warning_indicators = ['주의', '위험', '문제', '리스크', '유의', '조심', '처벌']
        return any(indicator in answer for indicator in warning_indicators)

def generate_transparency_report(score: LegalAccuracyScore, details: List[EvaluationDetail]) -> str:
    """완전 투명한 평가 보고서 생성"""
    
    report = f"""
# 🏛️ 법률 정확성 평가 보고서 v08240001

**평가 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**평가 방식**: 반자동화 (AI 분석 + 사람 검증 필요)
**신뢰성 지향**: 정밀한 평가 중심

## 📊 종합 평가 점수

| 평가 영역 | 세부 항목 | 점수 | 만점 | 달성률 |
|-----------|-----------|------|------|--------|
| **법조문 정확성** (최우선) | 정확한 조문 인용 | **{score.statute_citation_accuracy:.1f}점** | 30점 | {score.statute_citation_accuracy/30*100:.1f}% |
| | 조문 적용 타당성 | **{score.statute_application_validity:.1f}점** | 20점 | {score.statute_application_validity/20*100:.1f}% |
| **판례 적절성** | 사안 관련성 | **{score.precedent_relevance:.1f}점** | 15점 | {score.precedent_relevance/15*100:.1f}% |
| | 판시사항 정확성 | **{score.precedent_accuracy:.1f}점** | 10점 | {score.precedent_accuracy/10*100:.1f}% |
| **법리 논리성** | 논리적 추론 구조 | **{score.legal_reasoning_logic:.1f}점** | 15점 | {score.legal_reasoning_logic/15*100:.1f}% |
| **실무 적용성** | 구체적 해결방안 | **{score.practical_applicability:.1f}점** | 10점 | {score.practical_applicability/10*100:.1f}% |
| | | | | |
| **🎯 총점** | | **{score.total_score():.1f}점** | **100점** | **{score.total_score():.1f}%** |

"""

    # 상세 평가 내역
    report += "\n## 📋 상세 평가 내역 및 근거\n\n"
    
    for i, detail in enumerate(details, 1):
        confidence_bar = "🟢" * int(detail.confidence_level * 5) + "⚪" * (5 - int(detail.confidence_level * 5))
        
        report += f"""
### {i}. {detail.criterion}

**점수**: {detail.actual_score:.1f}/{detail.max_score:.0f}점 ({detail.actual_score/detail.max_score*100:.1f}%)
**평가 근거**: {detail.evaluation_reason}
**신뢰도**: {confidence_bar} ({detail.confidence_level:.2f})

#### 🤖 AI 자동 분석
{detail.ai_analysis}

#### 📝 발견 증거들
"""
        for evidence in detail.evidence_texts:
            report += f"- {evidence}\n"
        
        report += f"""
#### 👤 사람 검증 필요 사항
{detail.human_verification}

---
"""
    
    # 총 신뢰도 계산
    avg_confidence = sum(d.confidence_level for d in details) / len(details) if details else 0
    report += f"""
## 🎯 평가 신뢰도 및 검증 권고사항

**전체 신뢰도**: {"🟢" * int(avg_confidence * 5) + "⚪" * (5 - int(avg_confidence * 5))} ({avg_confidence:.2f})

### 🔍 사람 검증이 특히 필요한 영역:
"""
    
    low_confidence_items = [d for d in details if d.confidence_level < 0.7]
    if low_confidence_items:
        for item in low_confidence_items:
            report += f"- **{item.criterion}**: {item.human_verification}\n"
    else:
        report += "- 모든 영역의 신뢰도가 양호합니다.\n"
    
    report += f"""
### 💡 개선 권고사항:

1. **법조문 정확성 강화**: 더 많은 관련 법조문 학습 필요
2. **판례 활용 개선**: 사안별 적절한 판례 선별 능력 향상
3. **논리 구조 체계화**: 전제-추론-결론의 명확한 구조 확립
4. **실무 지향성 제고**: 구체적이고 실행 가능한 해답 제시

### ⚖️ 법률 전문가 최종 검토 필수

이 평가는 AI 기반 1차 분석 결과이며, 법률적 판단의 정확성을 위해서는 
**반드시 법률 전문가의 최종 검토가 필요**합니다.

---
*법률 정확성 평가 시스템 v08240001 - 투명하고 정밀한 평가를 위하여*
"""
    
    return report

if __name__ == "__main__":
    # 테스트용 코드
    evaluator = LegalAccuracyEvaluator()
    
    # 샘플 테스트
    pure_answer = "취업규칙 변경시 근로자 동의가 필요합니다."
    rag_answer = "근로기준법 제94조에 따르면, 사용자는 취업규칙을 근로자에게 불리하게 변경하는 경우에는 근로자 과반수의 동의를 받아야 합니다. 2022다200249 판례에서도 이를 명확히 하고 있습니다."
    question = "취업규칙을 근로자에게 불리하게 변경할 때 사용자가 지켜야 할 법적 요건은 무엇인가요?"
    
    case_data = [{"case_number": "2022다200249", "summary": "취업규칙 불리한 변경", "holdings": "동의 필요"}]
    
    score, details = evaluator.evaluate_legal_accuracy(pure_answer, rag_answer, question, case_data)
    report = generate_transparency_report(score, details)
    
    print(report)