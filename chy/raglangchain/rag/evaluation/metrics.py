"""
TODO
1. 키워드의 경우 기본 키워드 + 특정 문서 특화 단어를 활용해야 함
2. 쿼리–컨텍스트–정답(GT) 세트 준비
3. 실험 종류
   리트리버 스위치(BM25/FS/Hybrid/ColBERT/Elastic/Weaviate/PGV),
   리랭커(BGE, monoT5 등),
   컨텍스트 윈도우/슬라이딩/Chunker 전략, LLM 라우팅
4. 매트릭
   Retrieval: Recall@k, MRR, nDCG, Coverage
   Generation: Faithfulness(Attributable QA), Answer F1/Exact Match, Hallucination rate, Toxicity, Groundedness
   운영: p50/p95 지연, 비용/토큰, 캐시 히트율
5. 분석 관점
- 비용, 성능, 지연시간 간 트레이드오프 최적화:
  모델/토큰단가, 인퍼런스 지연, 품질(정확성·충실성·사실성) 간 균형점을 사내 데이터 분포에 맞춰 정량화
- 도메인 적합성 검증:
  회귀 방지와 지속 개선: 모델/인덱스 업데이트 시 자동 회귀 테스트로 품질 하락 조기 탐지, 실험 이력과 메트릭의 추적성 확보
"""

# fmt: off
LEGAL_KEYWORDS = [
    "법률", "조문", "판례", "법원", "대법원", "민법", "형법", "근로기준법",
    "상법", "헌법", "규정", "위반", "처벌", "손해배상", "소송", "판결", "항소",
    "상고", "재판", "선고", "형사", "민사", "행정", "헌재"
]
# fmt: on


def compute_specificity(answer: str, case_numbers=None) -> int:
    if not answer:
        return 0
    case_numbers = case_numbers or []
    a = answer.lower()
    return sum(1 for c in case_numbers if c and c.lower() in a)


def compute_legal_keyword_density(answer: str) -> float:
    if not answer:
        return 0.0
    a = answer.lower()
    count = sum(a.count(k) for k in LEGAL_KEYWORDS)
    return (count / max(len(answer), 1)) * 1000


def improvement_score(specificity: int, keyword_delta: float, length_delta: int, case_count: int) -> float:
    return min(100, max(0, specificity * 20 + keyword_delta * 5 + min(length_delta, 500) / 10 + case_count * 5))
