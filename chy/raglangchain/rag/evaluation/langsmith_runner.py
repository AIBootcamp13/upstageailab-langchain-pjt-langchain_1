from rag.evaluation.metrics import compute_specificity, compute_legal_keyword_density, improvement_score
from rag.utils.langsmith_utils import maybe_traceable


class LangSmithEvaluatorRunner:
    def __init__(self, enabled: bool, project: str, tags: list):
        self.enabled = enabled
        self.project = project
        self.tags = tags

    def run_single(self, llm_label: str, retriever_label: str, question: str, pure_resp: dict, rag_resp: dict) -> dict:
        pure_answer = (pure_resp or {}).get("answer", "")
        rag_answer = (rag_resp or {}).get("answer", "")
        case_numbers = [m.get("id") or "" for m in (rag_resp or {}).get("used_docs", [])]
        spec = compute_specificity(rag_answer, case_numbers)
        pure_kw = compute_legal_keyword_density(pure_answer)
        rag_kw = compute_legal_keyword_density(rag_answer)
        kw_delta = max(0, rag_kw - pure_kw)
        length_delta = len(rag_answer) - len(pure_answer)
        case_count = (rag_resp or {}).get("doc_count", 0)
        score = improvement_score(spec, kw_delta, length_delta, case_count)

        payload = {
            "llm": llm_label,
            "retriever": retriever_label,
            "question": question,
            "metrics": {
                "specificity": spec,
                "legal_keyword_density_rag": rag_kw,
                "legal_keyword_density_pure": pure_kw,
                "legal_keyword_density_delta": kw_delta,
                "length_delta": length_delta,
                "case_count": case_count,
                "improvement_score": score,
            },
            "tags": self.tags,
            "project": self.project,
        }

        if self.enabled:
            with maybe_traceable("rag_comparison", metadata={"project": self.project, "tags": self.tags}) as trace:
                trace(payload)

        return payload
