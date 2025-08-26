from typing import List, Dict

from rag.retriever.base import BaseRetriever


class FrequencyRetriever(BaseRetriever):
    def __init__(self, corpus: List[Dict] = None, top_k: int = 3):
        self.top_k = top_k
        self.docs = corpus or []

    def retrieve(self, query: str) -> List[Dict]:
        scored = []
        q = query.lower().split()

        for i, doc in enumerate(self.docs):
            title = doc.get("title", "")
            case_law_text = doc.get("text", "")
            text = (title + " " + case_law_text).lower()
            score = sum(text.count(tok) * len(tok) for tok in q if len(tok) > 1)
            if score > 0:
                scored.append({**doc, "id": doc.get("id", i), "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: self.top_k]
