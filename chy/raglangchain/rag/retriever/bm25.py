from typing import List, Dict

from langchain_core.retrievers import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self._bm25 = None
        self._docs = []

    def build(self, corpus: List[Dict]):
        from rank_bm25 import BM25Okapi

        texts = [(d.get("title", "") + " " + d.get("text", "")).lower().split() for d in corpus]
        self._bm25 = BM25Okapi(texts)
        self._docs = corpus

    def retrieve(self, query: str) -> List[Dict]:
        if not self._bm25:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            [
                {"doc": self._docs[i], "score": scores[i], "id": self._docs[i].get("id", i)}
                for i in range(len(self._docs))
            ],
            key=lambda x: x["score"],
            reverse=True,
        )[: self.top_k]
        return [{**r["doc"], "score": float(r["score"]), "id": r["id"]} for r in ranked]
