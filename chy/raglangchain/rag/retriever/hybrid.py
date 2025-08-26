from typing import List, Dict

from langchain_core.retrievers import BaseRetriever

from rag.retriever.bm25 import BM25Retriever
from rag.retriever.dense import DenseRetriever


class HybridRetriever(BaseRetriever):
    def __init__(self, weights: dict = None, top_k_total: int = 3):
        self.weights = weights or {"bm25": 0.5, "dense": 0.5}
        self.top_k_total = top_k_total
        self._bm25 = BM25Retriever(top_k=top_k_total * 2)
        self._dense = DenseRetriever(top_k=top_k_total * 2)
        self._built = False

    def build(self, corpus: List[Dict]):
        self._bm25.build(corpus)
        self._dense.build(corpus)
        self._built = True

    def retrieve(self, query: str) -> List[Dict]:
        if not self._built:
            return []

        bm = self._bm25.retrieve(query)
        de = self._dense.retrieve(query)

        # 정규화
        def norm(scores):
            if not scores:
                return []
            vals = [s["score"] for s in scores]
            lo, hi = min(vals), max(vals)
            return [(s["id"], 0.0 if hi == lo else (s["score"] - lo) / (hi - lo), s) for s in scores]

        bm_n = norm(bm)
        de_n = norm(de)

        # 가중 합
        agg = {}
        for _id, v, s in bm_n:
            agg[_id] = agg.get(_id, 0.0) + v * self.weights.get("bm25", 0.5)
        for _id, v, s in de_n:
            agg[_id] = agg.get(_id, 0.0) + v * self.weights.get("dense", 0.5)

        # 결과 재정렬
        id2doc = {s["id"]: s for s in (bm + de)}
        ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)[: self.top_k_total]

        return [id2doc[i] for i, _ in ranked if i in id2doc]
