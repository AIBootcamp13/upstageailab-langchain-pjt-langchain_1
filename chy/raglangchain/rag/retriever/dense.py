from typing import List, Dict

from langchain_core.retrievers import BaseRetriever


class DenseRetriever(BaseRetriever):
    def __init__(
            self,
            index_path: str = "indexes/dense.faiss",
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            top_k: int = 3,
    ):
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.top_k = top_k
        self._index = None
        self._emb = None
        self._docs = []

    def build(self, corpus: List[Dict]):
        from sentence_transformers import SentenceTransformer
        import faiss

        self._emb = SentenceTransformer(self.embedding_model)
        texts = [d.get("title", "") + " " + d.get("text", "") for d in corpus]
        mat = self._emb.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        dim = mat.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(mat)
        index.add(mat)
        self._index = index
        self._docs = corpus

    def retrieve(self, query: str) -> List[Dict]:
        if self._index is None or self._emb is None:
            return []
        import faiss

        qv = self._emb.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(qv)
        D, I = self._index.search(qv, self.top_k)
        out = []
        for rank, (idx, score) in enumerate(zip(I[0], D[0])):
            if idx == -1:
                continue
            doc = self._docs[int(idx)]
            out.append({**doc, "score": float(score), "id": doc.get("id", int(idx))})
        return out
