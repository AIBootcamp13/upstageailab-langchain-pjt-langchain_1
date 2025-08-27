from typing import Any, List

import faiss
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer

from rag.retriever.base_extended import ExtendedBaseRetriever
from rag.utils.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever(ExtendedBaseRetriever):
    index_path: str
    embedding_model_name: str

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._docs = []
        self._index = None
        self._embedder = self._make_embedder()

    def _make_embedder(self):
        logger.info(f"make embedder model {self.embedding_model_name}")
        return SentenceTransformer(self.embedding_model_name)

    def _chunk_docs(self, docs: List[Document], cfg: DictConfig) -> List[Document]:
        # TODO: text splitter 처리시 원본 문서 정보 메타데이터 기록 여부 체크
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.data.chunk_size, chunk_overlap=cfg.data.chunk_overlap
        )
        return text_splitter.split_documents(docs)

    def build(self, docs: List[Document], cfg: DictConfig):
        logger.info(f"*** building [{len(docs)}] embeddings... ***")
        self._docs = self._chunk_docs(docs, cfg)  # (일단 _docs가 원문과 청크를 보관하는 DB 라고 가정)

        texts = [d.page_content for d in docs]
        mat = self._embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        dim = mat.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(mat)
        index.add(mat)
        self._index = index

        logger.info(f"*** complete building embeddings ***")

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        if self._index is None or self._embedder is None:
            return []

        qv = self._embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(qv)
        D, I = self._index.search(qv, self.top_k)

        outs = []
        for rank, (idx, score) in enumerate(zip(I[0], D[0])):
            if idx == -1:
                continue

            # TODO 튜닝 지점: top-k 여유롭게 선정 후 매칭 점수로 실제 n개 필터링
            doc: Document = self._docs[int(idx)]
            doc.metadata["score"] = float(score)
            outs.append(doc)

        logger.info("=" * 80)
        logger.info(f"매칭 자료 개수: {len(outs)}")
        logger.info(f"매칭 점수 {[d.metadata['score'] for d in outs]}")
        logger.info("=" * 80)

        return outs
