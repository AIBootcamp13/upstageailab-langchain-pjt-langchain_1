from typing import List, Dict

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from omegaconf import DictConfig

from rag.retriever.base_extended import ExtendedBaseRetriever
from rag.utils.logger import get_logger

logger = get_logger(__name__)


def _make_search_text(doc: Document):
    case_name = doc.metadata.get("case_name", "")
    origin_text = doc.page_content
    return (case_name + " " + origin_text).lower()


class NaiveRetriever(ExtendedBaseRetriever):
    docs: List[Document] = []

    def build(self, corpus: List[Document], cfg: DictConfig):
        self.docs: List[Document] = corpus

    def extract_question(self, query: str | Dict):
        return query["question"]

    def _get_relevant_documents(
        self, query: str | Dict, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if not self.docs:
            return []

        scored_cases = []
        question = self.extract_question(query)
        for doc in self.docs:
            score = 0
            search_text = _make_search_text(doc)
            for keyword in question.lower().split():
                if len(keyword) == 1:  # 한 글자 키워드 제외
                    continue
                count = search_text.count(keyword)
                score += count * len(keyword)  # 긴 키워드 가중치
            if score > 0:
                scored_cases.append((doc, score))

        scored_cases.sort(key=lambda x: x[1], reverse=True)
        selected_cases = [case[0] for case in scored_cases[: self.top_k]]
        logger.info(f"매칭 자료 개수: {len(selected_cases)}")

        return selected_cases
