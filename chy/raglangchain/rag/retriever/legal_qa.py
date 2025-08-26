from typing import List, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun as cbm
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag.utils.logger import get_logger

logger = get_logger(__name__)


class LegalQARetriever(BaseRetriever):
    def __init__(self, index_path: str, top_k: int, **kwargs: Any):
        super().__init__(index_path=index_path, top_k=top_k, **kwargs)

    def _get_relevant_documents(self, query: str, *, run_manager: cbm) -> List[Document]:
        text = f"'{query}'에 대한 검색 결과입니다. 이 내용은 직접 구현한 검색 엔진에서 가져왔습니다."
        return [Document(page_content=text)]

    async def _aget_relevant_documents(self, query: str, *, run_manager: cbm) -> List[Document]:
        text = f"'{query}'에 대한 비동기 검색 결과입니다."
        return [Document(page_content=text)]
