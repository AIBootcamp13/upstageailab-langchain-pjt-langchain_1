from typing import List, Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class ExtendedBaseRetriever(BaseRetriever):
    top_k: int

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def build(self, corpus: List[Document]):
        pass

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        raise NotImplementedError()
