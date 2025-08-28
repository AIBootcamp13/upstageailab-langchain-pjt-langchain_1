from deprecated import deprecated
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableWithMessageHistory

from rag.pipeline.pipeline_base import BasePipeline
from rag.reference.naive_referer import law_docs_to_ref
from rag.utils.logger import get_logger

logger = get_logger(__name__)


class SimplePipeline(BasePipeline):

    def _define_chain(self):
        head_node = {
            self.qa_key: RunnablePassthrough(),
            self.ref_key: self.retriever | RunnableLambda(law_docs_to_ref),
        }

        chain = head_node | self.prompt.to_chain(self.hist_key) | self.llm.chat_with() | StrOutputParser()
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history=self.chat_history.get_session_history,
            input_messages_key=self.qa_key,
            history_messages_key=self.hist_key,
        )

        return chain_with_history
