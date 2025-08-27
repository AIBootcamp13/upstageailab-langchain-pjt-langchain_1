from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableLambda
from langsmith import traceable

from rag.pipeline.pipeline_base import BasePipeline
from rag.reference.naive_referer import law_docs_to_ref
from rag.utils.logger import get_logger

logger = get_logger(__name__)


class SimplePipeline(BasePipeline):

    def _run_for_test_none_lcel(self, query: str):
        """for none-LCEL run test"""
        docs = self.retriever.retrieve(query)
        prompt_with_context = self.prompt.to_chain()

        max_tokens = self.cfg.exp.max_tokens
        temperature = self.cfg.exp.temperature
        answer = self.llm.generate(prompt_with_context, temperature=temperature, max_tokens=max_tokens)

        return {"answer": answer, "doc_count": len(docs), "used_docs": docs}

    @traceable
    def _define_chain(self, question: str):
        ins = {
            self.ref_key: self.retriever | RunnableLambda(law_docs_to_ref),
            self.qa_key: RunnablePassthrough(),
        }
        chain = ins | self.prompt.to_chain() | self.llm.chat_with() | StrOutputParser()

        return chain
