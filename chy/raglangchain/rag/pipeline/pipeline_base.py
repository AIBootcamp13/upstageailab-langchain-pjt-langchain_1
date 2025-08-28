from typing import List, Dict

from hydra.utils import instantiate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langsmith import traceable

from rag.datasets.base import BaseDataset
from rag.history.chat_history import InMemoryChatHistory
from rag.llms.base_llm import BaseLLM
from rag.prompts.qa_prompt import QAPrompt
from rag.utils.logger import get_logger

logger = get_logger(__name__)


def debug_step(input_data):
    print("--- 디버깅 시작 ---")
    print(f"[타입] {type(input_data)}")
    print(f"[내용] {input_data}")
    print("--- 디버깅 끝 ---")
    return input_data


class BasePipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.docs: List[Dict]
        self.dataset: BaseDataset
        self.llm: BaseLLM
        self.prompt: QAPrompt
        self.embedder = None
        self.retriever: BaseRetriever
        self.chat_history: InMemoryChatHistory
        self._build()

    def _build(self):
        logger.info(f"build {self.__class__.__name__}")
        self.dataset: BaseDataset = self._make_dataset()
        self.docs: List[Dict] = self._load_dataset()
        self.embedder = self._make_embedder()
        self.retriever: BaseRetriever = self._make_retriever()
        self.prompt: QAPrompt = self._make_prompt()
        self.llm: BaseLLM = self._make_llm_client()
        self.chat_history: InMemoryChatHistory = InMemoryChatHistory()

        if self.cfg.exp.make_embedding:
            self.retriever.build(self.docs, self.cfg)

    def _make_dataset(self):
        return instantiate(self.cfg.dataset)

    def _load_dataset(self):
        self.dataset.load_docs()
        return self.dataset.get_docs()

    def _make_embedder(self):
        # return instantiate(self.cfg.embedder)
        return None  # 임시 중단

    def _make_retriever(self):
        return instantiate(self.cfg.retriever)

    def _make_llm_client(self):
        return instantiate(self.cfg.llm)

    def _make_prompt(self):
        cp = self.cfg.prompt
        return QAPrompt(system_template=cp.system, human_template=cp.human)

    def _add_debug_chain(self):
        return RunnableLambda(debug_step)

    @property
    def qa_key(self):
        return self.cfg.prompt.qa_key

    @property
    def ref_key(self):
        return self.cfg.prompt.ref_key

    @property
    def hist_key(self):
        return self.cfg.prompt.history_key

    @property
    def session_id(self):
        return self.cfg.exp.sid

    @traceable(name="define-chain")
    def _define_chain(self):
        # (for debug) use RunnableLambda with debug_step fun ( | RunnableLambda(debug_step) )
        raise NotImplementedError()

    @traceable(name="run-single-turn")
    def run(self, question: str):
        chain = self._define_chain()
        config = {"configurable": {"session_id": self.session_id}}
        return chain.invoke({self.qa_key: question}, config=config)

    @traceable(name="run-multi-turn")
    def run_multi_turn(self, questions: List[str]):
        chain = self._define_chain()
        config = {"configurable": {"session_id": self.session_id}}

        responses = []
        for question in questions:
            resp = chain.invoke({self.qa_key: question}, config=config)
            responses.append(resp)

        return responses
