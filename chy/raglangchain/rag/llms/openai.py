from langchain_openai import ChatOpenAI

from rag.llms.base_llm import BaseLLM
from rag.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAILLM(BaseLLM):

    def _create_client(self):
        return ChatOpenAI(model=self.model, temperature=self.temperature)
