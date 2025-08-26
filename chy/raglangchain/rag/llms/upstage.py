from langchain_openai import ChatOpenAI

from rag.llms.openai import OpenAILLM
from rag.utils.logger import get_logger

logger = get_logger(__name__)


class UpstageLLM(OpenAILLM):
    api_key: str
    endpoint: str

    def _create_client(self):
        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            openai_api_key=self.api_key,
            openai_api_base=self.endpoint,
        )
