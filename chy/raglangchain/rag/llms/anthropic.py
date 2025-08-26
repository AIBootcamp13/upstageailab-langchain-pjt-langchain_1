from langchain_anthropic import ChatAnthropic

from rag.llms.base_llm import BaseLLM
from rag.utils.logger import get_logger

logger = get_logger(__name__)


class AnthropicLLM(BaseLLM):

    def _create_client(self):
        return ChatAnthropic(model=self.model, temperature=self.temperature)
