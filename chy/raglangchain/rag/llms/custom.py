from typing import Optional, Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from rag.llms.base_llm import BaseLLM
from rag.utils.logger import get_logger

logger = get_logger(__name__)


class CustomLLM(BaseLLM):

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        generated_message = BaseMessage(content="아직 커스텀 LLM 구현이 완성되지 않났습니다", type="ai")
        generation = ChatGeneration(message=generated_message)
        return ChatResult(generations=[generation])
