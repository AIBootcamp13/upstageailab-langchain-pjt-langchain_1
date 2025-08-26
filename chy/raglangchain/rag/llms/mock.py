from typing import Optional, Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from rag.llms.base_llm import BaseLLM
from rag.utils.logger import get_logger

logger = get_logger(__name__)


class MockLLM(BaseLLM):

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        logger.info("=" * 80)
        logger.info("실제 모델이 전달받은 프롬프트")
        logger.info("=" * 80)
        logger.info(f"\n{messages}")
        logger.info("=" * 80 + "\n")

        # TODO: 사전 데이터 매핑 응담
        generated_message = BaseMessage(
            content="MOCK CHATBOT 입니다. 해당 데이터는 답변이 준비되지 않았습니다", type="ai"
        )
        generation = ChatGeneration(message=generated_message)
        return ChatResult(generations=[generation])
