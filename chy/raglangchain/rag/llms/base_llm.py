from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from rag.utils.logger import get_logger

logger = get_logger(__name__)


class BaseLLM(BaseChatModel):
    model: str
    max_tokens: int = 1024
    temperature: float = 0.8
    client: Any = None

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.client = self._create_client()

    def _create_client(self):
        return self  # 커스텀의 경우 자식 인스턴스 활용

    def chat_with(self):
        return self.client

    """ BaseChatModel Abstract """

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        raise NotImplementedError()  # 로컬 혹은 자체 커스텀 확장 LLM의 경우 상속 받아 구현

    @property
    def _llm_type(self) -> str:
        return self.__class__.__name__
