from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from rag.utils.logger import get_logger

logger = get_logger(__name__)

"""
TODO: 
- 파일 fallback 처리 (장기 처리)
  - shutdown hook에 기존 이력 저장
- FileChatMessageHistory(f"chat_{session_id}.txt")
- LRU 적용
- 최대 대화 보관수 설정
"""


class InMemoryChatHistory(BaseChatMessageHistory):
    _histories = {}

    def clear(self) -> None:
        self._histories = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        logger.info(f"채팅 이력 조회 - 세션: [{session_id}]")
        if session_id not in self._histories:
            self._histories[session_id] = InMemoryChatMessageHistory()
        return self._histories.get(session_id)
