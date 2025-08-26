from langchain_core.prompts import ChatPromptTemplate

from rag.utils.logger import get_logger

logger = get_logger(__name__)


class QAPrompt:
    def __init__(self, system_template: str, human_template: str):
        self.system_template = system_template
        self.human_template = human_template

    def to_chain(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("human", self.human_template),
        ])
