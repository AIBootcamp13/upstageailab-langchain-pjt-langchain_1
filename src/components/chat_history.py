from typing import List, Dict, Any, Optional
from collections import deque
import logging
import json
from datetime import datetime

class ChatHistoryManager:
    """Chat History를 기억하여 지정한 페르소나, 주어진 역할, 답변의 출력 양식등을 유지하는 클래스"""
    
    def __init__(self, max_history: int = 5, save_path: Optional[str] = None):
        """
        초기화
        Args:
            max_history: QA Engine에서는 5번 이상 기억 (기본값 5)
            save_path: 대화 기록 저장 경로 (선택사항)
        """
        self.max_history = max_history
        self.save_path = save_path
        self.history = deque(maxlen=max_history * 2)  # Q&A 쌍이므로 *2
        self.session_id = None
        self.logger = logging.getLogger(__name__)
        self._initialize_session()
    
    def _initialize_session(self):
        """세션 초기화"""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"Chat history session initialized: {self.session_id}")
    
    def add_user_message(self, message: str, metadata: Optional[Dict] = None):
        """사용자 메시지 추가"""
        entry = {
            "type": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.history.append(entry)
        self.logger.debug(f"User message added: {message[:50]}...")
    
    def add_assistant_message(self, message: str, metadata: Optional[Dict] = None):
        """어시스턴트 응답 추가"""
        entry = {
            "type": "assistant", 
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.history.append(entry)
        self.logger.debug(f"Assistant message added: {message[:50]}...")
        
        # 자동 저장 (경로가 설정된 경우)
        if self.save_path:
            self.save_history()
    
    def get_conversation_context(self, include_system_prompt: bool = True) -> List[Dict]:
        """대화 맥락을 Langchain 형식으로 반환"""
        context = []
        
        # 시스템 프롬프트 추가 (필요한 경우)
        if include_system_prompt:
            context.append({
                "role": "system",
                "content": "당신은 법률 전문가입니다. 이전 대화 맥락을 고려하여 일관된 답변을 제공하세요."
            })
        
        # 대화 기록 추가
        for entry in self.history:
            role = "user" if entry["type"] == "user" else "assistant"
            context.append({
                "role": role,
                "content": entry["content"]
            })
        
        return context
    
    def get_recent_qa_pairs(self, num_pairs: int = 2) -> List[Dict]:
        """최근 Q&A 쌍 반환"""
        pairs = []
        temp_history = list(self.history)
        
        # 뒤에서부터 Q&A 쌍 찾기
        i = len(temp_history) - 1
        pair_count = 0
        
        while i >= 1 and pair_count < num_pairs:
            if (temp_history[i]["type"] == "assistant" and 
                temp_history[i-1]["type"] == "user"):
                pairs.insert(0, {
                    "question": temp_history[i-1]["content"],
                    "answer": temp_history[i]["content"],
                    "timestamp": temp_history[i]["timestamp"]
                })
                pair_count += 1
                i -= 2
            else:
                i -= 1
        
        return pairs
    
    def clear_history(self):
        """대화 기록 초기화"""
        self.history.clear()
        self.logger.info("Chat history cleared")
    
    def save_history(self, custom_path: Optional[str] = None):
        """대화 기록 파일로 저장"""
        try:
            path = custom_path or self.save_path
            if not path:
                return False
            
            history_data = {
                "session_id": self.session_id,
                "max_history": self.max_history,
                "created_at": datetime.now().isoformat(),
                "history": list(self.history)
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Chat history saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save chat history: {e}")
            return False
    
    def load_history(self, file_path: str):
        """파일에서 대화 기록 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.session_id = data.get("session_id")
            self.history = deque(data.get("history", []), maxlen=self.max_history * 2)
            
            self.logger.info(f"Chat history loaded from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load chat history: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """대화 기록 통계 정보"""
        user_messages = sum(1 for entry in self.history if entry["type"] == "user")
        assistant_messages = sum(1 for entry in self.history if entry["type"] == "assistant")
        
        return {
            "session_id": self.session_id,
            "total_messages": len(self.history),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "qa_pairs": min(user_messages, assistant_messages),
            "max_history": self.max_history,
            "memory_usage": f"{len(self.history)}/{self.max_history * 2}"
        }
    
    def format_for_prompt(self, separator: str = "\n---\n") -> str:
        """프롬프트에 포함할 수 있도록 대화 기록을 문자열로 포맷"""
        if not self.history:
            return ""
        
        formatted_history = []
        for entry in self.history:
            role = "사용자" if entry["type"] == "user" else "AI 법률 전문가"
            formatted_history.append(f"{role}: {entry['content']}")
        
        return separator.join(formatted_history)