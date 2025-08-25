from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Any
import logging

class LLMManager:
    """LLM API를 호출하고 response를 받는 클래스"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.llm = None
        self.logger = logging.getLogger(__name__)
        self._initialize_llm()
    
    def _initialize_llm(self):
        """LLM 초기화"""
        try:
            if self.cfg.llm.provider == "openai":
                self.llm = ChatOpenAI(
                    model_name=self.cfg.llm.model_name, 
                    temperature=self.cfg.llm.temperature
                )
            elif self.cfg.llm.provider == "google":
                self.llm = ChatGoogleGenerativeAI(
                    model=self.cfg.llm.model_name, 
                    temperature=self.cfg.llm.temperature
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.cfg.llm.provider}")
            
            self.logger.info(f"LLM initialized: {self.cfg.llm.provider}/{self.cfg.llm.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def validate_response(self, response: Any) -> bool:
        """API 응답이 제대로 생성되었는지 확인"""
        try:
            if response is None:
                return False
            
            # 응답 타입에 따른 검증
            if hasattr(response, 'content'):
                return bool(response.content and len(response.content.strip()) > 0)
            elif isinstance(response, str):
                return bool(response and len(response.strip()) > 0)
            elif isinstance(response, dict):
                return bool(response.get('content') or response.get('answer'))
            
            return bool(response)
        except Exception as e:
            self.logger.error(f"Response validation failed: {e}")
            return False
    
    def get_llm(self):
        """LLM 인스턴스 반환"""
        return self.llm
    
    def test_connection(self) -> Dict[str, Any]:
        """LLM 연결 테스트"""
        try:
            test_response = self.llm.invoke("Hello")
            is_valid = self.validate_response(test_response)
            
            return {
                "status": "success" if is_valid else "failed",
                "provider": self.cfg.llm.provider,
                "model": self.cfg.llm.model_name,
                "response_valid": is_valid,
                "test_response": str(test_response)[:100] if test_response else None
            }
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return {
                "status": "error",
                "provider": self.cfg.llm.provider,
                "model": self.cfg.llm.model_name,
                "error": str(e)
            }

def get_llm(cfg):
    """하위 호환성을 위한 함수 (기존 코드와의 호환성 유지)"""
    llm_manager = LLMManager(cfg)
    return llm_manager.get_llm()
