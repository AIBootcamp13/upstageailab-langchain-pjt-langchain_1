import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

class CustomLogger:
    """실행되면서 기록되는 내용들을 저장하는 로거 클래스"""
    
    def __init__(self, 
                 name: str = "LegalQA", 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 console_output: bool = True):
        """
        로거 초기화
        Args:
            name: 로거 이름
            log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: 로그 파일 경로
            console_output: 콘솔 출력 여부
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.log_file = log_file
        self.session_logs = []
        
        # 로그 레벨 설정
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # 기존 핸들러 제거 (중복 방지)
        self.logger.handlers.clear()
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # 콘솔 핸들러 설정
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 파일 핸들러 설정
        if log_file:
            # 로그 디렉토리 생성
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logger '{name}' initialized with level {log_level}")
    
    def log_qa_interaction(self, question: str, answer: str, metadata: dict = None):
        """QA 상호작용 로깅"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "qa_interaction",
            "question": question,
            "answer": answer[:200] + "..." if len(answer) > 200 else answer,
            "metadata": metadata or {}
        }
        
        self.session_logs.append(log_entry)
        self.logger.info(f"QA Interaction - Q: {question[:50]}... | A: {answer[:50]}...")
    
    def log_system_event(self, event_type: str, details: dict = None):
        """시스템 이벤트 로깅"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "system_event",
            "event_type": event_type,
            "details": details or {}
        }
        
        self.session_logs.append(log_entry)
        self.logger.info(f"System Event - {event_type}: {details}")
    
    def log_error(self, error_type: str, error_message: str, traceback: str = None):
        """에러 로깅"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "error_type": error_type,
            "message": error_message,
            "traceback": traceback
        }
        
        self.session_logs.append(log_entry)
        self.logger.error(f"Error - {error_type}: {error_message}")
    
    def log_performance(self, operation: str, duration: float, details: dict = None):
        """성능 로깅"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "performance",
            "operation": operation,
            "duration_seconds": duration,
            "details": details or {}
        }
        
        self.session_logs.append(log_entry)
        self.logger.info(f"Performance - {operation}: {duration:.2f}s")
    
    def save_session_logs(self, file_path: str):
        """세션 로그를 JSON 파일로 저장"""
        try:
            session_data = {
                "logger_name": self.name,
                "session_start": datetime.now().isoformat(),
                "total_logs": len(self.session_logs),
                "logs": self.session_logs
            }
            
            # 로그 디렉토리 생성
            log_path = Path(file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Session logs saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save session logs: {e}")
            return False
    
    def get_session_stats(self) -> dict:
        """세션 통계 정보"""
        log_types = {}
        for log in self.session_logs:
            log_type = log.get("type", "unknown")
            log_types[log_type] = log_types.get(log_type, 0) + 1
        
        return {
            "total_logs": len(self.session_logs),
            "log_types": log_types,
            "session_duration": "active"
        }
    
    def clear_session_logs(self):
        """세션 로그 초기화"""
        self.session_logs.clear()
        self.logger.info("Session logs cleared")

def setup_project_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """프로젝트 전체 로깅 설정"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 메인 로거
    main_logger = CustomLogger(
        name="LegalQA_Main",
        log_level=log_level,
        log_file=str(log_path / f"main_{datetime.now().strftime('%Y%m%d')}.log")
    )
    
    # 각 모듈별 로거 설정
    module_loggers = {
        "llm": CustomLogger("LegalQA_LLM", log_level, str(log_path / "llm.log")),
        "vectorstore": CustomLogger("LegalQA_VectorStore", log_level, str(log_path / "vectorstore.log")),
        "chat_history": CustomLogger("LegalQA_ChatHistory", log_level, str(log_path / "chat_history.log")),
        "retriever": CustomLogger("LegalQA_Retriever", log_level, str(log_path / "retriever.log"))
    }
    
    return main_logger, module_loggers