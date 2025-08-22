import os
import uuid
from datetime import datetime
import pytz
from langsmith import Client
import logging

class LangSmithManager:
    def __init__(self, cfg=None, version_manager=None):
        self.cfg = cfg
        self.version_manager = version_manager
        self.client = None
        self.session_id = None
        self.enabled = False
        
        if cfg and cfg.get('langsmith', {}).get('enabled', False):
            self.setup_langsmith()
    
    def setup_langsmith(self):
        """LangSmith 클라이언트 설정"""
        try:
            # 환경 변수 확인
            if not os.getenv("LANGCHAIN_API_KEY"):
                if self.version_manager:
                    self.version_manager.logger.warning("LANGCHAIN_API_KEY가 설정되지 않았습니다. LangSmith 추적이 비활성화됩니다.")
                return
            
            # LangSmith 환경 변수 설정
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.cfg.langsmith.project_name
            
            # LangSmith 클라이언트 초기화
            self.client = Client(
                api_key=os.getenv("LANGCHAIN_API_KEY"),
                api_url="https://api.smith.langchain.com"
            )
            
            # 세션 ID 생성 (한국시각 기반)
            kst = pytz.timezone('Asia/Seoul')
            now = datetime.now(kst)
            self.session_id = f"{self.cfg.langsmith.session_name}_{now.strftime('%Y%m%d_%H%M%S')}"
            
            self.enabled = True
            
            if self.version_manager:
                self.version_manager.logger.info(f"LangSmith 추적이 활성화되었습니다. 프로젝트: {self.cfg.langsmith.project_name}")
                self.version_manager.logger.info(f"세션 ID: {self.session_id}")
            
        except Exception as e:
            if self.version_manager:
                self.version_manager.logger.error(f"LangSmith 설정 중 오류 발생: {e}")
            self.enabled = False
    
    def start_run(self, name, run_type="chain", inputs=None, tags=None):
        """새로운 실행 시작"""
        if not self.enabled:
            return None
            
        try:
            all_tags = list(self.cfg.langsmith.tags) if self.cfg.langsmith.tags else []
            if tags:
                all_tags.extend(tags)
            
            run = self.client.create_run(
                name=name,
                run_type=run_type,
                inputs=inputs or {},
                session_name=self.session_id,
                tags=all_tags
            )
            return run.id
        except Exception as e:
            if self.version_manager:
                self.version_manager.logger.error(f"LangSmith 실행 시작 중 오류: {e}")
            return None
    
    def end_run(self, run_id, outputs=None, error=None):
        """실행 종료"""
        if not self.enabled or not run_id:
            return
            
        try:
            self.client.update_run(
                run_id=run_id,
                outputs=outputs or {},
                error=str(error) if error else None,
                end_time=datetime.utcnow()
            )
        except Exception as e:
            if self.version_manager:
                self.version_manager.logger.error(f"LangSmith 실행 종료 중 오류: {e}")
    
    def log_feedback(self, run_id, score, comment=None):
        """피드백 로깅"""
        if not self.enabled or not run_id:
            return
            
        try:
            self.client.create_feedback(
                run_id=run_id,
                key="user_score",
                score=score,
                comment=comment
            )
        except Exception as e:
            if self.version_manager:
                self.version_manager.logger.error(f"LangSmith 피드백 로깅 중 오류: {e}")
    
    def get_session_url(self):
        """세션 URL 반환"""
        if not self.enabled:
            return None
        
        try:
            # LangSmith 대시보드 URL 생성
            project_name = self.cfg.langsmith.project_name.replace(" ", "%20")
            return f"https://smith.langchain.com/projects/p/{project_name}"
        except:
            return None