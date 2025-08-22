import os
from datetime import datetime
import pytz

class LangSmithSimple:
    def __init__(self, cfg=None, version_manager=None):
        self.cfg = cfg
        self.version_manager = version_manager
        self.enabled = False
        
        if cfg and cfg.get('langsmith', {}).get('enabled', False):
            self.setup_langsmith()
    
    def setup_langsmith(self):
        """LangSmith 환경 변수 설정 (자동 추적)"""
        try:
            # API 키 확인
            if not os.getenv("LANGCHAIN_API_KEY"):
                if self.version_manager:
                    self.version_manager.logger.warning("LANGCHAIN_API_KEY가 설정되지 않았습니다. LangSmith 추적이 비활성화됩니다.")
                return
            
            # LangChain 자동 추적 활성화
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.cfg.langsmith.project_name
            
            # 선택적 환경 변수
            if hasattr(self.cfg.langsmith, 'session_name'):
                kst = pytz.timezone('Asia/Seoul')
                now = datetime.now(kst)
                session_id = f"{self.cfg.langsmith.session_name}_{now.strftime('%Y%m%d_%H%M%S')}"
                os.environ["LANGCHAIN_SESSION"] = session_id
            
            self.enabled = True
            
            if self.version_manager:
                self.version_manager.logger.info(f"LangSmith 자동 추적이 활성화되었습니다.")
                self.version_manager.logger.info(f"프로젝트: {self.cfg.langsmith.project_name}")
                if 'LANGCHAIN_SESSION' in os.environ:
                    self.version_manager.logger.info(f"세션: {os.environ['LANGCHAIN_SESSION']}")
            
        except Exception as e:
            if self.version_manager:
                self.version_manager.logger.error(f"LangSmith 설정 중 오류 발생: {e}")
            self.enabled = False
    
    def get_project_url(self):
        """프로젝트 대시보드 URL 반환"""
        if not self.enabled:
            return None
        
        try:
            project_name = self.cfg.langsmith.project_name.replace(" ", "%20")
            return f"https://smith.langchain.com/projects/p/{project_name}"
        except:
            return "https://smith.langchain.com/"
    
    def get_session_info(self):
        """세션 정보 반환"""
        if not self.enabled:
            return None
            
        return {
            "project": self.cfg.langsmith.project_name,
            "session": os.environ.get("LANGCHAIN_SESSION", "default"),
            "url": self.get_project_url()
        }