import os
import shutil
from datetime import datetime
import pytz
import logging
from pathlib import Path

class VersionManager:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        """로그 시스템 설정"""
        log_file = self.log_dir / "version_history.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_korean_timestamp(self):
        """한국시각 기반 타임스탬프 생성 (_MMDDHHMM 형태)"""
        kst = pytz.timezone('Asia/Seoul')
        now = datetime.now(kst)
        return now.strftime("_%m%d%H%M")
    
    def create_versioned_file(self, original_path, description="", langsmith_manager=None):
        """파일 버전 생성 및 로그 기록"""
        original_path = Path(original_path)
        
        if not original_path.exists():
            raise FileNotFoundError(f"원본 파일을 찾을 수 없습니다: {original_path}")
        
        # 버전 파일명 생성
        timestamp = self.get_korean_timestamp()
        name_parts = original_path.name.rsplit('.', 1)
        if len(name_parts) == 2:
            versioned_name = f"{name_parts[0]}{timestamp}.{name_parts[1]}"
        else:
            versioned_name = f"{original_path.name}{timestamp}"
        
        versioned_path = original_path.parent / versioned_name
        
        # 파일 복사
        shutil.copy2(original_path, versioned_path)
        
        # 로그 기록
        self.logger.info(f"버전 생성: {original_path} -> {versioned_path}")
        if description:
            self.logger.info(f"변경 사항: {description}")
        
        # LangSmith에도 버전 정보 기록 (선택사항)
        if langsmith_manager and langsmith_manager.enabled:
            try:
                version_run_id = langsmith_manager.start_run(
                    name="File_Version_Created",
                    run_type="tool",
                    inputs={
                        "original_file": str(original_path),
                        "versioned_file": str(versioned_path),
                        "timestamp": timestamp,
                        "description": description
                    },
                    tags=["version_management", "file_backup"]
                )
                langsmith_manager.end_run(version_run_id, outputs={"success": True})
            except Exception as e:
                self.logger.warning(f"LangSmith 버전 기록 중 오류: {e}")
        
        return versioned_path
    
    def backup_directory(self, dir_path, description=""):
        """디렉토리 전체 백업"""
        dir_path = Path(dir_path)
        timestamp = self.get_korean_timestamp()
        backup_name = f"{dir_path.name}_backup{timestamp}"
        backup_path = dir_path.parent / backup_name
        
        shutil.copytree(dir_path, backup_path)
        
        self.logger.info(f"디렉토리 백업: {dir_path} -> {backup_path}")
        if description:
            self.logger.info(f"백업 사유: {description}")
        
        return backup_path
    
    def fix_windows_paths(self, file_path):
        """Windows/Ubuntu 호환 경로 수정"""
        file_path = Path(file_path)
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 백업 생성
            versioned_path = self.create_versioned_file(file_path, "Windows/Ubuntu 경로 호환성 수정")
            
            # 경로 구분자 수정
            content = content.replace('\\', '/')
            # 절대 경로를 상대 경로로 변경
            content = content.replace('../data/', 'data/')
            content = content.replace('./faiss_db', 'faiss_db')
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Windows/Ubuntu 호환 경로로 수정 완료: {file_path}")
            return True
        
        return False