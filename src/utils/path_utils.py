import os
from pathlib import Path

def get_project_root():
    """프로젝트 루트 디렉토리 경로 반환"""
    return Path(__file__).parent.parent.parent

def get_data_path(filename):
    """데이터 파일 경로 반환 (OS 독립적)"""
    return get_project_root() / "data" / filename

def get_config_path():
    """설정 파일 경로 반환"""
    return get_project_root() / "conf"

def get_faiss_db_path():
    """FAISS DB 경로 반환"""
    return get_project_root() / "faiss_db"

def ensure_directory_exists(path):
    """디렉토리가 존재하지 않으면 생성"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path