from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from .path_utils import get_project_root

def load_documents(cfg):
    if cfg.data.path.endswith(".pdf"):
        # 상대 경로를 절대 경로로 변환 (Windows/Ubuntu 호환)
        file_path = Path(cfg.data.path)
        if not file_path.is_absolute():
            file_path = get_project_root() / cfg.data.path
        
        # 파일 존재 확인 및 디버그 정보 출력
        print(f"파일 경로: {file_path}")
        print(f"파일 존재: {file_path.exists()}")
        print(f"절대 경로: {file_path.resolve()}")
        
        if not file_path.exists():
            # 다른 가능한 경로들 확인
            alt_paths = [
                get_project_root() / "data" / "SPRI_AI_Brief_2023년12월호_F.pdf",
                Path("data") / "SPRI_AI_Brief_2023년12월호_F.pdf",
                Path("./data/SPRI_AI_Brief_2023년12월호_F.pdf")
            ]
            
            for alt_path in alt_paths:
                print(f"대안 경로 확인: {alt_path} -> {alt_path.exists()}")
                if alt_path.exists():
                    file_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {file_path}")
        
        # Windows/Ubuntu 모두 호환되는 경로 문자열로 변환
        file_path_str = str(file_path.resolve())
        
        loader = PyMuPDFLoader(file_path_str)
        return loader.load()
    else:
        raise ValueError(f"Unsupported document type for path: {cfg.data.path}")
