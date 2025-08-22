from langchain_community.vectorstores import FAISS, Chroma
from pathlib import Path
from ..utils.path_utils import get_project_root, ensure_directory_exists

def get_vector_store(cfg, documents, embeddings):
    if cfg.vector_store.type == "faiss":
        return FAISS.from_documents(documents=documents, embedding=embeddings)
    elif cfg.vector_store.type == "chromadb":
        # 상대 경로를 절대 경로로 변환 (Windows/Ubuntu 호환)
        persist_dir = Path(cfg.vector_store.persist_directory)
        if not persist_dir.is_absolute():
            persist_dir = get_project_root() / cfg.vector_store.persist_directory
        
        # 디렉토리 존재 확인 및 생성
        ensure_directory_exists(persist_dir)
        
        # Windows/Ubuntu 모두 호환되는 경로 문자열로 변환
        persist_dir_str = str(persist_dir).replace('\\', '/')
        
        return Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_dir_str)
    else:
        raise ValueError(f"Unsupported vector store type: {cfg.vector_store.type}")
