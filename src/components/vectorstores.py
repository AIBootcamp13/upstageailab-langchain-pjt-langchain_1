from langchain_community.vectorstores import FAISS, Chroma
from typing import List, Any
import logging

class VectorStoreManager:
    """RAG를 구축하기 위해 텍스트 데이터를 Split하고 Vector DB에 store하는 클래스"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.vectorstore = None
        self.logger = logging.getLogger(__name__)
    
    def create_vector_store(self, documents: List[Any], embeddings: Any):
        """벡터 스토어 생성"""
        try:
            if self.cfg.vector_store.type == "faiss":
                self.vectorstore = FAISS.from_documents(
                    documents=documents, 
                    embedding=embeddings
                )
                self.logger.info(f"FAISS vector store created with {len(documents)} documents")
                
            elif self.cfg.vector_store.type == "chromadb":
                self.vectorstore = Chroma.from_documents(
                    documents=documents, 
                    embedding=embeddings,
                    persist_directory=self.cfg.vector_store.persist_directory
                )
                self.logger.info(f"ChromaDB vector store created with {len(documents)} documents")
                
            else:
                raise ValueError(f"Unsupported vector store type: {self.cfg.vector_store.type}")
            
            return self.vectorstore
            
        except Exception as e:
            self.logger.error(f"Failed to create vector store: {e}")
            raise
    
    def save_vector_store(self, path: str = None):
        """벡터 스토어 저장"""
        try:
            if self.cfg.vector_store.type == "faiss" and self.vectorstore:
                save_path = path or self.cfg.vector_store.get('persist_directory', './faiss_db')
                self.vectorstore.save_local(save_path)
                self.logger.info(f"Vector store saved to {save_path}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to save vector store: {e}")
            return False
    
    def load_vector_store(self, embeddings: Any, path: str = None):
        """벡터 스토어 로드"""
        try:
            if self.cfg.vector_store.type == "faiss":
                load_path = path or self.cfg.vector_store.get('persist_directory', './faiss_db')
                self.vectorstore = FAISS.load_local(load_path, embeddings)
                self.logger.info(f"Vector store loaded from {load_path}")
                return self.vectorstore
        except Exception as e:
            self.logger.error(f"Failed to load vector store: {e}")
            return None
    
    def get_vector_store(self):
        """벡터 스토어 반환"""
        return self.vectorstore
    
    def get_stats(self) -> dict:
        """벡터 스토어 통계 정보"""
        if not self.vectorstore:
            return {"status": "not_initialized"}
        
        try:
            if hasattr(self.vectorstore, 'index'):
                # FAISS의 경우
                return {
                    "type": self.cfg.vector_store.type,
                    "status": "ready",
                    "total_vectors": self.vectorstore.index.ntotal if hasattr(self.vectorstore.index, 'ntotal') else "unknown"
                }
            else:
                return {
                    "type": self.cfg.vector_store.type,
                    "status": "ready"
                }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"status": "error", "error": str(e)}

def get_vector_store(cfg, documents, embeddings):
    """하위 호환성을 위한 함수 (기존 코드와의 호환성 유지)"""
    manager = VectorStoreManager(cfg)
    return manager.create_vector_store(documents, embeddings)
