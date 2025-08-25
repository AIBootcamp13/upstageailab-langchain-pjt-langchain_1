import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
import os
import time
from typing import Dict, Any, Optional

# 모든 모듈들을 클래스로 import
from .components.llms import LLMManager
from .components.embeddings import get_embedding_model
from .components.vectorstores import VectorStoreManager
from .components.retrievers import get_retriever
from .components.chat_history import ChatHistoryManager
from .components.logger import setup_project_logging
from .utils.document_loaders import load_documents
from .utils.text_splitters import split_documents
from .prompts.qa_prompts import get_qa_prompt
from .chains.qa_chain import get_qa_chain

class LegalQAEngine:
    """법률 QA 엔진 메인 클래스 - 모든 모듈을 통합하여 관리"""
    
    def __init__(self, cfg: DictConfig):
        """
        법률 QA 엔진 초기화
        Args:
            cfg: Hydra 설정
        """
        self.cfg = cfg
        self.logger = None
        self.llm_manager = None
        self.vectorstore_manager = None
        self.chat_history = None
        self.qa_chain = None
        self.is_initialized = False
        
        # 초기화
        self._setup_environment()
        self._initialize_components()
    
    def _setup_environment(self):
        """환경 설정"""
        load_dotenv()
        
        # 로깅 설정
        self.logger, _ = setup_project_logging(
            log_level=self.cfg.get('logging', {}).get('level', 'INFO'),
            log_dir=self.cfg.get('logging', {}).get('directory', 'logs')
        )
        
        # LangSmith 설정
        if self.cfg.get('langsmith', {}).get('enabled', False):
            if os.getenv('LANGCHAIN_API_KEY'):
                os.environ['LANGCHAIN_TRACING_V2'] = 'true'
                os.environ['LANGCHAIN_PROJECT'] = self.cfg.langsmith.project_name
                os.environ['LANGCHAIN_ENDPOINT'] = self.cfg.langsmith.endpoint
                self.logger.logger.info(f"LangSmith 추적 활성화: {self.cfg.langsmith.project_name}")
            else:
                self.logger.logger.warning("LANGCHAIN_API_KEY가 설정되지 않았습니다. LangSmith 추적 비활성화")
    
    def _initialize_components(self):
        """모든 구성요소 초기화"""
        try:
            # 1. LLM 관리자 초기화
            self.logger.log_system_event("llm_initialization_start")
            self.llm_manager = LLMManager(self.cfg)
            
            # 2. 문서 처리
            self.logger.log_system_event("document_loading_start")
            documents = load_documents(self.cfg)
            self.logger.logger.info(f"문서 로드 완료: {len(documents)}개")
            
            split_docs = split_documents(self.cfg, documents)
            self.logger.logger.info(f"문서 분할 완료: {len(split_docs)}개 청크")
            
            # 3. 임베딩 및 벡터스토어
            self.logger.log_system_event("vectorstore_initialization_start")
            embeddings = get_embedding_model(self.cfg)
            self.vectorstore_manager = VectorStoreManager(self.cfg)
            vectorstore = self.vectorstore_manager.create_vector_store(split_docs, embeddings)
            
            # 4. 리트리버 설정
            if self.cfg.retriever.type in ["bm25", "ensemble"]:
                retriever = get_retriever(self.cfg, vectorstore, documents=split_docs)
            else:
                retriever = get_retriever(self.cfg, vectorstore)
            
            # 5. QA 체인 생성
            llm = self.llm_manager.get_llm()
            prompt = get_qa_prompt()
            self.qa_chain = get_qa_chain(llm, retriever, prompt)
            
            # 6. 채팅 히스토리 초기화 (QA Engine: 5번 이상 기억)
            history_config = self.cfg.get('chat_history', {})
            max_history = history_config.get('max_history', 5)
            save_path = history_config.get('save_path', 'logs/chat_history.json')
            
            self.chat_history = ChatHistoryManager(max_history=max_history, save_path=save_path)
            
            
            self.is_initialized = True
            self.logger.log_system_event("qa_engine_initialization_complete", {
                "llm_provider": self.cfg.llm.provider,
                "vectorstore_type": self.cfg.vector_store.type,
                "retriever_type": self.cfg.retriever.type,
                "max_chat_history": max_history
            })
            
        except Exception as e:
            self.logger.log_error("initialization_error", str(e))
            raise
    
    def ask(self, question: str, use_chat_history: bool = True) -> Dict[str, Any]:
        """
        질문 처리 및 답변 생성
        Args:
            question: 사용자 질문
            use_chat_history: 채팅 히스토리 사용 여부
        """
        if not self.is_initialized:
            return {"error": "QA Engine이 초기화되지 않았습니다."}
        
        try:
            start_time = time.time()
            
            # 채팅 히스토리에 질문 추가
            if use_chat_history:
                self.chat_history.add_user_message(question)
                
                # 이전 대화 맥락 포함
                context = self.chat_history.format_for_prompt()
                if context:
                    enhanced_question = f"이전 대화:\n{context}\n\n현재 질문: {question}"
                else:
                    enhanced_question = question
            else:
                enhanced_question = question
            
            # QA 체인 실행
            response = self.qa_chain.invoke(enhanced_question)
            response_time = time.time() - start_time
            
            # 응답 검증
            is_valid = self.llm_manager.validate_response(response)
            answer = response.get("result", response.get("answer", str(response))) if isinstance(response, dict) else str(response)
            
            # 채팅 히스토리에 응답 추가
            if use_chat_history and is_valid:
                self.chat_history.add_assistant_message(answer, {
                    "response_time": response_time,
                    "model": f"{self.cfg.llm.provider}/{self.cfg.llm.model_name}"
                })
            
            # 로깅
            self.logger.log_qa_interaction(question, answer, {
                "response_time": response_time,
                "is_valid": is_valid,
                "model": f"{self.cfg.llm.provider}/{self.cfg.llm.model_name}"
            })
            
            return {
                "question": question,
                "answer": answer,
                "response_time": response_time,
                "success": is_valid,
                "chat_history_used": use_chat_history,
                "model": f"{self.cfg.llm.provider}/{self.cfg.llm.model_name}"
            }
            
        except Exception as e:
            error_msg = str(e)
            self.logger.log_error("qa_processing_error", error_msg)
            return {
                "question": question,
                "error": error_msg,
                "success": False
            }
    
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 전체 상태 확인"""
        status = {
            "initialized": self.is_initialized,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if self.is_initialized:
            # LLM 상태
            llm_status = self.llm_manager.test_connection()
            status["llm"] = llm_status
            
            # 벡터스토어 상태
            status["vectorstore"] = self.vectorstore_manager.get_stats()
            
            # 채팅 히스토리 상태
            status["chat_history"] = self.chat_history.get_stats()
            
            # 로거 상태
            status["logging"] = self.logger.get_session_stats()
        
        return status
    
    def run_interactive_session(self):
        """대화형 세션 실행"""
        if not self.is_initialized:
            print("QA Engine이 초기화되지 않았습니다.")
            return
        
        print("법률 QA 시스템이 시작되었습니다.")
        print("'exit', 'quit', 'q'를 입력하면 종료됩니다.")
        print("'status'를 입력하면 시스템 상태를 확인할 수 있습니다.\n")
        
        while True:
            try:
                question = input("\n질문을 입력하세요: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("법률 QA 시스템을 종료합니다.")
                    break
                
                elif question.lower() == 'status':
                    status = self.get_system_status()
                    print("시스템 상태:")
                    for key, value in status.items():
                        print(f"  - {key}: {value}")
                    continue
                
                
                if not question:
                    continue
                
                print("답변을 생성 중입니다...")
                result = self.ask(question)
                
                if result.get("success"):
                    print(f"\n답변: {result['answer']}")
                    print(f"응답 시간: {result['response_time']:.2f}초")
                else:
                    print(f"오류: {result.get('error', '알 수 없는 오류')}")
                
            except KeyboardInterrupt:
                print("\n법률 QA 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"예상치 못한 오류: {e}")

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """메인 실행 함수 - 모든 모듈을 통합하여 실행"""
    
    print("법률 QA 엔진을 초기화하는 중...")
    print(OmegaConf.to_yaml(cfg))
    
    try:
        # QA 엔진 초기화
        qa_engine = LegalQAEngine(cfg)
        
        # 시스템 상태 확인
        status = qa_engine.get_system_status()
        print(f"시스템 초기화 완료")
        
        # 대화형 모드 시작 (선택사항)
        interactive_mode = cfg.get('interactive', False)
        if interactive_mode:
            qa_engine.run_interactive_session()
            
    except Exception as e:
        print(f"초기화 실패: {e}")
        raise

if __name__ == "__main__":
    main()
