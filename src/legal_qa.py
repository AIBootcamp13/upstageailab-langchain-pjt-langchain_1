"""
법률 QA 챗봇 - 터미널/주피터에서 사용 가능한 인터페이스

사용법:
    from src.legal_qa import LegalQA
    
    # 기본 설정으로 초기화
    qa = LegalQA()
    
    # 특정 실험 설정으로 초기화  
    qa = LegalQA(config_name="gemini_exp")
    
    # 질문하기
    response = qa.ask("전세사기 관련 판례를 알려주세요")
    
    # 다양한 실험 비교
    qa.compare_models(["gemini_exp", "huggingface_exp"])
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import hydra
from omegaconf import OmegaConf
from dotenv import load_dotenv
import pandas as pd

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.components.llms import get_llm
from src.components.embeddings import get_embedding_model
from src.components.vectorstores import get_vector_store
from src.components.retrievers import get_retriever
from src.utils.document_loaders import load_documents
from src.utils.text_splitters import split_documents
from src.prompts.qa_prompts import get_qa_prompt
from src.chains.qa_chain import get_qa_chain


class LegalQA:
    """법률 QA 챗봇 클래스"""
    
    def __init__(self, config_name: str = "config", config_path: Optional[str] = None):
        """
        초기화
        
        Args:
            config_name: 사용할 설정 파일명 (확장자 제외)
            config_path: 설정 파일 경로 (기본: conf)
        """
        self.config_name = config_name
        # Hydra는 상대 경로를 요구함
        self.config_path = config_path or "conf"
        self.cfg = None
        self.qa_chain = None
        self.is_initialized = False
        
        # .env 로드
        load_dotenv()
        
        # 현재 작업 디렉토리를 프로젝트 루트로 변경
        self.original_cwd = os.getcwd()
        
        # 초기화
        self._initialize()
    
    def _initialize(self):
        """시스템 초기화"""
        try:
            # Hydra 초기화 상태 확인 및 정리 (주피터 환경에서 재실행 문제 해결)
            if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
                hydra.core.global_hydra.GlobalHydra.instance().clear()
            
            # 무조건 프로젝트 루트로 변경
            os.chdir(project_root)
            
            # conf 디렉토리 존재 확인
            conf_path = Path("conf")
            if not conf_path.exists():
                raise FileNotFoundError(f"conf 디렉토리를 찾을 수 없습니다: {conf_path.absolute()}")
            
            # Hydra 설정 로드 (initialize_config_dir 사용)
            config_dir = project_root / "conf"
            
            with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
                self.cfg = hydra.compose(config_name=self.config_name)
            
            # LangSmith 설정
            self._setup_langsmith()
            
            # QA 체인 구성
            self._setup_qa_chain()
            
            self.is_initialized = True
            print(f"LegalQA 시스템이 초기화되었습니다. (설정: {self.config_name})")
            
        except Exception as e:
            print(f"초기화 실패: {e}")
            self.is_initialized = False
    
    def _setup_langsmith(self):
        """LangSmith 설정"""
        if self.cfg.get('langsmith', {}).get('enabled', False):
            if os.getenv('LANGCHAIN_API_KEY'):
                os.environ['LANGCHAIN_TRACING_V2'] = 'true'
                os.environ['LANGCHAIN_PROJECT'] = self.cfg.langsmith.project_name
                os.environ['LANGCHAIN_ENDPOINT'] = self.cfg.langsmith.endpoint
                print(f"LangSmith 추적 활성화: {self.cfg.langsmith.project_name}")
    
    def _setup_qa_chain(self):
        """QA 체인 구성"""
        documents = load_documents(self.cfg)
        print(f"문서 수: {len(documents)}")
        
        split_docs = split_documents(self.cfg, documents)
        print(f"청크 수: {len(split_docs)}")
        
        embeddings = get_embedding_model(self.cfg)
        vectorstore = get_vector_store(self.cfg, split_docs, embeddings)
        
        if self.cfg.retriever.type in ["bm25", "ensemble"]:
            retriever = get_retriever(self.cfg, vectorstore, documents=split_docs)
        else:
            retriever = get_retriever(self.cfg, vectorstore)
        
        llm = get_llm(self.cfg)
        prompt = get_qa_prompt()
        self.qa_chain = get_qa_chain(llm, retriever, prompt)
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        질문하기
        
        Args:
            question: 질문 내용
            
        Returns:
            Dict containing answer and metadata
        """
        if not self.is_initialized:
            return {"error": "시스템이 초기화되지 않았습니다."}
        
        try:
            response = self.qa_chain.invoke(question)
            # 응답 형식 처리 (dict 또는 str 모두 처리)
            if isinstance(response, dict):
                answer = response.get("result", response.get("answer", str(response)))
            else:
                answer = str(response)
            
            return {
                "question": question,
                "answer": answer,
                "config": self.config_name,
                "model": f"{self.cfg.llm.provider}/{self.cfg.llm.model_name}"
            }
        except Exception as e:
            return {
                "question": question,
                "error": str(e),
                "config": self.config_name
            }
    
    def get_config(self) -> Dict:
        """현재 설정 반환"""
        if self.cfg:
            return OmegaConf.to_container(self.cfg, resolve=True)
        return {}
    
    def get_example_questions(self) -> List[str]:
        """예시 질문들 반환"""
        if self.cfg and 'query' in self.cfg:
            return self.cfg.query.get('examples', [])
        return []
    
    @staticmethod
    def list_available_configs() -> List[str]:
        """사용 가능한 설정들 나열"""
        # 현재 디렉토리를 임시로 변경하여 경로 문제 해결
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        try:
            conf_path = Path("conf")
            configs = []
            
            # 기본 config.yaml
            if (conf_path / "config.yaml").exists():
                configs.append("config")
            
            # experiment 폴더의 설정들
            exp_path = conf_path / "experiment"
            if exp_path.exists():
                for f in exp_path.glob("*.yaml"):
                    configs.append(f.stem)
            
            return configs
        finally:
            os.chdir(original_cwd)
    
    def compare_models(self, config_names: List[str], question: str) -> pd.DataFrame:
        """
        여러 모델로 같은 질문을 테스트하고 비교
        
        Args:
            config_names: 비교할 설정들
            question: 테스트 질문
            
        Returns:
            비교 결과 DataFrame
        """
        results = []
        
        for config_name in config_names:
            print(f"{config_name} 모델로 테스트 중...")
            
            # 새로운 인스턴스 생성
            qa = LegalQA(config_name=config_name)
            result = qa.ask(question)
            
            results.append({
                "config": config_name,
                "model": result.get("model", "unknown"),
                "question": question,
                "answer": result.get("answer", result.get("error", "No response")),
                "has_error": "error" in result
            })
        
        return pd.DataFrame(results)


# 편의 함수들
def quick_ask(question: str, config: str = "config") -> str:
    """빠른 질문하기"""
    # 현재 디렉토리를 프로젝트 루트로 변경
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        qa = LegalQA(config_name=config)
        result = qa.ask(question)
        return result.get("answer", result.get("error", "No response"))
    finally:
        os.chdir(original_cwd)


def show_examples():
    """예시 질문들 출력"""
    # 현재 디렉토리를 프로젝트 루트로 변경
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        qa = LegalQA()
        examples = qa.get_example_questions()
        print("예시 질문들:")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    # 테스트 실행
    print("LegalQA 테스트")
    
    # 사용 가능한 설정들 출력
    configs = LegalQA.list_available_configs()
    print(f"사용 가능한 설정들: {configs}")
    
    # 기본 설정으로 질문
    qa = LegalQA()
    if qa.is_initialized:
        test_question = "전세사기 관련 판례를 알려주세요"
        result = qa.ask(test_question)
        print(f"질문: {test_question}")
        print(f"답변: {result.get('answer', result.get('error'))}")