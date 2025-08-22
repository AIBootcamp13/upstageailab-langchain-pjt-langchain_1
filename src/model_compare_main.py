import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
import os
from pathlib import Path

# 버전 관리 시스템 임포트
from src.utils.version_manager import VersionManager
from src.utils.langsmith_simple import LangSmithSimple
from src.utils.model_comparison import ModelComparison

from src.components.llms import get_llm
from src.components.embeddings import get_embedding_model
from src.components.vectorstores import get_vector_store
from src.components.retrievers import get_retriever
from src.utils.document_loaders import load_documents
from src.utils.text_splitters import split_documents
from src.prompts.qa_prompts import get_qa_prompt
from src.chains.qa_chain import get_qa_chain

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    load_dotenv()
    
    # 버전 관리 시스템 초기화
    version_manager = VersionManager()
    version_manager.logger.info("=== 모델 비교 실행 시작 ===")
    version_manager.logger.info(f"설정 정보:\n{OmegaConf.to_yaml(cfg)}")
    
    # LangSmith 자동 추적 시스템 초기화
    langsmith = LangSmithSimple(cfg, version_manager)
    
    # 모델 비교 시스템 초기화
    model_comparison = ModelComparison(version_manager, langsmith)
    
    print("🤖 GPT-4o vs Claude Haiku 3.5 모델 비교를 시작합니다...")
    
    try:
        # 공통 RAG 파이프라인 구성 요소 준비
        print("📚 문서 로딩 및 전처리...")
        documents = load_documents(cfg)
        split_documents_list = split_documents(cfg, documents)
        embeddings = get_embedding_model(cfg)
        vectorstore = get_vector_store(cfg, split_documents_list, embeddings)
        
        # Retriever 생성
        if cfg.retriever.type == "bm25" or cfg.retriever.type == "ensemble":
            retriever = get_retriever(cfg, vectorstore, documents=split_documents_list)
        else:
            retriever = get_retriever(cfg, vectorstore)
        
        prompt = get_qa_prompt()
        
        # 비교할 모델 설정
        model_configs = [
            {
                "name": "GPT-4o",
                "provider": "openai",
                "model_name": "gpt-4o",
                "temperature": 0.7
            },
            {
                "name": "Claude-3.5-Haiku",
                "provider": "anthropic", 
                "model_name": "claude-3-5-haiku-20241022",
                "temperature": 0.7
            }
        ]
        
        # QA 체인 팩토리 함수
        def create_qa_chain(model_config):
            # 모델별 설정으로 LLM 생성
            model_cfg = OmegaConf.create({
                "llm": {
                    "provider": model_config["provider"],
                    "model_name": model_config["model_name"],
                    "temperature": model_config["temperature"]
                }
            })
            llm = get_llm(model_cfg)
            return get_qa_chain(llm, retriever, prompt)
        
        # 테스트 질문들
        test_questions = [
            "미국 바이든 대통령이 몇년 몇월 몇일에 연방정부 차원에서 안전하고 신뢰할 수 있는 AI 개발과 사용을 보장하기 위한 행정명령을 발표했나요?",
            "AI 안전성 정상회의에 참가한 28개국들이 AI 안전 보장을 위한 협력 방안을 담은 블레츨리 선언을 발표한 나라는 어디인가요?",
            "구글이 앤스로픽에 투자한 금액은 총 얼마인가요?",
            "삼성전자가 자체 개발한 생성 AI 모델의 이름은 무엇인가요?",
            "갈릴레오의 LLM 환각 지수 평가에서 가장 우수한 성능을 보인 모델은 무엇인가요?"
        ]
        
        print(f"🔄 {len(test_questions)}개 질문으로 {len(model_configs)}개 모델 비교 중...")
        
        # 모델 비교 실행
        results = model_comparison.compare_models(
            questions=test_questions,
            model_configs=model_configs,
            qa_chain_factory=create_qa_chain
        )
        
        # 결과 요약 출력
        print("\n" + "="*80)
        print("🏆 모델 비교 결과 요약")
        print("="*80)
        
        for model_name, performance in results["model_performance"].items():
            print(f"\n📊 {model_name}")
            print(f"   성공률: {performance['success_rate']:.1%}")
            print(f"   평균 응답시간: {performance['average_response_time']:.2f}초")
            print(f"   평균 응답길이: {performance['average_response_length']:.0f}자")
            print(f"   평균 토큰수: {performance['average_tokens']:.0f}개")
        
        # LangSmith 정보
        if langsmith.enabled:
            session_info = langsmith.get_session_info()
            print(f"\n🔍 LangSmith 추적:")
            print(f"   프로젝트: {session_info['project']}")
            print(f"   대시보드: {session_info['url']}")
        
        version_manager.logger.info("=== 모델 비교 실행 완료 ===")
        
    except Exception as e:
        version_manager.logger.error(f"모델 비교 중 오류 발생: {e}")
        print(f"❌ 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()