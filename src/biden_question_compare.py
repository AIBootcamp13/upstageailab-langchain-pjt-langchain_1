import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
import os
import time

# 버전 관리 시스템 임포트
from src.utils.version_manager import VersionManager
from src.utils.langsmith_simple import LangSmithSimple

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
    version_manager.logger.info("=== 바이든 질문 모델 비교 시작 ===")
    
    # LangSmith 자동 추적 시스템 초기화
    langsmith = LangSmithSimple(cfg, version_manager)
    
    print("🔍 바이든 행정명령 질문으로 GPT-4o vs Claude-3.5-Haiku 상세 비교")
    print("="*80)
    
    try:
        # 공통 RAG 파이프라인 구성 요소 준비
        print("📚 RAG 파이프라인 준비 중...")
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
        
        # 테스트 질문
        question = "미국 바이든 대통령이 몇년 몇월 몇일에 연방정부 차원에서 안전하고 신뢰할 수 있는 AI 개발과 사용을 보장하기 위한 행정명령을 발표했나요?"
        
        print(f"\n❓ 질문: {question}")
        print("\n" + "="*80)
        
        # 모델별 테스트
        models = [
            {"name": "GPT-4o", "provider": "openai", "model_name": "gpt-4o", "temperature": 0.7},
            {"name": "Claude-3.5-Haiku", "provider": "anthropic", "model_name": "claude-3-5-haiku-20241022", "temperature": 0.7}
        ]
        
        results = []
        
        for i, model_config in enumerate(models, 1):
            print(f"\n🤖 [{i}/2] {model_config['name']} 테스트 중...")
            
            # 모델별 설정으로 LLM 생성
            model_cfg = OmegaConf.create({
                "llm": {
                    "provider": model_config["provider"],
                    "model_name": model_config["model_name"],
                    "temperature": model_config["temperature"]
                }
            })
            
            # 실행 시간 측정
            start_time = time.time()
            
            try:
                llm = get_llm(model_cfg)
                qa_chain = get_qa_chain(llm, retriever, prompt)
                response = qa_chain.invoke(question)
                execution_time = time.time() - start_time
                
                result = {
                    "model": model_config['name'],
                    "response": response,
                    "execution_time": execution_time,
                    "response_length": len(str(response)),
                    "word_count": len(str(response).split()),
                    "success": True,
                    "error": None
                }
                
                print(f"   ✅ 성공 ({execution_time:.2f}초)")
                
            except Exception as e:
                execution_time = time.time() - start_time
                result = {
                    "model": model_config['name'],
                    "response": None,
                    "execution_time": execution_time,
                    "response_length": 0,
                    "word_count": 0,
                    "success": False,
                    "error": str(e)
                }
                
                print(f"   ❌ 실패: {e}")
            
            results.append(result)
            version_manager.logger.info(f"[{model_config['name']}] 실행시간: {execution_time:.2f}초, 성공: {result['success']}")
        
        # 결과 출력
        print("\n" + "="*80)
        print("📊 비교 결과")
        print("="*80)
        
        for result in results:
            if result["success"]:
                print(f"\n🤖 **{result['model']}**")
                print(f"   📝 응답: {result['response']}")
                print(f"   ⏱️  실행시간: {result['execution_time']:.2f}초")
                print(f"   📏 응답길이: {result['response_length']}자")
                print(f"   📊 단어수: {result['word_count']}개")
                print("-" * 60)
            else:
                print(f"\n❌ **{result['model']}** - 실패")
                print(f"   오류: {result['error']}")
                print("-" * 60)
        
        # 성능 비교 분석
        successful_results = [r for r in results if r["success"]]
        
        if len(successful_results) >= 2:
            print(f"\n🏆 성능 비교 분석")
            print("=" * 60)
            
            # 속도 비교
            fastest = min(successful_results, key=lambda x: x["execution_time"])
            slowest = max(successful_results, key=lambda x: x["execution_time"])
            speed_diff = ((slowest["execution_time"] - fastest["execution_time"]) / fastest["execution_time"]) * 100
            
            print(f"⚡ 속도: {fastest['model']}이 {speed_diff:.1f}% 빠름")
            print(f"   - {fastest['model']}: {fastest['execution_time']:.2f}초")
            print(f"   - {slowest['model']}: {slowest['execution_time']:.2f}초")
            
            # 응답 길이 비교  
            longest = max(successful_results, key=lambda x: x["response_length"])
            shortest = min(successful_results, key=lambda x: x["response_length"])
            length_diff = ((longest["response_length"] - shortest["response_length"]) / shortest["response_length"]) * 100
            
            print(f"\n📏 응답 길이: {longest['model']}이 {length_diff:.1f}% 길음")
            print(f"   - {longest['model']}: {longest['response_length']}자")
            print(f"   - {shortest['model']}: {shortest['response_length']}자")
            
            # 응답 스타일 분석
            print(f"\n📋 응답 스타일 분석:")
            for result in successful_results:
                response_str = str(result['response'])
                has_page_ref = '페이지' in response_str or 'page' in response_str.lower()
                has_specific_date = '2023년 10월 30일' in response_str
                
                print(f"   - {result['model']}:")
                print(f"     • 페이지 참조: {'✅' if has_page_ref else '❌'}")
                print(f"     • 구체적 날짜: {'✅' if has_specific_date else '❌'}")
                print(f"     • 간결성: {'높음' if result['response_length'] < 100 else '보통'}")
        
        # LangSmith 정보
        if langsmith.enabled:
            session_info = langsmith.get_session_info()
            print(f"\n🔍 LangSmith 추적: {session_info['url']}")
        
        version_manager.logger.info("=== 바이든 질문 모델 비교 완료 ===")
        
    except Exception as e:
        version_manager.logger.error(f"비교 중 오류 발생: {e}")
        print(f"❌ 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()