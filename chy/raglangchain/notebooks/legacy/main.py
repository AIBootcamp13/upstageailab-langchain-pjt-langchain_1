import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from src.chains.qa_chain import get_qa_chain
from src.components.embeddings import get_embedding_model
from src.components.llms import get_llm
from src.components.retrievers import get_retriever
from src.components.vectorstores import get_vector_store
from src.prompts.qa_prompts import get_qa_prompt
from src.utils.document_loaders import load_documents
from src.utils.langsmith_simple import LangSmithSimple
from src.utils.text_splitters import split_documents

# 버전 관리 시스템 임포트
from src.utils.version_manager import VersionManager


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    load_dotenv()

    # 버전 관리 시스템 초기화
    version_manager = VersionManager()
    version_manager.logger.info("=== 새로운 실행 시작 ===")
    version_manager.logger.info(f"설정 정보:\n{OmegaConf.to_yaml(cfg)}")

    # LangSmith 자동 추적 시스템 초기화
    langsmith = LangSmithSimple(cfg, version_manager)

    print(OmegaConf.to_yaml(cfg))

    try:
        # 문서 로드 (자동 추적됨)
        documents = load_documents(cfg)
        print(f"문서의 페이지수: {len(documents)}")

        # 문서 분할 (자동 추적됨)
        split_documents_list = split_documents(cfg, documents)
        print(f"분할된 청크의수: {len(split_documents_list)}")

        # 임베딩 생성 (자동 추적됨)
        embeddings = get_embedding_model(cfg)

        # 벡터 스토어 생성 (자동 추적됨)
        vectorstore = get_vector_store(cfg, split_documents_list, embeddings)

        # Retriever 생성 (자동 추적됨)
        if cfg.retriever.type == "bm25" or cfg.retriever.type == "ensemble":
            retriever = get_retriever(cfg, vectorstore, documents=split_documents_list)
        else:  # cfg.retriever.type == "vectorstore"
            retriever = get_retriever(cfg, vectorstore)

        # LLM 및 체인 생성 (자동 추적됨)
        llm = get_llm(cfg)  # LLM 로드
        prompt = get_qa_prompt()  # Prompt 로드
        qa_chain = get_qa_chain(llm, retriever, prompt)  # QA Chain 생성

        # 질문 및 답변 (자동 추적됨)
        question = "미국 바이든 대통령이 몇년 몇월 몇일에 연방정부 차원에서 안전하고 신뢰할 수 있는 AI 개발과 사용을 보장하기 위한 행정명령을 발표했나요?"
        response = qa_chain.invoke(question)

        # 결과 로깅
        version_manager.logger.info(f"질문: {question}")
        version_manager.logger.info(f"답변: {response}")

        # LangSmith 정보 로깅
        if langsmith.enabled:
            session_info = langsmith.get_session_info()
            version_manager.logger.info(f"LangSmith 프로젝트: {session_info['project']}")
            version_manager.logger.info(f"LangSmith 세션: {session_info['session']}")
            version_manager.logger.info(f"LangSmith 대시보드: {session_info['url']}")

        version_manager.logger.info("=== 실행 완료 ===")

        print(f"질문: {question}")
        print(f"답변: {response}")

        # LangSmith 추적 정보 출력
        if langsmith.enabled:
            session_info = langsmith.get_session_info()
            print(f"\n🔍 LangSmith 추적 정보:")
            print(f"   프로젝트: {session_info['project']}")
            print(f"   세션: {session_info['session']}")
            print(f"   대시보드: {session_info['url']}")

    except Exception as e:
        version_manager.logger.error(f"실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
