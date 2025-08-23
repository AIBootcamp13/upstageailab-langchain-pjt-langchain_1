#!/usr/bin/env python3
"""
법률 도메인 LLM 모델 비교 시스템 - 새 버전
JSON 판례 데이터를 기반으로 한 RAG 시스템으로 GPT-4o vs Claude-3.5-Haiku 비교
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.utils.path_utils import get_project_root, ensure_directory_exists
from src.utils.version_manager import VersionManager
from src.utils.langsmith_simple import LangSmithSimple
from src.utils.model_comparison import ModelComparison
from src.components.llms import get_llm
from src.components.embeddings import create_embeddings
from src.components.vectorstores import create_vectorstore
from src.components.retrievers import create_retriever
from src.chains.qa_chain import create_qa_chain
from src.prompts.qa_prompts import get_qa_prompt


class LawDocumentLoader:
    """법률 JSON 문서 로더"""
    
    def __init__(self, law_data_dir: str = "data/law"):
        self.law_data_dir = Path(law_data_dir)
        
    def load_legal_documents(self):
        """법률 JSON 파일들 로드"""
        documents = []
        
        if not self.law_data_dir.exists():
            raise FileNotFoundError(f"법률 데이터 디렉토리를 찾을 수 없습니다: {self.law_data_dir}")
        
        json_files = list(self.law_data_dir.glob("*.json"))
        print(f"발견된 JSON 파일: {len(json_files)}개")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)
                
                # JSON 데이터를 텍스트로 변환
                document_text = self._format_legal_case(case_data)
                
                documents.append({
                    'content': document_text,
                    'metadata': {
                        'source': str(json_file),
                        'case_number': case_data.get('사건번호', ''),
                        'case_name': case_data.get('사건명', ''),
                        'court': case_data.get('법원명', ''),
                        'date': case_data.get('선고일자', ''),
                        'case_type': case_data.get('사건종류명', '')
                    }
                })
                
                print(f"로드 완료: {case_data.get('사건번호', '')} - {case_data.get('사건명', '')}")
                
            except Exception as e:
                print(f"파일 로드 오류 {json_file}: {e}")
                continue
                
        print(f"총 {len(documents)}개 법률 문서 로드 완료")
        return documents
    
    def _format_legal_case(self, case_data: dict) -> str:
        """법률 사건 데이터를 텍스트로 포맷팅"""
        formatted_text = f"""
==== 법률 판례 정보 ====
사건번호: {case_data.get('사건번호', 'N/A')}
사건명: {case_data.get('사건명', 'N/A')}
법원명: {case_data.get('법원명', 'N/A')}
선고일자: {case_data.get('선고일자', 'N/A')}
사건종류: {case_data.get('사건종류명', 'N/A')}

==== 판시사항 ====
{case_data.get('판시사항', 'N/A')}

==== 판결요지 ====
{case_data.get('판결요지', 'N/A')}

==== 참조조문 ====
{case_data.get('참조조문', 'N/A')}

==== 참조판례 ====
{case_data.get('참조판례', 'N/A')}

==== 판례내용 (상세) ====
{case_data.get('판례내용', 'N/A')}
"""
        return formatted_text.strip()


def create_law_rag_pipeline(cfg):
    """법률 도메인 RAG 파이프라인 생성"""
    
    print("=== 법률 도메인 RAG 파이프라인 초기화 ===")
    
    # 1. 법률 문서 로드
    law_loader = LawDocumentLoader()
    documents = law_loader.load_legal_documents()
    
    if not documents:
        raise ValueError("로드된 법률 문서가 없습니다.")
    
    # 2. 텍스트 분할 (법률 문서는 긴 텍스트이므로 청크 크기 조정)
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # 법률 문서는 더 큰 청크 사용
        chunk_overlap=200,
        separators=["====", "\n\n", "\n", " ", ""]
    )
    
    split_docs = []
    for doc in documents:
        splits = text_splitter.split_text(doc['content'])
        for i, split in enumerate(splits):
            split_docs.append({
                'page_content': split,
                'metadata': {**doc['metadata'], 'chunk': i}
            })
    
    print(f"총 {len(split_docs)}개 텍스트 청크 생성")
    
    # 3. 임베딩 및 벡터스토어 생성
    embeddings = create_embeddings(cfg)
    vectorstore = create_vectorstore(cfg, split_docs, embeddings)
    
    # 4. 검색기 생성 (법률 도메인용으로 조정)
    retriever = create_retriever(cfg, vectorstore, split_docs)
    
    return retriever


def main():
    """메인 실행 함수"""
    
    # 환경 설정
    load_dotenv()
    project_root = get_project_root()
    os.chdir(project_root)
    
    # 버전 매니저 초기화
    version_manager = VersionManager()
    version_manager.logger.info("=== 법률 도메인 LLM 모델 비교 시작 ===")
    
    # 설정 로드
    cfg = OmegaConf.create({
        'embedding': {
            'provider': 'openai',
            'model_name': 'text-embedding-3-small',
            'chunk_size': 1000
        },
        'vector_store': {
            'type': 'faiss',
            'persist_directory': 'faiss_db_law'
        },
        'retriever': {
            'search_type': 'similarity',
            'search_kwargs': {'k': 5}  # 법률 문서는 더 많은 관련 문서 검색
        },
        'llm': {
            'temperature': 0.1,  # 법률 답변은 더 정확해야 함
            'max_tokens': 1000
        },
        'langsmith': {
            'enabled': True,
            'project_name': 'law-domain-comparison',
            'session_name': f'law-session_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    })
    
    # LangSmith 초기화
    langsmith_manager = LangSmithSimple(cfg, version_manager)
    
    try:
        # RAG 파이프라인 생성
        retriever = create_law_rag_pipeline(cfg)
        
        # 법률 도메인 특화 질문들
        test_questions = [
            "취업규칙을 근로자에게 불리하게 변경할 때 사용자가 지켜야 할 법적 요건은 무엇인가요?",
            "퇴직금 지급 기일을 연장하는 합의를 했더라도 연장된 기일까지 지급하지 않으면 형사처벌을 받나요?"
        ]
        
        print(f"\n=== 법률 질문 {len(test_questions)}개로 모델 비교 시작 ===")
        
        # 모델 설정들
        models_config = [
            {
                'name': 'GPT-4o',
                'provider': 'openai',
                'model_name': 'gpt-4o',
                'temperature': cfg.llm.temperature
            },
            {
                'name': 'Claude-3.5-Haiku',
                'provider': 'anthropic', 
                'model_name': 'claude-3-5-haiku-20241022',
                'temperature': cfg.llm.temperature
            }
        ]
        
        # 모델 비교 실행
        results = {}
        
        for question in test_questions:
            print(f"\n{'='*60}")
            print(f"질문: {question}")
            print(f"{'='*60}")
            
            question_results = {}
            
            for model_config in models_config:
                print(f"\n--- {model_config['name']} 응답 ---")
                
                start_time = time.time()
                
                # LLM 생성
                llm_cfg = OmegaConf.create({
                    'llm': {
                        'provider': model_config['provider'],
                        'model_name': model_config['model_name'],
                        'temperature': model_config['temperature']
                    }
                })
                
                llm = get_llm(llm_cfg)
                
                # QA 체인 생성 (법률 도메인용 프롬프트 사용)
                qa_chain = create_qa_chain(retriever, llm, get_qa_prompt())
                
                # 질문 실행
                try:
                    response = qa_chain.invoke({"question": question})
                    answer = response['answer'] if isinstance(response, dict) else str(response)
                    
                    response_time = time.time() - start_time
                    
                    print(f"답변: {answer}")
                    print(f"응답 시간: {response_time:.2f}초")
                    
                    question_results[model_config['name']] = {
                        'answer': answer,
                        'response_time': response_time,
                        'model_config': model_config
                    }
                        
                except Exception as e:
                    error_msg = f"오류 발생: {str(e)}"
                    print(error_msg)
                    question_results[model_config['name']] = {
                        'answer': error_msg,
                        'response_time': 0,
                        'model_config': model_config,
                        'error': True
                    }
            
            results[question] = question_results
        
        # 결과 저장 및 출력
        output_dir = ensure_directory_exists("results/law_comparison")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 결과 저장
        json_output_path = output_dir / f"law_comparison_results_{timestamp}.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 마크다운 보고서 생성
        report_path = output_dir / f"law_comparison_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 법률 도메인 LLM 모델 비교 보고서\n\n")
            f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 비교 모델\n")
            for model_config in models_config:
                f.write(f"- **{model_config['name']}**: {model_config['provider']} / {model_config['model_name']}\n")
            
            f.write("\n## 테스트 결과\n\n")
            
            for question, question_results in results.items():
                f.write(f"### 질문: {question}\n\n")
                
                for model_name, result in question_results.items():
                    f.write(f"#### {model_name}\n")
                    f.write(f"- **응답시간**: {result['response_time']:.2f}초\n")
                    f.write(f"- **답변**:\n```\n{result['answer']}\n```\n\n")
        
        print(f"\n=== 법률 도메인 모델 비교 완료 ===")
        print(f"결과 저장: {json_output_path}")
        print(f"보고서 저장: {report_path}")
        
        version_manager.logger.info(f"법률 도메인 모델 비교 완료 - 결과: {json_output_path}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        version_manager.logger.error(f"법률 도메인 모델 비교 중 오류: {e}")
        raise


if __name__ == "__main__":
    main()