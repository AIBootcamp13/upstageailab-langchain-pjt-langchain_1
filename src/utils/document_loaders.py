from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import pandas as pd
from .data_preprocessor import preprocess_legal_documents, filter_documents_by_case_type, add_court_hierarchy_info, enhance_metadata_with_year

def load_documents(cfg):
    if cfg.data.path.endswith(".pdf"):
        loader = PyMuPDFLoader(cfg.data.path)
        return loader.load()
    elif cfg.data.path.endswith(".csv"):
        df = pd.read_csv(cfg.data.path)
        documents = []
        
        for idx, row in df.iterrows():
            # 중요한 텍스트 필드들을 결합하여 문서 내용 생성
            content_parts = []
            
            if pd.notna(row.get('판시사항', '')) and str(row.get('판시사항', '')) != 'nan':
                content_parts.append(f"판시사항: {row['판시사항']}")
            
            if pd.notna(row.get('판결요지', '')) and str(row.get('판결요지', '')) != 'nan':
                content_parts.append(f"판결요지: {row['판결요지']}")
                
            if pd.notna(row.get('판례내용', '')) and str(row.get('판례내용', '')) != 'nan':
                content_parts.append(f"판례내용: {row['판례내용']}")
            
            if pd.notna(row.get('참조조문', '')) and str(row.get('참조조문', '')) != 'nan':
                content_parts.append(f"참조조문: {row['참조조문']}")
            
            page_content = '\n\n'.join(content_parts)
            
            # 메타데이터 설정
            metadata = {
                'source': f"판례_{idx}",
                '판례일련번호': str(row.get('판례일련번호', '')),
                '사건명': str(row.get('사건명', '')),
                '사건번호': str(row.get('사건번호', '')),
                '선고일자': str(row.get('선고일자', '')),
                '법원명': str(row.get('법원명', '')),
                '사건종류명': str(row.get('사건종류명', '')),
                '판결유형': str(row.get('판결유형', '')),
            }
            
            documents.append(Document(page_content=page_content, metadata=metadata))
        
        # CSV 데이터에 대해 전처리 적용
        if hasattr(cfg, 'preprocessing') and cfg.preprocessing.get('enabled', True):
            print(f"전처리 전 문서 수: {len(documents)}")
            
            # 1. 기본 전처리 (텍스트 정제, 법령 추출 등)
            documents = preprocess_legal_documents(documents)
            print(f"기본 전처리 후 문서 수: {len(documents)}")
            
            # 2. 사건종류별 필터링 (설정이 있는 경우)
            if cfg.preprocessing.get('filter_case_types', False):
                target_types = cfg.preprocessing.get('target_case_types', None)
                documents = filter_documents_by_case_type(documents, target_types)
                print(f"사건종류 필터링 후 문서 수: {len(documents)}")
            
            # 3. 법원 계층 정보 추가
            documents = add_court_hierarchy_info(documents)
            
            # 4. 연도 정보 추가
            documents = enhance_metadata_with_year(documents)
            
            print("전처리 완료")
        
        return documents
    else:
        raise ValueError(f"Unsupported document type for path: {cfg.data.path}")
