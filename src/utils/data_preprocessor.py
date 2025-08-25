import pandas as pd
import re
from typing import List
from langchain_core.documents import Document

def preprocess_legal_documents(documents: List[Document]) -> List[Document]:
    """
    법률 문서 전처리를 위한 함수
    EDA 분석 결과를 바탕으로 다음 처리를 수행:
    1. 문서 길이 필터링 (너무 짧은 문서 제거)
    2. 법령 정보 추출 및 메타데이터 추가
    3. 텍스트 정제
    """
    processed_documents = []
    
    for doc in documents:
        # 1. 너무 짧은 문서 필터링 (글자수 기준)
        if len(doc.page_content) < 50:  # EDA 결과 판시사항 최소 길이 고려
            continue
            
        # 2. 텍스트 정제
        cleaned_content = clean_legal_text(doc.page_content)
        
        # 3. 법령 정보 추출
        laws = extract_laws(cleaned_content)
        
        # 4. 메타데이터 업데이트
        updated_metadata = doc.metadata.copy()
        updated_metadata['extracted_laws'] = laws
        updated_metadata['content_length'] = len(cleaned_content)
        updated_metadata['word_count'] = len(cleaned_content.split())
        
        # 5. 처리된 문서 생성
        processed_doc = Document(
            page_content=cleaned_content,
            metadata=updated_metadata
        )
        
        processed_documents.append(processed_doc)
    
    return processed_documents

def clean_legal_text(text: str) -> str:
    """
    법률 문서 텍스트 정제
    """
    # 연속된 공백을 단일 공백으로 변환
    text = re.sub(r'\s+', ' ', text)
    
    # 특수 문자 및 불필요한 기호 제거 (법률 문서에 필요한 것들은 보존)
    text = re.sub(r'[^\w\s\.\,\:\;\(\)\[\]【】\n가-힣]', '', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text

def extract_laws(text: str) -> List[str]:
    """
    텍스트에서 법령명 추출 (EDA 분석 결과 기반)
    """
    law_patterns = [
        r'[가-힣]+법',
        r'[가-힣]+령',
        r'[가-힣]+규칙',
        r'근로기준법',
        r'산업재해보상보험법',
        r'민법',
        r'개발법',
        r'사립학교법',
        r'변호사법'
    ]
    
    extracted_laws = set()
    for pattern in law_patterns:
        matches = re.findall(pattern, text)
        extracted_laws.update(matches)
    
    return list(extracted_laws)

def filter_documents_by_case_type(documents: List[Document], target_types: List[str] = None) -> List[Document]:
    """
    사건종류명을 기반으로 문서 필터링
    EDA 결과에서 주요 사건종류: 일반행정, 근로_임금, 민사, 특허 등
    """
    if target_types is None:
        # EDA 결과 기반 주요 사건종류
        target_types = ['일반행정', '근로_임금', '민사', '특허', '손해배상']
    
    filtered_docs = []
    for doc in documents:
        case_type = doc.metadata.get('사건종류명', '')
        if any(target_type in str(case_type) for target_type in target_types):
            filtered_docs.append(doc)
    
    return filtered_docs

def add_court_hierarchy_info(documents: List[Document]) -> List[Document]:
    """
    법원 계층 정보 추가 (EDA 결과에서 대법원이 가장 많음)
    """
    court_hierarchy = {
        '대법원': '최고법원',
        '고등법원': '항소법원',
        '지방법원': '1심법원',
        '행정법원': '특별법원',
        '특허법원': '특별법원'
    }
    
    processed_docs = []
    for doc in documents:
        court_name = str(doc.metadata.get('법원명', ''))
        
        hierarchy = '기타'
        for court_type, level in court_hierarchy.items():
            if court_type in court_name:
                hierarchy = level
                break
        
        updated_metadata = doc.metadata.copy()
        updated_metadata['court_hierarchy'] = hierarchy
        
        processed_doc = Document(
            page_content=doc.page_content,
            metadata=updated_metadata
        )
        processed_docs.append(processed_doc)
    
    return processed_docs

def enhance_metadata_with_year(documents: List[Document]) -> List[Document]:
    """
    선고일자에서 연도 정보 추출 및 메타데이터 추가
    """
    processed_docs = []
    for doc in documents:
        judgment_date = str(doc.metadata.get('선고일자', ''))
        
        # 연도 추출 (YYYY-MM-DD 형식)
        year_match = re.match(r'(\d{4})', judgment_date)
        year = year_match.group(1) if year_match else 'unknown'
        
        updated_metadata = doc.metadata.copy()
        updated_metadata['judgment_year'] = year
        
        processed_doc = Document(
            page_content=doc.page_content,
            metadata=updated_metadata
        )
        processed_docs.append(processed_doc)
    
    return processed_docs