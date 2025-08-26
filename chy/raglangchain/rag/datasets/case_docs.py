import json
import pathlib

from langchain_core.documents import Document

from rag.datasets.base import BaseDataset
from rag.utils.logger import get_logger

logger = get_logger(__name__)


def format_case(case_text: dict) -> str:
    return f"""
        사건번호: {case_text.get('사건번호', '') or 'N/A'}
        사건명: {case_text.get('사건명', '') or 'N/A'}
        법원명: {case_text.get('법원명', '') or 'N/A'}
        선고일자: {case_text.get('선고일자', '') or 'N/A'}
        사건종류: {case_text.get('사건종류명', '') or 'N/A'}

        판시사항:
        {case_text.get('판시사항', '') or 'N/A'}

        판결요지:
        {case_text.get('판결요지', '') or 'N/A'}

        참조조문:
        {case_text.get('참조조문', '') or 'N/A'}

        판례내용:
        {case_text.get('판례내용', '') or 'N/A'}
    """.strip()


def case_json_to_doc(case_text):
    datat = {
        'case_number': case_text.get('사건번호', ''),
        'case_name': case_text.get('사건명', ''),
        'court': case_text.get('법원명', ''),
        'date': case_text.get('선고일자', ''),
        'case_type': case_text.get('사건종류명', ''),
        'summary': case_text.get('판시사항', ''),
        'decision': case_text.get('판결요지', ''),
        'references': case_text.get('참조조문', ''),
        'content': case_text.get('판례내용', ''),
    }

    metadata = {
        'case_number': case_text.get('사건번호', ''),
        'case_name': case_text.get('사건명', ''),
        'court': case_text.get('법원명', ''),
        'date': case_text.get('선고일자', ''),
        'case_type': case_text.get('사건종류명', ''),
        'summary': case_text.get('판시사항', ''),
        'decision': case_text.get('판결요지', ''),
        'references': case_text.get('참조조문', ''),
        'content': case_text.get('판례내용', ''),
    }

    origin_text = format_case(case_text)
    return Document(page_content=origin_text, metadata=metadata)


class CaseDocsDataset(BaseDataset):
    def load_docs(self):
        ds_dir = pathlib.Path(self.dataset_dir)
        if not ds_dir.exists():
            raise FileNotFoundError(f"법률 데이터 디렉토리를 찾을 수 없습니다: {self.dataset_dir}")
        else:
            logger.info(f"법률 데이터 디렉토리: {self.dataset_dir}")

        docs = []
        json_paths = list(ds_dir.glob("*.json"))
        for file_path in json_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    case_text = json.load(f)
                docs.append(case_json_to_doc(case_text))
            except Exception as e:
                logger.error(f"판례 파싱 오류 {file_path}: {e}")
                continue

        self.docs = docs
        logger.info(f"총 {len(self.docs)}개 판례 로드 완료")

    def get_docs(self):
        return self.docs
