import json
import pathlib
from typing import List, Dict

from rag.utils.logger import get_logger

logger = get_logger(__name__)


"""
# 공식 판례 정보 샘플 링크
- https://law.go.kr/LSW/precInfoP.do?mode=0&precSeq=231669#sa
"""


def make_document(data, file_case_number: str):
    issues = data.get("판시사항", "")
    case_details = data.get("판례내용", "")
    summary = data.get("판결요지", "")
    case_title = data.get("사건명", "")
    case_number = data.get("사건번호", "") or file_case_number

    title = case_title or case_number
    case_law_text = f"{issues}\n{case_details}\n{summary}"

    # fmt: off
    doc = {
        "id": case_number,
        "title": title,
        "text": case_law_text,
        "metadata": {"file": file_case_number}
    }
    # fmt: on

    return doc


class DatasetRepository:
    def __init__(self, ds_dir: str = "data/law"):
        self.dataset_root = pathlib.Path(ds_dir)

    def load_docs(self) -> List[Dict]:
        docs = []

        if not self.dataset_root.exists():
            logger.warning(f"{self.dataset_root} 경로가 없습니다.")
            return docs

        for path in self.dataset_root.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.exception(f"{path} 데이터 파싱 에러")
                continue

            case_number = str(path.stem)
            docs.append(make_document(data, case_number))

        if not docs:
            logger.warning(f"{self.dataset_root}에 데이터가 없습니다. RAG은 제한됩니다")

        return docs
