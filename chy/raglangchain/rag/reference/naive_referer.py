from typing import List

from langchain_core.documents import Document


def law_docs_to_ref(docs: List[Document]):
    # TODO
    #  만약 전체 문서의 경우 적절한 요약이나 처리를 추가할 수 있음
    #  부분 문자열이라도 메타데이터의 정보를 활용할 수 있음 (우선 순위, 중요도로 재선정 등)
    return "\n\n".join([doc.page_content for doc in docs])
    # return "\n".join([f"[문서 {i + 1}] {d.get('title', '')} {d.get('text', '')}" for i, d in enumerate(docs)])
