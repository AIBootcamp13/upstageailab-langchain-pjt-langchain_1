from langchain_huggingface import HuggingFaceEmbeddings

"""
NOTE
- 이 임베딩 모델을 사용하려면 einops 파이썬 패키지가 필요 (pip install einops / uv add einops)

- 가끔 file not found 에러 발생 시 > .cache 디렉토리 지우고 새롭게 파일 받기

- flash-attn 설치 이슈로 실행마다 아래 로그가 반복적 찍힘 (미관상 좋지 않음)
- flash_attn is not installed. Using PyTorch native attention implementation.
  (업스테이지 환경에서는 외부에서 빌드한 휠을 주입하지 않는 이상 어쩔 수 없음)
"""


class JinaV3Embedder:
    MODEL_NAME = "jinaai/jina-embeddings-v3"

    def __init__(self):
        kwargs = {"trust_remote_code": True}
        self.emb = HuggingFaceEmbeddings(model_name=self.MODEL_NAME, model_kwargs=kwargs)
