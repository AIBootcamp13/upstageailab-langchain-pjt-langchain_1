from langchain_openai import OpenAIEmbeddings

def get_embedding_model(cfg):
    if cfg.embedding.provider == "openai":
        return OpenAIEmbeddings(model=cfg.embedding.model_name)
    elif cfg.embedding.provider == "google":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(model=cfg.embedding.model_name)
        except ImportError:
            raise ImportError("Google Generative AI 패키지가 설치되지 않았습니다. pip install langchain-google-genai를 실행하세요.")
    else:
        raise ValueError(f"Unsupported embedding provider: {cfg.embedding.provider}")
