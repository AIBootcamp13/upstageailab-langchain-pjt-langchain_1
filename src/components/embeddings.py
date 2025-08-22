from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embedding_model(cfg):
    if cfg.embedding.provider == "openai":
        return OpenAIEmbeddings(model=cfg.embedding.model_name)
    elif cfg.embedding.provider == "google":
        return GoogleGenerativeAIEmbeddings(model=cfg.embedding.model_name)
    else:
        raise ValueError(f"Unsupported embedding provider: {cfg.embedding.provider}")
