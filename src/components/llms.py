from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(cfg):
    if cfg.llm.provider == "openai":
        return ChatOpenAI(model_name=cfg.llm.model_name, temperature=cfg.llm.temperature)
    elif cfg.llm.provider == "google":
        return ChatGoogleGenerativeAI(model=cfg.llm.model_name, temperature=cfg.llm.temperature)
    else:
        raise ValueError(f"Unsupported LLM provider: {cfg.llm.provider}")
