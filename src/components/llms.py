from langchain_openai import ChatOpenAI

def get_llm(cfg):
    if cfg.llm.provider == "openai":
        return ChatOpenAI(model_name=cfg.llm.model_name, temperature=cfg.llm.temperature)
    elif cfg.llm.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=cfg.llm.model_name, temperature=cfg.llm.temperature)
        except ImportError:
            raise ImportError("Anthropic 패키지가 설치되지 않았습니다. pip install langchain-anthropic를 실행하세요.")
    elif cfg.llm.provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=cfg.llm.model_name, temperature=cfg.llm.temperature)
        except ImportError:
            raise ImportError("Google Generative AI 패키지가 설치되지 않았습니다. pip install langchain-google-genai를 실행하세요.")
    else:
        raise ValueError(f"Unsupported LLM provider: {cfg.llm.provider}")
