from langchain_community.document_loaders import PyMuPDFLoader

def load_documents(cfg):
    if cfg.data.path.endswith(".pdf"):
        loader = PyMuPDFLoader(cfg.data.path)
        return loader.load()
    else:
        raise ValueError(f"Unsupported document type for path: {cfg.data.path}")
