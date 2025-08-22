from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(cfg, documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.data.chunk_size,
        chunk_overlap=cfg.data.chunk_overlap
    )
    return text_splitter.split_documents(documents)
