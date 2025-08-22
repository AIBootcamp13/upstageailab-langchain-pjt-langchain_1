from langchain_community.vectorstores import FAISS, Chroma

def get_vector_store(cfg, documents, embeddings):
    if cfg.vector_store.type == "faiss":
        return FAISS.from_documents(documents=documents, embedding=embeddings)
    elif cfg.vector_store.type == "chromadb":
        return Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=cfg.vector_store.persist_directory)
    else:
        raise ValueError(f"Unsupported vector store type: {cfg.vector_store.type}")
