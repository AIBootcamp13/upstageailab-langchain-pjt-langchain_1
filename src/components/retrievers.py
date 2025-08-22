from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def get_retriever(cfg, vector_store, documents=None):
    if cfg.retriever.type == "bm25":
        if documents is None:
            raise ValueError("BM25Retriever requires documents to be initialized.")
        return BM25Retriever.from_documents(documents=documents)
    elif cfg.retriever.type == "vectorstore":
        return vector_store.as_retriever(search_kwargs={"k": cfg.chain.retriever_k})
    elif cfg.retriever.type == "ensemble":
        if documents is None:
            raise ValueError("EnsembleRetriever with BM25 requires documents to be initialized.")
        bm25_retriever = BM25Retriever.from_documents(documents=documents)
        vectorstore_retriever = vector_store.as_retriever(search_kwargs={"k": cfg.chain.retriever_k})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vectorstore_retriever], weights=cfg.retriever.weights
        )
        return ensemble_retriever
    else:
        raise ValueError(f"Unsupported retriever type: {cfg.retriever.type}")
