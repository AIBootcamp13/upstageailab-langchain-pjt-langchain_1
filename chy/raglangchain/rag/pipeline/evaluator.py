from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from rag.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluatorPipeline:
    def __init__(
        self,
        llm_builders: Dict[str, callable],
        retriever_builders: Dict[str, callable],
        prompt,
        evaluator_runner,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_workers: int = 4,
        corpus: list | None = None,
    ):
        self.llm_builders = llm_builders
        self.retriever_builders = retriever_builders
        self.prompt = prompt
        self.evaluator_runner = evaluator_runner
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.corpus = corpus or []

    def _build_retriever(self, label: str):
        r = self.retriever_builders[label]()
        # build(corpus) 지원 시 인덱스 구축
        if hasattr(r, "build"):
            r._build(self.corpus)
        return r

    def _run_one(self, llm_label: str, retriever_label: str, question: str) -> dict:
        llm = self.llm_builders[llm_label]()
        retriever = self._build_retriever(retriever_label)

        # 순수 LLM
        pure_prompt = self.prompt.format(question=question, docs=[])
        pure_resp = {"answer": llm.generate(pure_prompt, temperature=self.temperature, max_tokens=self.max_tokens)}

        # RAG
        from rag.pipeline.simple_rag import SimpleRAG

        rag = SimpleRAG(llm=llm, retriever=retriever, prompt=self.prompt)
        rag_resp = rag.run(question, temperature=self.temperature, max_tokens=self.max_tokens)

        # 평가 기록
        metrics = self.evaluator_runner.run_single(llm_label, retriever_label, question, pure_resp, rag_resp)
        return {
            "llm": llm_label,
            "retriever": retriever_label,
            "question": question,
            "pure": pure_resp,
            "rag": rag_resp,
            "metrics": metrics["metrics"],
        }

    def run(self, questions: List[str], models: List[str], retrievers: List[str], progress_cb=None) -> dict:
        results = {"models": models, "retriever": retrievers, "total_questions": len(questions), "items": []}
        tasks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for q in questions:
                for m in models:
                    for r in retrievers:
                        tasks.append(ex.submit(self._run_one, m, r, q))
            total = len(tasks)
            done = 0
            for fut in as_completed(tasks):
                try:
                    results["items"].append(fut.result())
                except Exception as e:
                    logger.error(f"evaluation task error: {e}")
                done += 1
                if progress_cb:
                    progress_cb(done / max(total, 1))
        return results
