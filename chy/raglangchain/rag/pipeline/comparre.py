from hydra.utils import instantiate

from rag.evaluation.langsmith_runner import LangSmithEvaluatorRunner
from rag.pipeline.evaluator import EvaluatorPipeline
from rag.prompts.qa_query_eval import QuerySamples
from rag.utils.logger import get_logger

logger = get_logger(__name__)


def comparator(cfg, corpus, prompt):
    def llm_build(label: str):
        from omegaconf import OmegaConf

        def _b():
            return instantiate(OmegaConf.load(f"conf/llm/{label}.yaml"))

        return _b

    def ret_build(label: str):
        from omegaconf import OmegaConf

        def _b():
            r = instantiate(OmegaConf.load(f"conf/retriever/{label}.yaml"))
            if hasattr(r, "build"):
                r._build(corpus)
            return r

        return _b

    llm_builders = {label: llm_build(label) for label in cfg.exp.models}
    retriever_builders = {label: ret_build(label) for label in cfg.exp.retrievers}
    evaluator = LangSmithEvaluatorRunner(
        enabled=cfg.langsmith.enabled, project=cfg.langsmith.project, tags=list(cfg.langsmith.tags)
    )
    eval_pipe = EvaluatorPipeline(
        llm_builders,
        retriever_builders,
        prompt,
        evaluator,
        temperature=cfg.exp.temperature,
        max_tokens=cfg.exp.max_tokens,
        max_workers=4,
        corpus=corpus,
    )

    def progress_cb(p):
        logger.info(f"Progress: {p * 100:.1f}%")

    results = eval_pipe.run(QuerySamples.for_eval(), cfg.exp.models, cfg.exp.retrievers, progress_cb)
    logger.info(f"Total eval items: {len(results['items'])}")
