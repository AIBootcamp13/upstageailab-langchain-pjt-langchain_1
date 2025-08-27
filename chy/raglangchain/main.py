import hydra
from langsmith import traceable
from omegaconf import DictConfig

from rag.pipeline.simple import SimplePipeline
from rag.trace.langsmith import init_langsmith_trace
from rag.utils.env_loader import load_env
from rag.utils.logger import get_logger

logger = get_logger(__name__)


@traceable(name="pipeline switch")
def make_pipeline(cfg):
    scenario = cfg.exp.scenario
    if scenario == "simple":
        return SimplePipeline(cfg)
    elif scenario == "comparison":
        raise NotImplementedError()
    elif scenario == "demo":
        return SimplePipeline(cfg)
    return None


def report_response(out):
    logger.info("=" * 80)
    logger.info(f"LLM 응답:\n{out}")
    logger.info("=" * 80 + "\n")
    # TODO: 리포트 파일로 저장


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info(f">>> {cfg.project.name} ({cfg.project.version}) <<<")

    cfg = load_env(cfg)
    init_langsmith_trace(cfg)

    ppl = make_pipeline(cfg)
    out = ppl.run(cfg.exp.question)
    report_response(out)

    logger.info(f">>> complete <<<")


if __name__ == "__main__":
    main()
