from typing import List

import hydra
from omegaconf import DictConfig

from rag.pipeline.simple import SimplePipeline
from rag.trace.langsmith import init_langsmith_trace
from rag.utils.env_loader import load_env
from rag.utils.logger import get_logger

logger = get_logger(__name__)


def make_pipeline(cfg):
    scenario = cfg.exp.scenario
    if scenario == "simple":
        return SimplePipeline(cfg)
    elif scenario == "comparison":
        raise NotImplementedError()
    elif scenario == "demo":
        return SimplePipeline(cfg)
    return None


def report_responses(responses: List[str]):
    logger.info("=" * 80)
    for i, res in enumerate(responses):
        logger.info(f"LLM 응답[{i}]:\n{res}")
    logger.info("=" * 80 + "\n")
    # TODO: 리포트 파일로 저장


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info(f">>> {cfg.project.name} ({cfg.project.version}) <<<")

    cfg = load_env(cfg)
    init_langsmith_trace(cfg)

    ppl = make_pipeline(cfg)

    # responses = ppl.run(cfg.exp.question)
    responses = ppl.run_multi_turn(cfg.exp.questions)
    report_responses(responses)

    logger.info(f">>> complete <<<")


if __name__ == "__main__":
    main()
