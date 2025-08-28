from typing import List

import hydra
from omegaconf import DictConfig

from rag.pipeline.factory import build_pipeline_from_cfg
from rag.utils.logger import get_logger

logger = get_logger(__name__)


def report_responses(responses: List[str] | str):
    if isinstance(responses, str):
        responses = [responses]

    logger.info("=" * 80)
    for i, res in enumerate(responses):
        logger.info(f"LLM 응답[{i}]:\n{res}")
    logger.info("=" * 80 + "\n")
    # TODO: 리포트 파일로 저장


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info(f">>> {cfg.project.name} ({cfg.project.version}) <<<")

    ppl = build_pipeline_from_cfg(cfg)
    # responses = ppl.run(cfg.exp.question)
    responses = ppl.run_multi_turn(cfg.exp.questions)
    report_responses(responses)

    logger.info(f">>> complete <<<")


if __name__ == "__main__":
    main()
