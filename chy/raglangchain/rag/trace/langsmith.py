import langsmith as ls
from langsmith import traceable
from omegaconf import DictConfig

from rag.utils.logger import get_logger

logger = get_logger(__name__)


@traceable()
def check_ls_runtime(cfg):
    run_tree = ls.get_current_run_tree()
    logger.info(f"langsmith session trace id: {run_tree.trace_id}")

    if run_tree:
        # TODO: 이 함수의 URL로 나오게 되어 수정 필요
        logger.info(f"langsmith link: {run_tree.get_url()}")


def init_langsmith_trace(cfg: DictConfig):
    logger.info("=" * 80)

    if not cfg.langsmith.enabled:
        logger.info("langsmith tracer (disabled)")
        logger.info("=" * 80)
        return

    logger.info("init langsmith tracer (enabled)")
    lcfg = cfg.langsmith

    ls.configure(
        enabled=True,
        project_name=lcfg.project,
        metadata=lcfg.metadata,
    )

    check_ls_runtime(lcfg)

    logger.info("=" * 80)
