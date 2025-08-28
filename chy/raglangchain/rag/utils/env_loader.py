import os

from dotenv import load_dotenv
from langsmith import traceable
from omegaconf import OmegaConf

from rag.utils.logger import get_logger

logger = get_logger(__name__)


def extend_cfg_with_env(cfg):
    load_dotenv()

    # custom injection
    upstage_api_key = os.getenv("UPSTAGE_API_KEY")
    OmegaConf.update(cfg, "llm.api_key", upstage_api_key)

    logger.info(f"loaded env file (done)")
    return cfg
