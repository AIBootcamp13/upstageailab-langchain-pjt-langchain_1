import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from rag.pipeline.simple import SimplePipeline
from rag.pipeline.pipeline_base import BasePipeline
from rag.trace.langsmith import init_langsmith_trace
from rag.utils.env_loader import extend_cfg_with_env
from rag.utils.logger import get_logger, log_block

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


def build_pipeline_from_cfg(cfg) -> BasePipeline:
    with log_block(logger):
        logger.info(f"loaded cfg: {cfg}")

    cfg = extend_cfg_with_env(cfg)
    init_langsmith_trace(cfg)
    pipeline = make_pipeline(cfg)

    if not pipeline:
        raise NotImplementedError(f"unsupported scenario: {cfg.exp.scenario}")
    return pipeline


def load_pipeline_with_opts(config_dir="conf", opts=None) -> BasePipeline:
    if opts is None:
        opts = ["exp.sid=session_100"]

    GlobalHydra.instance().clear()
    abs_conf_dir = str(Path(os.getcwd()).resolve() / config_dir)
    with initialize_config_dir(config_dir=abs_conf_dir, version_base=None):
        cfg = compose(config_name="config", overrides=opts)

    return build_pipeline_from_cfg(cfg)
