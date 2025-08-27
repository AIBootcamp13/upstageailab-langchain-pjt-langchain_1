from hydra.utils import instantiate

from rag.utils.logger import get_logger

logger = get_logger(__name__)


def comparator(cfg, corpus, prompt):
    # instantiate(OmegaConf.load(f"conf/llm/{label}.yaml"))
    # r = instantiate(OmegaConf.load(f"conf/retriever/{label}.yaml"))
    # llm_builders = {label: llm_build(label) for label in cfg.exp.models}
    # retriever_builders = {label: ret_build(label) for label in cfg.exp.retrievers}
    pass
