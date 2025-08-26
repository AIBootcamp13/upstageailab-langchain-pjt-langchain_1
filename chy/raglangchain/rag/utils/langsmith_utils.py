import os
from contextlib import contextmanager


def is_enabled():
    return os.getenv("LANGSMITH_API_KEY") is not None


@contextmanager
def maybe_traceable(name: str, metadata: dict = None):
    try:
        from langsmith import traceable

        @traceable(name=name, metadata=metadata or {})
        def identity(x):
            return x

        yield identity
    except Exception:
        yield lambda x: x
