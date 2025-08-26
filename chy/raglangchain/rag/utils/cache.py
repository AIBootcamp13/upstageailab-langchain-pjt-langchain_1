import functools
from typing import Callable, Any, Tuple


class ResponseCache:
    def __init__(self):
        self._cache = {}

    def get(self, key: Tuple):
        return self._cache.get(key)

    def set(self, key: Tuple, value: Any):
        self._cache[key] = value


cache = ResponseCache()


def cache_response(func: Callable):
    @functools.wraps(func)
    def wrapper(self, prompt: str, **kwargs):
        key = (self.__class__.__name__, getattr(self, "model", None), prompt, tuple(sorted(kwargs.items())))
        hit = cache.get(key)
        if hit is not None:
            return hit
        out = func(self, prompt, **kwargs)
        cache.set(key, out)
        return out

    return wrapper
