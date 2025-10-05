from typing import ParamSpec, TypeVar, Callable
from functools import wraps

P = ParamSpec('P')
R = TypeVar('R')

def inherit_signature(base_func: Callable[P, R]) -> Callable[[Callable], Callable[P, R]]:
    """Inherit signature from base function."""
    def decorator(wrapper_func: Callable) -> Callable[P, R]:
        @wraps(base_func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            return wrapper_func(*args, **kwargs)
        return inner
    return decorator