from collections.abc import Callable
from functools import wraps

type SltPredicateFn[T] = Callable[[T], bool]


class SltPredicate[T]:
    __slots__ = ("fn", "description", "name")

    def __init__(self, fn: SltPredicateFn[T], description: str):
        self.fn = fn
        self.description = description
        self.name: str = ""

    def __call__(self, object: T) -> bool:
        return self.fn(object)

    def __str__(self) -> str:
        return self.description

    def __repr__(self) -> str:
        return self.description

    def __and__(self, other: SltPredicate[T]) -> SltPredicate[T]:
        return SltPredicate[T](
            lambda object: self(object) and other(object), f"({self!r}) and ({other!r})"
        )

    def __or__(self, other: SltPredicate[T]) -> SltPredicate[T]:
        return SltPredicate[T](
            lambda object: self(object) or other(object), f"({self!r}) or ({other!r})"
        )

    def __not__(self) -> SltPredicate[T]:
        return SltPredicate[T](lambda object: not self(object), f"not ({self!r})")

    def __invert__(self) -> SltPredicate[T]:
        return SltPredicate[T](lambda object: not self(object), f"not ({self!r})")

    def __eq__(self, other: SltPredicate[T]) -> bool:
        return self.fn == other.fn

    def __ne__(self, other: SltPredicate) -> bool:
        return self.fn != other.fn


def rule[T](fn: SltPredicateFn[T]) -> SltPredicate[T]:
    @wraps(fn)
    def wrapper(object: T) -> bool:
        return fn(object)

    return SltPredicate[T](wrapper, fn.__name__)
