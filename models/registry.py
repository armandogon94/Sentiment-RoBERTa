"""``@register("name")`` → ``create_model(name, cfg)``. All config-string dispatch lives here.

Consequence, and the reason for the pattern: ``train.py`` contains no ``if cfg.MODEL.NAME ==
...`` branch anywhere. It reads as orchestration, and adding a model touches this file and
one new module rather than the entrypoint.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from models.protocols import SentimentModel

_REGISTRY: dict[str, Callable[..., SentimentModel]] = {}

F = TypeVar("F", bound=Callable[..., SentimentModel])


def register(name: str) -> Callable[[F], F]:
    """Register a model factory under ``name``. Duplicate names are a hard error."""

    def decorator(factory: F) -> F:
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} is already registered by {_REGISTRY[name]!r}")
        _REGISTRY[name] = factory
        return factory

    return decorator


def create_model(kind: str, **kwargs: Any) -> SentimentModel:
    """Instantiate a registered model.

    The first parameter is ``kind``, not ``name``: models carry their own ``name`` (a row
    label that varies per ablation cell) and the two must not collide at the call site.
    """
    _load_builtins()
    if kind not in _REGISTRY:
        raise KeyError(f"unknown model {kind!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[kind](**kwargs)


def registered_names() -> list[str]:
    """Every registered model name. Used by ``tests/test_models.py``."""
    _load_builtins()
    return sorted(_REGISTRY)


def _load_builtins() -> None:
    """Import the concrete model modules so their decorators run.

    Done lazily: importing ``models.roberta`` pulls in torch and transformers, which costs
    seconds. A caller that only needs the TF-IDF control should not pay that.
    """
    import models.baselines
    import models.roberta  # noqa: F401
