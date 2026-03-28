"""Search backend interface and factory for mini-prophet."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Protocol

from miniprophet.environment.source_registry import Source


@dataclass
class SearchResult:
    """Return type for SearchBackend.search()."""

    sources: list[Source]
    cost: float = 0.0


class SearchBackend(Protocol):
    """Protocol that any search backend must satisfy."""

    async def search(self, query: str, limit: int = 5, **kwargs) -> SearchResult: ...

    def serialize(self) -> dict: ...


_SEARCH_CLASS_MAPPING: dict[str, str] = {
    "brave": "miniprophet.tools.search.brave.BraveSearchBackend",
    "perplexity": "miniprophet.tools.search.perplexity.PerplexitySearchBackend",
    "exa": "miniprophet.tools.search.exa.ExaSearchBackend",
    "tavily": "miniprophet.tools.search.tavily.TavilySearchBackend",
}


def get_search_backend(search_cfg: dict) -> SearchBackend:
    """Instantiate a search backend from a config dict.

    The 'search_class' key selects the implementation (default: "perplexity").
    Remaining keys are forwarded as keyword arguments to the constructor.
    """
    search_cls = search_cfg.get("search_class", "perplexity")
    search_cls_config = search_cfg.get(search_cls, {})
    full_path = _SEARCH_CLASS_MAPPING.get(search_cls, search_cls)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ValueError(
            f"Unknown search class: {search_cls} (resolved to {full_path}, "
            f"available: {list(_SEARCH_CLASS_MAPPING)})"
        ) from exc
    sig = inspect.signature(cls.__init__)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        accepted = search_cls_config
    else:
        valid_keys = set(sig.parameters.keys()) - {"self"}
        accepted = {k: v for k, v in search_cls_config.items() if k in valid_keys}
    return cls(**accepted)


__all__ = ["SearchBackend", "SearchResult", "get_search_backend", "Source"]
