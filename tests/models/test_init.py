from __future__ import annotations

import types

import pytest

import miniprophet.models as models_module
from miniprophet.models import GlobalModelStats, get_model


class _DummyModelClass:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


@pytest.fixture
def patch_dummy_model_class(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("_tests_dummy_model_module")
    mod.DummyModel = _DummyModelClass

    import sys

    sys.modules[mod.__name__] = mod
    monkeypatch.setitem(models_module._MODEL_CLASS_MAPPING, "dummy", f"{mod.__name__}.DummyModel")


def test_get_model_uses_mapping_and_kwargs(patch_dummy_model_class) -> None:
    model = get_model({"model_class": "dummy", "model_name": "x", "temperature": 0.2})
    assert isinstance(model, _DummyModelClass)
    assert model.kwargs["model_name"] == "x"
    assert model.kwargs["temperature"] == 0.2


def test_get_model_requires_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINIPROPHET_MODEL_NAME", raising=False)
    with pytest.raises(ValueError, match="No model name set"):
        get_model({"model_class": "dummy"})


def test_global_model_stats_add_tokens() -> None:
    stats = GlobalModelStats()
    stats.add_tokens(1000, 500)
    stats.add_tokens(2000, 1500)
    assert stats.total_prompt_tokens == 3000
    assert stats.total_cached_tokens == 2000
    assert stats.cache_hit_rate == pytest.approx(2000 / 3000)


def test_global_model_stats_add_tokens_none_cached() -> None:
    stats = GlobalModelStats()
    stats.add_tokens(1000, None)
    assert stats.total_prompt_tokens == 1000
    assert stats.total_cached_tokens == 0
    assert stats.cache_hit_rate is None


def test_global_model_stats_add_and_limit() -> None:
    stats = GlobalModelStats()
    stats.cost_limit = 0.05
    stats.add(0.03)
    assert stats.cost == pytest.approx(0.03)
    assert stats.n_calls == 1
    with pytest.raises(RuntimeError, match="Global model cost limit exceeded"):
        stats.add(0.03)
