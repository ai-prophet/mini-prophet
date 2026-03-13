"""Tests for miniprophet.eval.agent_factory module."""

from __future__ import annotations

import pytest

from miniprophet.eval.agent_factory import EvalAgentFactory


class TestImportAgentClass:
    def test_missing_colon_raises(self) -> None:
        with pytest.raises(ValueError, match="module.path:ClassName"):
            EvalAgentFactory._import_agent_class("no.colon.here")

    def test_nonexistent_module_raises(self) -> None:
        with pytest.raises(ValueError, match="Failed to import"):
            EvalAgentFactory._import_agent_class("nonexistent.module:Foo")

    def test_missing_class_raises(self) -> None:
        with pytest.raises(ValueError, match="has no class"):
            EvalAgentFactory._import_agent_class("miniprophet.exceptions:NonExistentClass")

    def test_valid_import_succeeds(self) -> None:
        cls = EvalAgentFactory._import_agent_class("miniprophet.exceptions:SearchError")
        from miniprophet.exceptions import SearchError

        assert cls is SearchError


class TestResolveAgentClass:
    def test_agent_class_takes_precedence(self) -> None:
        class MyAgent:
            pass

        result = EvalAgentFactory._resolve_agent_class(
            agent_name=None, agent_import_path=None, agent_class=MyAgent
        )
        assert result is MyAgent

    def test_import_path_used_when_no_class(self) -> None:
        result = EvalAgentFactory._resolve_agent_class(
            agent_name=None,
            agent_import_path="miniprophet.exceptions:SearchError",
            agent_class=None,
        )
        from miniprophet.exceptions import SearchError

        assert result is SearchError

    def test_default_agent_resolved(self) -> None:
        result = EvalAgentFactory._resolve_agent_class(
            agent_name="default", agent_import_path=None, agent_class=None
        )
        from miniprophet.agent.default import DefaultForecastAgent

        assert result is DefaultForecastAgent

    def test_unknown_agent_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown built-in"):
            EvalAgentFactory._resolve_agent_class(
                agent_name="nonexistent", agent_import_path=None, agent_class=None
            )


class TestCreate:
    def test_incompatible_constructor_raises(self) -> None:
        class BadAgent:
            def __init__(self):
                pass  # No model/env params

        with pytest.raises(ValueError, match="incompatible"):
            EvalAgentFactory.create(
                model=object(),
                env=object(),
                context_manager=None,
                agent_class=BadAgent,
                agent_kwargs={},
                task_id="t1",
            )

    def test_context_manager_injected_when_accepted(self) -> None:
        received = {}

        class GoodAgent:
            def __init__(self, model, env, context_manager=None):
                received["cm"] = context_manager

        sentinel = object()
        EvalAgentFactory.create(
            model=object(),
            env=object(),
            context_manager=sentinel,
            agent_class=GoodAgent,
            agent_kwargs={},
            task_id="t1",
        )
        assert received["cm"] is sentinel
