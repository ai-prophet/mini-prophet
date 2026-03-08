from __future__ import annotations

import pytest

from miniprophet.run.services import get_market_service
from miniprophet.run.services.kalshi import KalshiService
from miniprophet.run.services.polymarket import PolymarketService


def test_get_market_service_returns_kalshi() -> None:
    svc = get_market_service("kalshi")
    assert isinstance(svc, KalshiService)


def test_get_market_service_returns_polymarket() -> None:
    svc = get_market_service("polymarket")
    assert isinstance(svc, PolymarketService)


def test_get_market_service_raises_for_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown market service"):
        get_market_service("missing-service")
