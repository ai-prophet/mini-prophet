from __future__ import annotations

import pytest

from miniprophet.run.services import get_market_service
from miniprophet.run.services.kalshi import KalshiService
from miniprophet.run.services.polymarket import PolymarketService


@pytest.mark.parametrize(
    "name,expected_type",
    [("kalshi", KalshiService), ("polymarket", PolymarketService)],
)
def test_get_market_service_returns_expected(name: str, expected_type: type) -> None:
    assert isinstance(get_market_service(name), expected_type)


def test_get_market_service_raises_for_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown market service"):
        get_market_service("missing-service")
