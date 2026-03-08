from __future__ import annotations

import pytest
import requests

from miniprophet.run.services.kalshi import KalshiService


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _ErrorResp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        raise requests.HTTPError(f"status={self.status_code}", response=self)

    def json(self) -> dict:
        return {}


def test_kalshi_fetch_event_parses_outcomes_and_unresolved_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "event": {"title": "Election", "event_ticker": "EVT"},
        "markets": [
            {"yes_sub_title": "A", "result": "yes", "ticker": "EVT-A", "status": "active"},
            {
                "yes_sub_title": "B",
                "result": None,
                "ticker": "EVT-B",
                "status": "active",
                "last_price_dollars": 0.42,
                "volume": 123,
            },
        ],
    }
    monkeypatch.setattr(
        "miniprophet.run.services.kalshi.requests.get", lambda *a, **k: _Resp(payload)
    )

    result = KalshiService().fetch("evt")

    assert result.title == "Election"
    assert result.outcomes == ["A", "B"]
    assert result.ground_truth is None
    assert result.metadata["event_ticker"] == "EVT"
    assert result.metadata["market_count"] == 2
    assert result.metadata["volume"] == pytest.approx(123.0)


def test_kalshi_fetch_market_returns_yes_no_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "market": {
            "title": "Will it rain in NYC tomorrow?",
            "result": "no",
            "ticker": "RAIN-NYC-20260305",
            "event_ticker": "WXRAIN",
            "status": "settled",
            "last_price_dollars": 0.0,
            "volume": 150,
        }
    }
    monkeypatch.setattr(
        "miniprophet.run.services.kalshi.requests.get", lambda *a, **k: _Resp(payload)
    )

    result = KalshiService().fetch("rain-nyc-20260305", ticker_type="market")

    assert result.title == "Will it rain in NYC tomorrow?"
    assert result.outcomes == ["Yes", "No"]
    assert result.ground_truth == {"Yes": 0, "No": 1}
    assert result.metadata["market_ticker"] == "RAIN-NYC-20260305"
    assert result.metadata["event_ticker"] == "WXRAIN"


def test_kalshi_fetch_auto_falls_back_to_market(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_urls: list[str] = []
    market_payload = {
        "market": {
            "title": "Will it rain in NYC tomorrow?",
            "result": "yes",
            "ticker": "RAIN-NYC-20260305",
            "event_ticker": "WXRAIN",
        }
    }

    def _fake_get(url: str, *args, **kwargs) -> _Resp | _ErrorResp:
        called_urls.append(url)
        if "/events/" in url:
            return _ErrorResp(404)
        return _Resp(market_payload)

    monkeypatch.setattr("miniprophet.run.services.kalshi.requests.get", _fake_get)

    result = KalshiService().fetch("rain-nyc-20260305")

    assert "/events/RAIN-NYC-20260305" in called_urls[0]
    assert "/markets/RAIN-NYC-20260305" in called_urls[1]
    assert result.outcomes == ["Yes", "No"]
    assert result.metadata["entity"] == "market"
