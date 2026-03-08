from __future__ import annotations

import pytest
import requests

from miniprophet.run.services.polymarket import PolymarketService


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


def test_polymarket_fetch_event_by_id_parses_market_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_urls: list[str] = []
    payload = {
        "id": "4242",
        "slug": "who-wins-2028",
        "title": "Who wins in 2028?",
        "markets": [
            {
                "id": "m1",
                "question": "Candidate A",
                "outcomes": '["Yes","No"]',
                "outcomePrices": '["1","0"]',
                "closed": True,
                "umaResolutionStatus": "resolved",
            },
            {
                "id": "m2",
                "question": "Candidate B",
                "outcomes": '["Yes","No"]',
                "outcomePrices": '["0","1"]',
                "closed": True,
                "umaResolutionStatus": "resolved",
            },
        ],
    }

    def _fake_get(url: str, *args, **kwargs) -> _Resp:
        called_urls.append(url)
        return _Resp(payload)

    monkeypatch.setattr("miniprophet.run.services.polymarket.requests.get", _fake_get)

    result = PolymarketService().fetch("4242", entity="event", identifier_type="id")

    assert called_urls == ["https://gamma-api.polymarket.com/events/4242"]
    assert result.title == "Who wins in 2028?"
    assert result.outcomes == ["Candidate A", "Candidate B"]
    assert result.ground_truth == {"Candidate A": 1, "Candidate B": 0}
    assert result.metadata["event_id"] == "4242"


def test_polymarket_fetch_market_by_slug_parses_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_urls: list[str] = []
    payload = {
        "id": "m1",
        "slug": "will-fed-cut-rates-by-june",
        "question": "Will the Fed cut rates by June?",
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["1","0"]',
        "closed": True,
        "umaResolutionStatus": "resolved",
        "volumeNum": 12000.5,
        "liquidityNum": 800.1,
    }

    def _fake_get(url: str, *args, **kwargs) -> _Resp:
        called_urls.append(url)
        return _Resp(payload)

    monkeypatch.setattr("miniprophet.run.services.polymarket.requests.get", _fake_get)

    result = PolymarketService().fetch(
        "will-fed-cut-rates-by-june",
        entity="market",
        identifier_type="slug",
    )

    assert called_urls == [
        "https://gamma-api.polymarket.com/markets/slug/will-fed-cut-rates-by-june"
    ]
    assert result.title == "Will the Fed cut rates by June?"
    assert result.outcomes == ["Yes", "No"]
    assert result.ground_truth == {"Yes": 1, "No": 0}
    assert result.metadata["market_slug"] == "will-fed-cut-rates-by-june"


def test_polymarket_fetch_auto_falls_back_to_market(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_urls: list[str] = []
    market_payload = {
        "id": "m1",
        "slug": "will-fed-cut-rates-by-june",
        "question": "Will the Fed cut rates by June?",
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["0.7","0.3"]',
    }

    def _fake_get(url: str, *args, **kwargs) -> _Resp | _ErrorResp:
        called_urls.append(url)
        if "/events/slug/" in url:
            return _ErrorResp(404)
        return _Resp(market_payload)

    monkeypatch.setattr("miniprophet.run.services.polymarket.requests.get", _fake_get)

    result = PolymarketService().fetch("will-fed-cut-rates-by-june", identifier_type="slug")

    assert called_urls == [
        "https://gamma-api.polymarket.com/events/slug/will-fed-cut-rates-by-june",
        "https://gamma-api.polymarket.com/markets/slug/will-fed-cut-rates-by-june",
    ]
    assert result.outcomes == ["Yes", "No"]
    assert result.metadata["entity"] == "market"
