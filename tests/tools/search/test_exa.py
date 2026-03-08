from __future__ import annotations

import pytest

from miniprophet.exceptions import SearchNetworkError
from miniprophet.tools.search.exa import ExaSearchBackend


def test_exa_date_mmddyyyy_to_iso_start_and_end() -> None:
    assert (
        ExaSearchBackend._date_mmddyyyy_to_iso("01/02/2026", end_of_day=False)
        == "2026-01-02T00:00:00Z"
    )
    assert (
        ExaSearchBackend._date_mmddyyyy_to_iso("01/02/2026", end_of_day=True)
        == "2026-01-02T23:59:59Z"
    )


def test_exa_date_mmddyyyy_to_iso_rejects_invalid_date() -> None:
    with pytest.raises(SearchNetworkError, match="Expected MM/DD/YYYY"):
        ExaSearchBackend._date_mmddyyyy_to_iso("2026-01-02", end_of_day=False)
