import logging

import pytest

from roms_tools.datasets.download import MAX_DOWNLOAD_ATTEMPTS, _fetch


class FakePooch:
    """Stand-in for ``pooch.Pooch`` that fails a fixed number of times."""

    def __init__(self, failures, error=None):
        """Fail the first ``failures`` calls with ``error``, then succeed."""
        self.failures = failures
        self.error = error or TimeoutError("read timed out")
        self.calls = 0

    def fetch(self, filename):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return f"/cache/{filename}"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Skip the backoff delays so the tests stay fast."""
    monkeypatch.setattr("roms_tools.datasets.download.time.sleep", lambda seconds: None)


def test_fetch_returns_immediately_when_download_succeeds():
    manager = FakePooch(failures=0)

    assert _fetch(manager, "etopo5.nc") == "/cache/etopo5.nc"
    assert manager.calls == 1


@pytest.mark.parametrize("failures", range(1, MAX_DOWNLOAD_ATTEMPTS))
def test_fetch_retries_transient_network_errors(failures, caplog):
    manager = FakePooch(failures=failures)

    with caplog.at_level(logging.WARNING):
        assert _fetch(manager, "etopo5.nc") == "/cache/etopo5.nc"

    assert manager.calls == failures + 1
    assert len(caplog.records) == failures


def test_fetch_retries_the_error_seen_in_ci():
    """`requests` errors subclass `OSError`, which is what the retry catches."""
    # Imported here rather than at module scope: requests is only present as a
    # transitive dependency of pooch, and roms-tools does not import it.
    import requests

    manager = FakePooch(
        failures=1, error=requests.exceptions.ReadTimeout("read timed out")
    )

    assert _fetch(manager, "etopo5.nc") == "/cache/etopo5.nc"
    assert manager.calls == 2


def test_fetch_raises_after_exhausting_attempts():
    manager = FakePooch(failures=MAX_DOWNLOAD_ATTEMPTS)

    with pytest.raises(TimeoutError):
        _fetch(manager, "etopo5.nc")

    assert manager.calls == MAX_DOWNLOAD_ATTEMPTS


def test_fetch_does_not_retry_checksum_mismatch():
    manager = FakePooch(failures=1, error=ValueError("hash mismatch"))

    with pytest.raises(ValueError):
        _fetch(manager, "etopo5.nc")

    assert manager.calls == 1
