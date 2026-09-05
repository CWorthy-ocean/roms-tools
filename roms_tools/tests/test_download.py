import logging
from urllib.error import HTTPError

import pytest

from roms_tools.datasets.download import MAX_DOWNLOAD_ATTEMPTS, _download_one, _fetch


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


# ---------------------------------------------------------------------------
# _download_one: the urllib-based streaming downloader used by download_woa23_bgc()
# ---------------------------------------------------------------------------


class FakeResponse:
    """Stand-in for the object ``urlopen(...)`` returns, used as a context manager."""

    def __init__(self, chunks, content_length=None):
        """Queue ``chunks`` to be returned one per ``.read()`` call, then EOF."""
        # A trailing b"" ends the `while chunk := response.read(...)` loop, exactly
        # like a real socket at EOF.
        self._chunks = [*chunks, b""]
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self, size):
        return self._chunks.pop(0)


def _http_error(code, url="https://example.com/file.nc"):
    return HTTPError(url, code, f"HTTP {code}", {}, None)


@pytest.fixture(autouse=True)
def _no_sleep_for_download_one(monkeypatch):
    """Skip backoff delays for `_download_one`'s tests too."""
    monkeypatch.setattr("roms_tools.datasets.download.time.sleep", lambda seconds: None)


def test_download_one_succeeds_first_try(tmp_path, monkeypatch):
    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append((url, timeout))
        return FakeResponse([b"abcd"], content_length=4)

    monkeypatch.setattr("roms_tools.datasets.download.urlopen", fake_urlopen)

    target = tmp_path / "file.nc"
    _download_one("https://example.com/file.nc", target)

    assert target.read_bytes() == b"abcd"
    assert calls == [("https://example.com/file.nc", 60)]
    assert list(tmp_path.glob("*.part")) == []


def test_download_one_does_not_retry_4xx(tmp_path, monkeypatch):
    """A 4xx is a permanent failure: it must not be retried, and must name the
    pre-staging hint rather than leaving the caller to guess.

    The original `HTTPError` is re-raised (not wrapped in a plain `OSError`), so
    a caller can still branch on `error.code`; the hint is attached as a note
    rather than folded into the exception's `str()`.
    """
    calls = {"n": 0}

    def fake_urlopen(url, timeout=None):
        calls["n"] += 1
        raise _http_error(404)

    monkeypatch.setattr("roms_tools.datasets.download.urlopen", fake_urlopen)

    target = tmp_path / "file.nc"
    with pytest.raises(HTTPError) as excinfo:
        _download_one("https://example.com/file.nc", target)

    assert excinfo.value.code == 404
    assert any("client error" in note for note in excinfo.value.__notes__)
    assert calls["n"] == 1, "a 4xx must not be retried"
    assert not target.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_download_one_retries_5xx_then_succeeds(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(url, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(503)
        return FakeResponse([b"ok"], content_length=2)

    monkeypatch.setattr("roms_tools.datasets.download.urlopen", fake_urlopen)

    target = tmp_path / "file.nc"
    _download_one("https://example.com/file.nc", target)

    assert calls["n"] == 2
    assert target.read_bytes() == b"ok"
    assert list(tmp_path.glob("*.part")) == []


def test_download_one_retries_content_length_mismatch(tmp_path, monkeypatch):
    """A truncated read (Content-Length disagrees with bytes written) is treated
    like any other transient failure: retried, not raised immediately.
    """
    calls = {"n": 0}

    def fake_urlopen(url, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse([b"short"], content_length=100)
        return FakeResponse([b"ok"], content_length=2)

    monkeypatch.setattr("roms_tools.datasets.download.urlopen", fake_urlopen)

    target = tmp_path / "file.nc"
    _download_one("https://example.com/file.nc", target)

    assert calls["n"] == 2
    assert target.read_bytes() == b"ok"
    assert list(tmp_path.glob("*.part")) == []


def test_download_one_no_part_file_left_after_exhausting_retries(tmp_path, monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise TimeoutError("still down")

    monkeypatch.setattr("roms_tools.datasets.download.urlopen", fake_urlopen)

    target = tmp_path / "file.nc"
    with pytest.raises(TimeoutError):
        _download_one("https://example.com/file.nc", target)

    assert not target.exists()
    assert list(tmp_path.glob("*.part")) == []
