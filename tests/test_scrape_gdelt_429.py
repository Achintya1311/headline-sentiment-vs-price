"""P160, P110: the GDELT step must keep what it fetched, and the offline
default must not write into the committed corpus.

On 3 and 4 Nov 2026 the first company that fetched ``ok`` was re-requested
at once to write its snapshot; that second request got HTTP 429, the
uncaught error ended the run before ``merge_and_write``, and the parsed
headlines were lost.
"""

import json
import urllib.error
from pathlib import Path

import pytest

from sentiment import scrape_gdelt
from sentiment.headline import read_csv

SAMPLE = Path(__file__).resolve().parent.parent / "fixtures" / "gdelt" / "infosys.json"


class FakeResp:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.body


@pytest.fixture
def one_ok_then_429(monkeypatch, tmp_path):
    """robots allowed, no waiting; the first urlopen returns the real sample,
    every later one raises 429."""
    from sentiment.robots import RateLimiter, RobotsCache

    monkeypatch.setattr(RobotsCache, "can_fetch", lambda self, url: True)
    monkeypatch.setattr(RobotsCache, "crawl_delay", lambda self, url: 0.0)
    monkeypatch.setattr(RateLimiter, "wait", lambda self, host, extra_delay=0.0: None)
    monkeypatch.setattr(scrape_gdelt, "FIXTURES_DIR", tmp_path / "gdelt")
    calls = []

    def urlopen(req, timeout=None):
        calls.append(getattr(req, "full_url", req))
        if len(calls) == 1:
            return FakeResp(SAMPLE.read_bytes())
        raise urllib.error.HTTPError(calls[-1], 429, "Too Many Requests", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return calls


def test_a_429_after_one_ok_keeps_that_companys_headlines(one_ok_then_429, tmp_path):
    out = tmp_path / "h.csv"
    code = scrape_gdelt.run(["Infosys", "Wipro"], live=True, out=out)
    assert code == 1  # Wipro failed, so the step still reports a failure
    assert len(read_csv(out)) > 0  # but Infosys's headlines were merged


def test_one_request_per_company_and_the_snapshot_is_the_fetched_bytes(one_ok_then_429, tmp_path):
    scrape_gdelt.run(["Infosys"], live=True, out=tmp_path / "h.csv")
    assert len(one_ok_then_429) == 1
    snap = tmp_path / "gdelt" / "infosys.json"
    assert json.loads(snap.read_bytes()) == json.loads(SAMPLE.read_bytes())


def test_a_snapshot_write_failure_does_not_lose_headlines(one_ok_then_429, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "write_bytes", lambda self, data: (_ for _ in ()).throw(OSError("disk full")))
    out = tmp_path / "h.csv"
    scrape_gdelt.run(["Infosys"], live=True, out=out)
    assert len(read_csv(out)) > 0


def test_offline_run_refuses_the_committed_corpus(capsys):
    before = scrape_gdelt.DEFAULT_OUT.read_bytes()
    code = scrape_gdelt.run(["Infosys"], live=False, out=scrape_gdelt.DEFAULT_OUT)
    assert code == 2
    assert scrape_gdelt.DEFAULT_OUT.read_bytes() == before
    assert "committed corpus" in capsys.readouterr().err


def test_offline_cli_default_writes_a_scratch_file(monkeypatch, tmp_path):
    scratch = tmp_path / "offline.csv"
    monkeypatch.setattr(scrape_gdelt, "OFFLINE_OUT", scratch)
    before = scrape_gdelt.DEFAULT_OUT.read_bytes()
    monkeypatch.setattr("sys.argv", ["scrape_gdelt", "--company", "Infosys"])
    with pytest.raises(SystemExit) as e:
        scrape_gdelt.main()
    assert e.value.code == 0
    assert scrape_gdelt.DEFAULT_OUT.read_bytes() == before
    assert len(read_csv(scratch)) > 0
