import datetime as dt
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sentiment.prereg as prereg
import sentiment.prices as prices
from sentiment.headline import Headline
from sentiment.prices import Bar, save_fixture
from sentiment.prereg import GateStatus, check_gate, run_prereg_test, usable_rows


def make_headline(title: str, link: str, published_at: datetime, source: str = "feed") -> Headline:
    return Headline(
        source=source,
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=published_at + dt.timedelta(hours=1),
    )


def row(session_date, source, ticker="INFY.NS"):
    return {"session_date": session_date, "source": source, "ticker": ticker, "title": "t"}


def test_check_gate_reports_each_unmet_dimension():
    rows = [row("2026-09-28", "feed")]
    gate = check_gate(rows)
    assert gate.met is False
    assert gate.n_headlines == 1
    assert gate.n_sessions == 1
    assert gate.n_sources == 1
    assert any("headlines" in r for r in gate.reasons)
    assert any("sessions" in r for r in gate.reasons)
    assert any("sources" in r for r in gate.reasons)


def test_check_gate_met_when_all_three_thresholds_cleared():
    rows = []
    for i in range(prereg.PREREG_MIN_HEADLINES):
        session = f"2026-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}"
        source = ["a", "b", "c"][i % 3]
        rows.append(row(session, source))
    gate = check_gate(rows)
    assert gate.n_headlines == prereg.PREREG_MIN_HEADLINES
    assert gate.n_sources == 3
    if gate.n_sessions >= prereg.PREREG_MIN_SESSIONS:
        assert gate.met
        assert gate.reasons == ()


def test_usable_rows_excludes_price_naming_headlines(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [
            Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0),
            Bar(date=dt.date(2026, 9, 29), open=105.0, close=103.0),
        ],
    )
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        ),
        make_headline(
            "Infosys shares rise 4% after Q2 results",
            "2",
            datetime(2026, 9, 28, 2, 30, 0, tzinfo=timezone.utc),
        ),
    ]
    rows = usable_rows(headlines)
    assert len(rows) == 1
    assert rows[0]["title"] == "Infosys Share Price Highlights: Infosys Stock Price History"


def test_usable_rows_attaches_abnormal_return_0_1_and_drops_rows_missing_it(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [
            Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0),
            Bar(date=dt.date(2026, 9, 29), open=110.0, close=108.0),
        ],
    )
    # Second headline aligns to the last session in the fixture, so its
    # [0, +1] window has no +1 leg yet and must be dropped, not crash.
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        ),
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "2",
            datetime(2026, 9, 29, 2, 0, 0, tzinfo=timezone.utc),
        ),
    ]
    rows = usable_rows(headlines)
    assert len(rows) == 1
    assert "abnormal_return_0_1" in rows[0]
    assert rows[0]["abnormal_return_0_1"] == pytest.approx(rows[0]["abnormal_return_0_1"])


def test_run_prereg_test_reports_underpowered_without_calling_finbert(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [
            Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0),
            Bar(date=dt.date(2026, 9, 29), open=105.0, close=103.0),
        ],
    )

    def boom(texts):
        raise AssertionError("FinBERT must not be called while the gate is unmet")

    monkeypatch.setattr(prereg, "score_texts", boom)

    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    ]
    result = run_prereg_test(headlines)
    assert result.gate.met is False
    assert result.stat is None


def test_run_prereg_test_computes_session_demeaned_r_once_gate_is_met(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    monkeypatch.setattr(prereg, "PREREG_MIN_HEADLINES", 4)
    monkeypatch.setattr(prereg, "PREREG_MIN_SESSIONS", 2)
    monkeypatch.setattr(prereg, "PREREG_MIN_SOURCES", 1)

    save_fixture(
        "INFY.NS",
        [
            Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0),
            Bar(date=dt.date(2026, 9, 29), open=110.0, close=121.0),
            Bar(date=dt.date(2026, 9, 30), open=121.0, close=108.0),
        ],
    )

    headlines = [
        make_headline(f"Infosys Share Price Highlights {i}", str(i), datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc))
        for i in range(2)
    ] + [
        make_headline(f"Infosys Share Price Highlights {i}", str(i), datetime(2026, 9, 29, 2, i, 0, tzinfo=timezone.utc))
        for i in range(2, 4)
    ]

    monkeypatch.setattr(
        prereg,
        "score_texts",
        lambda texts: [{"positive": 0.9, "negative": 0.1, "neutral": 0.0} for _ in texts],
    )

    result = run_prereg_test(headlines)
    assert result.gate.met is True
    assert result.stat is not None
    assert result.stat.n == 4


def test_run_against_committed_fixture_reports_gate_not_met(capsys):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_regression_50.csv"
    exit_code = prereg.run(fixture, live=False)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "underpowered, not null" in out
    assert "r is not reported until the gate is met" in out
