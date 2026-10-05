from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path

import pytest

import sentiment.correlate as correlate
import sentiment.prices as prices
from sentiment.audit import run, run_audit, shuffle_timestamps
from sentiment.correlate import DEFAULT_IN
from sentiment.headline import Headline, read_csv, write_csv
from sentiment.market_hours import is_trading_day, next_trading_day
from sentiment.prices import Bar, save_fixture


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


@dataclass(frozen=True)
class FakeScored:
    compound: float


def fake_score_headline(compound_by_link: dict[str, float]):
    """A stand-in for ``sentiment.vader_score.score_headline`` that returns a
    chosen compound per headline instead of actually running VADER - the
    synthetic tests below need exact, known compound values to construct a
    correlation (or its absence) by hand, which real lexicon scoring can't
    give directly."""

    def fake(headline: Headline) -> FakeScored:
        return FakeScored(compound=compound_by_link[headline.link])

    return fake


def n_trading_days(start: dt.date, n: int) -> list[dt.date]:
    """``n`` consecutive NSE trading days starting at or after ``start`` -
    computed with the module's own calendar logic rather than a hand-picked
    date, so the test doesn't depend on knowing 2026's weekday layout."""
    first = start if is_trading_day(start) else next_trading_day(start - dt.timedelta(days=1))
    days = [first]
    while len(days) < n:
        days.append(next_trading_day(days[-1]))
    return days


# Six distinct single-company headline shapes sentiment.tickers.resolve
# already knows (see sentiment/tickers.py) - real patterns, not invented
# ones, so these tests exercise the same resolution path production does.
TICKERS = [
    ("SBI Life Share Price Highlights: SBI Life Stock Price History", "SBILIFE.NS"),
    ("Nestle India Share Price Highlights: Nestle India Stock Price History", "NESTLEIND.NS"),
    ("Sun Pharma Share Price Highlights: Sun Pharma Stock Price History", "SUNPHARMA.NS"),
    ("Grasim Inds Share Price Highlights: Grasim Inds Stock Price History", "GRASIM.NS"),
    ("Tech Mahindra Share Price Highlights: Tech Mahindra Stock Price History", "TECHM.NS"),
    ("Wipro Share Price Highlights: Wipro Stock Price History", "WIPRO.NS"),
]
COMPOUNDS = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]


def build_synthetic_scenario(tmp_path: Path, monkeypatch, *, leaky: bool) -> list[Headline]:
    """Six headlines, one per ticker, each published pre-open on its own
    distinct trading day, with a compound score Day 5's actual VADER scorer
    never has to produce (monkeypatched instead - see ``fake_score_headline``).

    Each ticker's price bars are built so that ``contemporaneous_return``
    equals that headline's own compound *exactly* when the headline lands on
    its own, correctly aligned session - a perfect r=1.0 relationship by
    construction, real alignment only.

    ``leaky=False`` (the honest case): every *other* day in the window has a
    flat 0.0 return for that ticker, so a shuffled timestamp (which reuses
    another headline's original published_at, landing on a different day)
    sees a return unrelated to its own compound - the correlation should
    collapse once timestamps no longer honestly pick out "each headline's
    own day".

    ``leaky=True`` (the bug this check exists to catch): every day in the
    window has the *same* return for that ticker (compound_i, flat across
    the whole window) - so which day a shuffled timestamp lands on makes no
    difference at all, and the correlation survives shuffling undiminished.
    This simulates a pipeline that doesn't actually depend on correct
    timestamp alignment, exactly the artifact NEXT_STEPS.md's "Done when"
    warns about.
    """
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    days = n_trading_days(dt.date(2026, 9, 7), len(TICKERS))

    compound_by_link: dict[str, float] = {}
    headlines: list[Headline] = []
    for i, ((title, ticker), compound) in enumerate(zip(TICKERS, COMPOUNDS)):
        link = str(i)
        compound_by_link[link] = compound
        published_at = datetime.combine(days[i], time(2, 0), tzinfo=timezone.utc)  # pre-open IST
        headlines.append(make_headline(title, link, published_at))

        bars = []
        for d in days:
            if leaky or d == days[i]:
                close = 100.0 * (1 + compound)
            else:
                close = 100.0
            bars.append(Bar(date=d, open=100.0, close=close))
        save_fixture(ticker, bars)

    monkeypatch.setattr(correlate, "score_headline", fake_score_headline(compound_by_link))
    return headlines


# --- shuffle_timestamps -------------------------------------------------


def test_shuffle_timestamps_preserves_the_multiset_of_timestamps():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 7 + i, 2, 0, tzinfo=timezone.utc))
        for i in range(6)
    ]
    shuffled = shuffle_timestamps(headlines, seed=1)

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # titles/links untouched - only published_at moves between headlines
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]


def test_shuffle_timestamps_is_deterministic_for_a_given_seed():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 7 + i, 2, 0, tzinfo=timezone.utc))
        for i in range(6)
    ]
    first = shuffle_timestamps(headlines, seed=42)
    second = shuffle_timestamps(headlines, seed=42)
    assert [h.published_at for h in first] == [h.published_at for h in second]


# --- run_audit ------------------------------------------------------------


def test_run_audit_raises_when_too_few_resolved_headlines(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0)])
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    ]

    with pytest.raises(ValueError, match="not enough resolved headlines"):
        run_audit(headlines, n_shuffles=10, seed=0)


def test_run_audit_true_signal_collapses_under_shuffled_timestamps(tmp_path: Path, monkeypatch):
    headlines = build_synthetic_scenario(tmp_path, monkeypatch, leaky=False)

    result = run_audit(headlines, n_shuffles=300, seed=0)

    assert result.r_real == pytest.approx(1.0, abs=1e-9)
    assert result.has_real_signal
    # With only 6 headlines, a uniform random permutation has 1 fixed point
    # on average (a standard derangement-theory result, independent of n) -
    # so shuffling doesn't drive the correlation all the way to 0, but it
    # should still land comfortably under the 50% retention this check
    # treats as "did not disappear".
    assert result.retained_fraction < 0.45
    assert result.leak_suspected is False


def test_run_audit_catches_a_timestamp_blind_leak(tmp_path: Path, monkeypatch):
    headlines = build_synthetic_scenario(tmp_path, monkeypatch, leaky=True)

    result = run_audit(headlines, n_shuffles=300, seed=0)

    assert result.r_real == pytest.approx(1.0, abs=1e-9)
    # every ticker's return is flat across the whole window, so no matter
    # which day a shuffled timestamp lands on, the correlation is unchanged.
    assert result.retained_fraction == pytest.approx(1.0, abs=1e-9)
    assert result.leak_suspected is True


# --- against the real committed fixture -----------------------------------


def test_audit_against_the_real_fixture_does_not_make_the_signal_fully_disappear():
    """Honest, documented result (see README Findings): on this repo's own
    23-headline fixture, the shuffle control retains roughly half of the
    real correlation's magnitude rather than collapsing it - not because of
    a software leak, but because all 23 headlines' session dates fall on
    just two trading days (28/29 Sep) with one headline per ticker, so a
    shuffle only ever toggles a headline between its *own* ticker's two
    already price-correlated-with-compound sessions. This test pins that
    finding so a future change that quietly fixes or worsens it gets
    noticed, rather than asserting a clean PASS that isn't true today."""
    headlines = read_csv(DEFAULT_IN)

    result = run_audit(headlines, n_shuffles=1000, seed=0)

    assert result.n_real == 23
    assert result.r_real == pytest.approx(-0.185, abs=0.01)
    assert result.has_real_signal
    assert 0.4 < result.retained_fraction < 0.65
    assert result.leak_suspected is True


# --- CLI (run) --------------------------------------------------------------


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    assert run(missing, n_shuffles=10, seed=0, live=False) == 1


def test_run_passes_for_a_genuine_timestamp_dependent_signal(tmp_path: Path, monkeypatch):
    headlines = build_synthetic_scenario(tmp_path, monkeypatch, leaky=False)
    in_path = tmp_path / "raw.csv"
    write_csv(headlines, in_path)

    assert run(in_path, n_shuffles=300, seed=0, live=False) == 0


def test_run_fails_for_a_timestamp_blind_leak(tmp_path: Path, monkeypatch):
    headlines = build_synthetic_scenario(tmp_path, monkeypatch, leaky=True)
    in_path = tmp_path / "raw.csv"
    write_csv(headlines, in_path)

    assert run(in_path, n_shuffles=300, seed=0, live=False) == 1
