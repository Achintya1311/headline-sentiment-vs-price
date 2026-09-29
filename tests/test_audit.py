from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.audit import contemporaneous_r, run, run_audit, shuffle_timestamps
from sentiment.headline import Headline, read_csv
from sentiment.prices import Bar, save_fixture
from sentiment.vader_score import score_headline


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_shuffle_timestamps_is_a_permutation_of_the_same_timestamps():
    headlines = [
        make_headline(f"headline {i}", f"link-{i}", datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(6)
    ]

    shuffled = shuffle_timestamps(headlines, seed=1)

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_shuffle_timestamps_is_deterministic_for_a_given_seed():
    headlines = [
        make_headline(f"headline {i}", f"link-{i}", datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(6)
    ]

    a = shuffle_timestamps(headlines, seed=7)
    b = shuffle_timestamps(headlines, seed=7)

    assert [h.published_at for h in a] == [h.published_at for h in b]


def test_contemporaneous_r_needs_at_least_two_resolved_headlines(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=date(2026, 9, 28), open=100.0, close=105.0)])
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    ]

    r, n = contemporaneous_r(headlines)

    assert n == 1
    assert r == 0.0


# --- Synthetic-signal proof -------------------------------------------------
#
# The real fixture's own contemporaneous correlation is already ~0 (see
# README Findings), so running the audit against it alone can't show the
# permutation test has any power to catch a genuine leak - a test that always
# reports "no signal" would pass against that fixture too. This builds a
# small synthetic universe with a real, deterministic relationship between
# VADER compound and the return that follows it, and checks that (a) the
# audit correctly detects it against the real timestamps, and (b) shuffling
# timestamps knocks it down.
#
# Six distinct single-company headline shapes from sentiment/tickers.py, with
# hand-picked, strongly-worded suffixes to spread their VADER compound scores
# out instead of leaving them all pinned to the same value.
_TICKER_HEADLINES = [
    ("SBI Life Share Price Highlights: wonderful outstanding fantastic record profit surge", "SBILIFE.NS"),
    ("Nestle India Share Price Highlights: solid decent steady growth continues nicely", "NESTLEIND.NS"),
    ("Sun Pharma Share Price Highlights: mixed cautious muted update, nothing special today", "SUNPHARMA.NS"),
    ("Grasim Inds Share Price Highlights: disappointing weak dismal quarter overall reported", "GRASIM.NS"),
    ("Tech Mahindra Share Price Highlights: terrible disastrous horrific losses reported today", "TECHM.NS"),
    ("Wipro Share Price Highlights: brilliant superb excellent record earnings beat estimates", "WIPRO.NS"),
]

PRE_OPEN = datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)  # 07:30 IST -> session_date 2026-09-28
INTRADAY = datetime(2026, 9, 28, 6, 0, 0, tzinfo=timezone.utc)  # 11:30 IST -> rolls to 2026-09-29

DAY_A = date(2026, 9, 28)
DAY_B = date(2026, 9, 29)


def _build_synthetic_universe(tmp_path, monkeypatch, k: float = 0.5):
    """Headlines with a genuine, deterministic compound/return relationship
    under their own real timestamps, and a decoy relationship under the
    *other* alignable date - so a shuffle that moves a headline to the wrong
    session pairs it with an unrelated target value, not its own."""
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    headlines = []
    for i, (title, _ticker) in enumerate(_TICKER_HEADLINES):
        published_at = PRE_OPEN if i % 2 == 0 else INTRADAY
        headlines.append(make_headline(title, f"link-{i}", published_at))

    compounds = [score_headline(h).compound for h in headlines]
    n = len(headlines)

    for i, (_title, ticker) in enumerate(_TICKER_HEADLINES):
        correct_return = k * compounds[i]
        decoy_return = k * compounds[(i + 1) % n]
        natural_date = DAY_A if i % 2 == 0 else DAY_B
        other_date = DAY_B if i % 2 == 0 else DAY_A
        returns_by_date = {natural_date: correct_return, other_date: decoy_return}
        save_fixture(
            ticker,
            [
                Bar(date=DAY_A, open=100.0, close=100.0 * (1 + returns_by_date[DAY_A])),
                Bar(date=DAY_B, open=100.0, close=100.0 * (1 + returns_by_date[DAY_B])),
            ],
        )

    return headlines, compounds


def test_permutation_test_detects_a_real_injected_relationship(tmp_path, monkeypatch):
    headlines, _compounds = _build_synthetic_universe(tmp_path, monkeypatch)

    result = run_audit(headlines, trials=200, seed=0)

    # Under the real timestamps, every headline's return equals k * its own
    # compound exactly - a perfect linear relationship.
    assert result.observed_r == pytest.approx(1.0, abs=1e-6)
    assert result.observed_n == len(headlines)
    # Reassembling that exact pairing by randomly shuffling 6 timestamps is
    # rare - the real result should sit far in the tail of the shuffled
    # distribution, not blend into it.
    assert result.p_value < 0.05
    assert result.mean_abs_shuffled_r < 0.9


def test_permutation_test_shuffled_trials_are_lower_than_the_real_signal_on_average(tmp_path, monkeypatch):
    headlines, _compounds = _build_synthetic_universe(tmp_path, monkeypatch)

    result = run_audit(headlines, trials=200, seed=0)

    assert all(abs(r) <= abs(result.observed_r) + 1e-9 for r in result.trial_rs)


def test_run_audit_against_committed_fixture_runs_without_crashing():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    headlines = read_csv(fixture)

    result = run_audit(headlines, trials=50, seed=0)

    # Same n Day 5's correlate.py documents for this fixture.
    assert result.observed_n == 23
    assert 0.0 <= result.p_value <= 1.0
    assert len(result.trial_rs) == 50


def test_run_missing_input_reports_failure(tmp_path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, trials=10, seed=0, live=False)

    assert exit_code == 1


def test_run_against_committed_fixture_succeeds():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, trials=20, seed=0, live=False)

    assert exit_code == 0
