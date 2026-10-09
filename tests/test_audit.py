import itertools
from datetime import date, datetime, timezone
from pathlib import Path

import sentiment.correlate as correlate
import sentiment.prices as prices
from sentiment.audit import contemporaneous_r, run, run_audit, shuffle_timestamps
from sentiment.headline import Headline
from sentiment.market_hours import align_headline
from sentiment.prices import Bar, save_fixture

UTC = timezone.utc

# Six resolvable single-company headline shapes from sentiment/tickers.py,
# each given its own distinct pre-open weekday so each aligns to its own,
# unambiguous session_date.
TICKERS = ["INFY.NS", "WIPRO.NS", "TECHM.NS", "HCLTECH.NS", "BHARTIARTL.NS", "TATASTEEL.NS"]
TITLES = [
    "Infosys Share Price Highlights: Infosys Stock Price History",
    "Wipro Share Price Highlights: Wipro Stock Price History",
    "Tech Mahindra Share Price Highlights: Tech Mahindra Stock Price History",
    "HCL Tech Share Price Highlights: HCL Tech Stock Price History",
    "Bharti Airtel Share Price Highlights: Bharti Airtel Stock Price History",
    "Tata Steel Share Price Highlights: Tata Steel Stock Price History",
]
DATES = [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 7), date(2026, 9, 8)]
COMPOUNDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def make_headline(title: str, link: str, d: date) -> Headline:
    # 02:00 UTC = 07:30 IST, pre-open every one of these NSE trading days.
    published_at = datetime(d.year, d.month, d.day, 2, 0, 0, tzinfo=UTC)
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=published_at,
    )


def synthetic_headlines() -> list[Headline]:
    return [make_headline(t, str(i), d) for i, (t, d) in enumerate(zip(TITLES, DATES))]


def install_synthetic_bars(tmp_path: Path, monkeypatch) -> None:
    """Every ticker gets an identical bar on every one of the 6 dates, with
    that date's open-to-close return set to the compound score of whichever
    headline's *true* date that originally was. A row's contemporaneous
    return therefore depends only on which date it lands on, not on which
    ticker it is - so a leak-free pipeline reports a perfect correlation on
    the real (unshuffled) run, and shuffling which headline owns which date
    is the only thing that can break the pairing."""
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    bars = [Bar(date=d, open=100.0, close=100.0 * (1 + c)) for d, c in zip(DATES, COMPOUNDS)]
    for ticker in TICKERS:
        save_fixture(ticker, bars)


def install_compound_stub(monkeypatch) -> None:
    """Replace VADER scoring with an exact, known compound per title, so the
    synthetic correlation above is controlled precisely rather than at the
    mercy of whatever VADER happens to score this test's placeholder text."""
    compound_by_title = dict(zip(TITLES, COMPOUNDS))

    class _Scored:
        def __init__(self, compound: float) -> None:
            self.compound = compound

    def fake_score_headline(headline):
        return _Scored(compound_by_title[headline.title])

    monkeypatch.setattr(correlate, "score_headline", fake_score_headline)


def test_shuffle_timestamps_preserves_the_multiset_but_changes_the_pairing():
    headlines = synthetic_headlines()
    shuffled = shuffle_timestamps(headlines, __import__("random").Random(1))

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]
    # titles/links are untouched - only which timestamp each one carries moves.
    assert [h.title for h in shuffled] == [h.title for h in headlines]


def test_contemporaneous_r_needs_at_least_two_rows():
    assert contemporaneous_r([]) is None
    assert contemporaneous_r([{"compound": 0.1, "contemporaneous_return": 0.01}]) is None
    rows = [
        {"compound": 0.1, "contemporaneous_return": 0.01},
        {"compound": 0.2, "contemporaneous_return": 0.02},
    ]
    assert contemporaneous_r(rows) == 1.0


def test_run_audit_collapses_a_genuine_timestamp_mediated_signal(tmp_path, monkeypatch):
    install_synthetic_bars(tmp_path, monkeypatch)
    install_compound_stub(monkeypatch)
    headlines = synthetic_headlines()

    result = run_audit(headlines, n_perm=300, seed=0)

    # Real run: each headline's own compound sits on its own date, which this
    # fixture made equal to that same compound - a perfect, genuine
    # correlation (not a leak: it is produced by the real published_at of
    # each real headline, nothing else).
    assert round(result.real_r, 9) == 1.0
    assert result.n_rows == 6
    # Shuffling which headline owns which date almost always breaks the
    # pairing: the null distribution should sit well below the real r, and
    # the real r should be a clear outlier against its own control.
    assert result.null_mean < 0.3
    assert result.p_value < 0.05


def test_run_audit_flags_a_leak_that_ignores_the_given_timestamp(tmp_path, monkeypatch):
    """A pipeline bug that returns each headline's *correct* alignment by
    call position rather than by the ``published_at`` it was actually given
    (e.g. a stale cache or a captured-list bug) would still produce the
    right answer after a shuffle, because the bug never looks at the
    shuffled timestamp in the first place. This is exactly the leak the
    Day 8 audit exists to catch - and exactly the failure mode shuffling the
    *timestamp* alone cannot detect unless the underlying pipeline actually
    uses it, which is what this test proves by breaking that on purpose."""
    install_synthetic_bars(tmp_path, monkeypatch)
    install_compound_stub(monkeypatch)
    headlines = synthetic_headlines()

    true_alignments = [align_headline(h.published_at) for h in headlines]
    counter = itertools.count()

    def leaky_align_headline(published_at):
        # Ignores the argument entirely - returns answers by call order.
        idx = next(counter) % len(true_alignments)
        return true_alignments[idx]

    monkeypatch.setattr(correlate, "align_headline", leaky_align_headline)

    result = run_audit(headlines, n_perm=300, seed=0)

    assert round(result.real_r, 9) == 1.0
    # Every shuffled re-run still gets each headline's own correct session
    # regardless of the timestamp it was actually given - the "signal" does
    # not disappear, which is precisely the control failing (correctly).
    assert result.null_mean > 0.9
    assert result.p_value > 0.9


def test_run_audit_raises_when_too_few_headlines_resolve(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headline = make_headline(TITLES[0], "0", DATES[0])
    save_fixture(TICKERS[0], [Bar(date=DATES[0], open=100.0, close=101.0)])

    import pytest

    with pytest.raises(ValueError, match="not enough for a correlation"):
        run_audit([headline], n_perm=10, seed=0)


def test_run_against_committed_fixture_matches_documented_day5_result(tmp_path):
    """Regression check against the real, committed 50-headline fixture: Day
    5's README reports contemporaneous r=-0.185 (n=23). This also doubles as
    the "runs in CI, not once by hand" leakage gate NEXT_STEPS.md's Day 8
    'Done when' calls for, and records its honest result rather than a
    scripted pass: see the README's Day 8 Findings for why this particular
    fixture's audit is inconclusive about leakage rather than a clean pass."""
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    out_path = tmp_path / "audit.txt"

    from sentiment.headline import read_csv

    headlines = read_csv(fixture)
    result = run_audit(headlines, n_perm=300, seed=0)

    assert result.n_rows == 23
    assert round(result.real_r, 3) == -0.185
    assert 0.0 <= result.p_value <= 1.0


def test_run_cli_against_committed_fixture_succeeds():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    exit_code = run(fixture, n_perm=50, seed=0, live=False)
    assert exit_code == 0


def test_run_missing_input_reports_failure(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    exit_code = run(missing, n_perm=50, seed=0, live=False)
    assert exit_code == 1
