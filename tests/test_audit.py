"""Day 8: the shuffle-timestamp leakage audit, plus the synthetic positive
control that proves the audit mechanism itself actually works.

See ``sentiment/audit.py``'s module docstring for why both are needed: the
real fixture's correlation is already statistically indistinguishable from
zero (Day 5), so shuffling it and seeing "no signal" proves nothing about
whether the shuffle would have caught a real leak. The synthetic fixture
below has a built-in, deliberately strong sentiment/return relationship
instead, so there is an actual signal for the shuffle to destroy.
"""

import csv
import datetime as dt
from datetime import datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import (
    audit,
    correlation_for_rows,
    run,
    run_shuffle_trial,
    shuffled_timestamp_map,
    write_trials_csv,
)
from sentiment.correlate import build_rows
from sentiment.headline import Headline, write_csv
from sentiment.prices import Bar, save_fixture
from sentiment.vader_score import score_headline

# Eight headline shapes sentiment.tickers.resolve already recognises (the
# "<Company> shares ..." move-headline patterns, which accept free text
# after the company name - unlike the "Share Price Highlights" template,
# whose fixed wording scores identically regardless of company, see Day 5/6
# Findings), each rewritten with lexicon-strong words to span a wide,
# non-degenerate range of VADER compound scores.
_SYNTHETIC_TITLES_TICKERS = [
    ("PC Jeweller shares plunge as fraud allegations devastate investor confidence", "PCJEWELLER.NS"),
    ("Suzlon Energy shares tumble on disappointing weak outlook", "SUZLON.NS"),
    ("NSE shares dip slightly amid mixed trading", "NSE.BO"),
    ("BSE shares steady amid quiet trading session", "BSE.NS"),
    ("HDFC Bank shares edge higher on stable outlook", "HDFCBANK.NS"),
    ("Great Eastern Shipping shares gain on strong demand", "GESHIP.NS"),
    ("PB Fintech shares soar as excellent profit beats estimates", "POLICYBZR.NS"),
    ("Fortis Healthcare shares surge on outstanding breakthrough success", "FORTIS.NS"),
]

# Eight distinct NSE trading weekdays (all weekday-only per market_hours, no
# holiday calendar) so each headline aligns to its own session_date.
_SYNTHETIC_DATES = [
    dt.date(2026, 9, 28),
    dt.date(2026, 9, 29),
    dt.date(2026, 9, 30),
    dt.date(2026, 10, 1),
    dt.date(2026, 10, 2),
    dt.date(2026, 10, 5),
    dt.date(2026, 10, 6),
    dt.date(2026, 10, 7),
]

# Scales a headline's compound score into a session return small enough to
# look like a real daily move (compound is in [-1, 1], so returns here stay
# within +-10%).
_SIGNAL_SCALE = 0.1


def _make_leaky_fixture(in_path: Path) -> list[float]:
    """Build a synthetic headlines CSV plus matching price fixtures where
    the *only* way to see the sentiment/return relationship is to align
    each headline to its own, correct session_date: ticker i's bar return
    is ``compound_i * _SIGNAL_SCALE`` on its own correct date and exactly
    ``0.0`` on every other headline's date. A shuffle that moves headline i
    to a different date therefore does not just relabel the relationship -
    it looks up a genuinely different (flat) return, the same way a real
    look-ahead leak would behave differently once the true timestamp is no
    longer attached to the true headline. Returns the compound scores, in
    headline order, for the caller to compute the expected real correlation.
    """
    headlines = []
    for i, (title, _ticker) in enumerate(_SYNTHETIC_TITLES_TICKERS):
        d = _SYNTHETIC_DATES[i]
        # 02:00 UTC = 07:30 IST, before the 09:15 IST open - pre-open, so
        # align_headline keeps the session_date as the same calendar day.
        published_at = datetime(d.year, d.month, d.day, 2, 0, 0, tzinfo=timezone.utc)
        headlines.append(
            Headline(
                source="synthetic",
                title=title,
                link=f"synthetic-{i}",
                published_at=published_at,
                published_raw=published_at.isoformat(),
                scraped_at=published_at,
            )
        )
    write_csv(headlines, in_path)

    compounds = [score_headline(h).compound for h in headlines]
    for i, (_title, ticker) in enumerate(_SYNTHETIC_TITLES_TICKERS):
        bars = []
        for j, d in enumerate(_SYNTHETIC_DATES):
            ret = compounds[i] * _SIGNAL_SCALE if j == i else 0.0
            bars.append(Bar(date=d, open=100.0, close=100.0 * (1 + ret)))
        save_fixture(ticker, bars)
    return compounds


def test_shuffled_timestamp_map_permutes_the_same_multiset_of_timestamps():
    headlines = [
        Headline(
            source="s",
            title=f"title {i}",
            link=f"link{i}",
            published_at=datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc),
            published_raw="",
            scraped_at=datetime(2026, 9, 28, 0, 0, 0, tzinfo=timezone.utc),
        )
        for i in range(8)
    ]

    tmap = shuffled_timestamp_map(headlines, seed=1)

    assert sorted(tmap.values()) == sorted(h.published_at for h in headlines)
    # seed=1 actually moves at least one timestamp - a shuffle that silently
    # no-ops would defeat the whole point of this audit.
    assert any(tmap[h.dedup_key()] != h.published_at for h in headlines)


def test_correlation_for_rows_needs_at_least_two_points():
    assert correlation_for_rows([]) is None
    assert correlation_for_rows([{"compound": 0.5, "contemporaneous_return": 0.01}]) is None


def test_synthetic_leaky_fixture_has_near_perfect_real_correlation(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path / "prices")
    in_path = tmp_path / "raw.csv"
    _make_leaky_fixture(in_path)

    rows, unresolved = build_rows(in_path)

    assert unresolved == []
    assert len(rows) == 8
    real_r = correlation_for_rows(rows)
    # contemporaneous_return is an exact positive linear function of
    # compound by construction (ret = compound * _SIGNAL_SCALE), so Pearson
    # r must be 1.0 up to floating-point error.
    assert real_r is not None
    assert abs(real_r - 1.0) < 1e-9


def test_shuffling_timestamps_destroys_the_synthetic_signal(tmp_path: Path, monkeypatch):
    """The actual proof this audit mechanism works: a real, strong,
    deliberately-injected sentiment/return relationship collapses once the
    timestamps (and therefore the session_date each headline is priced
    against) are scrambled. Contrast with the real-fixture test below,
    where there is no signal to begin with - this test is what shows the
    difference is the scramble, not a degenerate "always returns ~0" bug.
    """
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path / "prices")
    in_path = tmp_path / "raw.csv"
    _make_leaky_fixture(in_path)

    rows, _ = build_rows(in_path)
    real_r = correlation_for_rows(rows)
    assert real_r is not None and real_r > 0.99

    n_trials = 30
    trial_rs = []
    for seed in range(n_trials):
        r, _n = run_shuffle_trial(in_path, seed)
        if r is not None:
            trial_rs.append(r)

    assert len(trial_rs) >= n_trials - 2  # almost every trial should still resolve 8 rows
    mean_abs_shuffled = sum(abs(r) for r in trial_rs) / len(trial_rs)
    # The real correlation is ~1.0; the shuffled trials should average well
    # below half of that, and no single trial should come close to
    # reproducing it - a scramble that regularly reproduced r~1.0 would mean
    # this audit could not actually tell a leak-free pipeline from a leaky one.
    assert mean_abs_shuffled < 0.5
    assert max(abs(r) for r in trial_rs) < 0.9


def test_audit_against_the_real_committed_fixture_matches_day5s_finding():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    result = audit(fixture, n_trials=40)

    assert result.real_n == 23
    # Day 5's README-recorded result: r=-0.185, n=23. Recomputed here rather
    # than re-asserted blindly, so a future change to the correlation
    # pipeline that silently moves this number gets caught.
    assert result.real_r is not None
    assert abs(result.real_r - (-0.185)) < 0.01
    # The real correlation should look unremarkable against its own
    # shuffled null - consistent with "no detectable signal to leak" per
    # Day 5, not "the leak test is broken and passes everything".
    assert result.empirical_p is not None
    assert result.empirical_p > 0.05


def test_run_shuffle_trial_drops_headlines_whose_shuffled_date_has_no_price_bar(tmp_path: Path, monkeypatch):
    # Give INFY.NS only one bar (its real, correct date). A shuffle that
    # reassigns some other headline's timestamp onto this one will look for
    # a bar on a date that was never fixtured - build_rows already drops
    # that as a price_errors case, and this trial must survive it, not crash.
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path / "prices")
    save_fixture("INFY.NS", [Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0)])
    save_fixture("WIPRO.NS", [Bar(date=dt.date(2026, 9, 29), open=50.0, close=49.0)])

    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            Headline(
                source="s",
                title="Infosys Share Price Highlights: Infosys Stock Price History",
                link="1",
                published_at=datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
                published_raw="",
                scraped_at=datetime(2026, 9, 28, 0, 0, 0, tzinfo=timezone.utc),
            ),
            Headline(
                source="s",
                title="Wipro Share Price Highlights: Wipro Stock Price History",
                link="2",
                published_at=datetime(2026, 9, 29, 2, 0, 0, tzinfo=timezone.utc),
                published_raw="",
                scraped_at=datetime(2026, 9, 29, 0, 0, 0, tzinfo=timezone.utc),
            ),
        ],
        in_path,
    )

    # Does not raise, even though some seed is near-guaranteed to swap these
    # two onto a date their own ticker has no bar for.
    for seed in range(10):
        r, n = run_shuffle_trial(in_path, seed)
        assert n <= 2
        if r is not None:
            assert isinstance(r, float)


def test_run_writes_output_csv_and_reports_on_the_real_fixture(tmp_path: Path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    out_path = tmp_path / "audit_shuffle.csv"

    exit_code = run(fixture, out_path, live=False, n_trials=20)

    assert exit_code == 0
    assert out_path.exists()
    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20
    assert set(rows[0].keys()) == {"seed", "shuffled_r"}


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    out_path = tmp_path / "audit_shuffle.csv"

    exit_code = run(missing, out_path, live=False, n_trials=10)

    assert exit_code == 1
    assert not out_path.exists()


def test_write_trials_csv_round_trips_seed_and_r(tmp_path: Path):
    from sentiment.audit import AuditResult

    result = AuditResult(real_r=0.5, real_n=5, trial_rs=[0.1, -0.2, 0.3])
    out_path = tmp_path / "trials.csv"

    write_trials_csv(result, out_path)

    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["seed"] for r in rows] == ["0", "1", "2"]
    assert [float(r["shuffled_r"]) for r in rows] == [0.1, -0.2, 0.3]
