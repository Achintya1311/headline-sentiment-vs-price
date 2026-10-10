import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import sentiment.correlate as correlate_mod
import sentiment.prices as prices
from sentiment.audit import run, run_audit, shuffle_published_at
from sentiment.headline import Headline, write_csv
from sentiment.prices import Bar, save_fixture
from sentiment.vader_score import ScoredHeadline


def _patch_score_headline(monkeypatch, compound_for: Callable[[Headline], float]) -> None:
    """Replace sentiment.correlate's score_headline with one whose compound
    is whatever the test wants, keeping every other ScoredHeadline field a
    harmless placeholder - only ``.compound`` is read downstream."""

    def fake_score_headline(h: Headline) -> ScoredHeadline:
        return ScoredHeadline(
            source=h.source,
            title=h.title,
            link=h.link,
            published_at=h.published_at.isoformat(),
            neg=0.0,
            neu=0.0,
            pos=0.0,
            compound=compound_for(h),
            label="neutral",
        )

    monkeypatch.setattr(correlate_mod, "score_headline", fake_score_headline)


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_shuffle_published_at_preserves_the_multiset_of_timestamps_but_reassigns_them():
    headlines = [
        make_headline(f"H{i}", f"link-{i}", datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc)) for i in range(8)
    ]
    rng = random.Random(0)

    shuffled = shuffle_published_at(headlines, rng)

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    assert [h.title for h in shuffled] == [h.title for h in headlines]  # content untouched
    assert any(h.published_at != orig.published_at for h, orig in zip(shuffled, headlines))


def test_shuffle_published_at_is_deterministic_given_a_seeded_rng():
    headlines = [
        make_headline(f"H{i}", f"link-{i}", datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc)) for i in range(6)
    ]
    first = shuffle_published_at(headlines, random.Random(42))
    second = shuffle_published_at(headlines, random.Random(42))
    assert [h.published_at for h in first] == [h.published_at for h in second]


# Eight "Share Price Highlights" tickers sentiment.tickers.resolve already
# knows (see sentiment/tickers.py) - used below to build a synthetic
# multi-ticker fixture, since the real committed fixture's 23 headlines are
# the honest Day 5-7 null, not something that exercises the FAIL path.
_LEAK_TICKERS = [
    ("SBI Life", "SBILIFE.NS", -0.04),
    ("Nestle India", "NESTLEIND.NS", -0.03),
    ("Sun Pharma", "SUNPHARMA.NS", -0.02),
    ("Grasim Inds", "GRASIM.NS", -0.01),
    ("Tech Mahindra", "TECHM.NS", 0.01),
    ("Wipro", "WIPRO.NS", 0.02),
    ("Bharti Airtel", "BHARTIARTL.NS", 0.03),
    ("Tata Steel", "TATASTEEL.NS", 0.04),
]


def _write_flat_bars_leaking_fixture(tmp_path: Path) -> list[Headline]:
    """A ticker whose own price history is flat (same open/close every
    session) makes its contemporaneous return insensitive to which session
    date a headline lands on - so if compound is wired to that same flat
    return, the correlation survives shuffling the headline's timestamp,
    because shuffling only changes which (identical) bar gets picked. This
    is the "pipeline is leaking" case the audit exists to catch.
    """
    headlines = []
    for i, (company, ticker, ret) in enumerate(_LEAK_TICKERS):
        bars = [
            Bar(date=prices.date(2026, 9, 28), open=100.0, close=100.0 * (1 + ret)),
            Bar(date=prices.date(2026, 9, 29), open=100.0, close=100.0 * (1 + ret)),
        ]
        save_fixture(ticker, bars)
        published_day = 28 if i % 2 == 0 else 29
        headlines.append(
            make_headline(
                f"{company} Share Price Highlights: {company} Stock Price History",
                f"link-{i}",
                datetime(2026, 9, published_day, 2, i, 0, tzinfo=timezone.utc),  # pre-open IST either day
            )
        )
    return headlines


def test_run_audit_fails_when_a_flat_price_history_survives_shuffling(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = _write_flat_bars_leaking_fixture(tmp_path)
    # Wire each headline's compound to the same flat return its ticker's
    # price history always produces, regardless of which day is picked -
    # exactly the artifact this audit is built to catch. compound isn't a
    # Headline field; sentiment.correlate scores the title with VADER, so
    # the title itself has to carry that sentiment. Easier and exact:
    # monkeypatch score_headline instead of hand-tuning headline text.
    ret_by_title = {h.title: ret for h, (_, _, ret) in zip(headlines, _LEAK_TICKERS)}
    _patch_score_headline(monkeypatch, lambda h: 5.0 * ret_by_title[h.title])

    result = run_audit(headlines, n_shuffles=100, seed=0)

    assert result is not None
    assert abs(result.real.r) > 0.9  # perfect-ish fit by construction
    assert result.shuffled_significant_rate > 0.5  # survives shuffling almost every time
    assert result.passed is False


def _null_headlines() -> list[Headline]:
    """Compound uncorrelated with each ticker's (also flat) return - the
    PASS counterpart to the FAIL fixture above: a shuffled control should
    not manufacture significance when there is nothing for it to find
    either way, same honest-null shape as Day 5-7's real result."""
    return [
        make_headline(
            f"{company} Share Price Highlights: {company} Stock Price History",
            f"null-link-{i}",
            datetime(2026, 9, 28 if i % 2 == 0 else 29, 2, i, 0, tzinfo=timezone.utc),
        )
        for i, (company, _, _) in enumerate(_LEAK_TICKERS)
    ]


def test_run_audit_passes_when_the_real_pairing_has_no_signal_either(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = _null_headlines()
    for _, ticker, ret in _LEAK_TICKERS:
        bars = [
            Bar(date=prices.date(2026, 9, 28), open=100.0, close=100.0 * (1 + ret)),
            Bar(date=prices.date(2026, 9, 29), open=100.0, close=100.0 * (1 + ret)),
        ]
        save_fixture(ticker, bars)

    # Fixed compound scores with no relationship to ticker order/return at all.
    fixed_compounds = [0.1, -0.3, 0.2, -0.1, 0.05, -0.25, 0.15, -0.05]
    compound_by_title = {h.title: c for h, c in zip(headlines, fixed_compounds)}
    _patch_score_headline(monkeypatch, lambda h: compound_by_title[h.title])

    result = run_audit(headlines, n_shuffles=100, seed=0)

    assert result is not None
    assert result.shuffled_significant_rate <= 0.15
    assert result.passed is True


def test_run_against_committed_fixture_passes_consistent_with_days_5_through_7(tmp_path: Path):
    # The honest Day 5-7 finding is a null result (r=-0.185, CI including
    # 0), so there is no real signal for shuffling to make disappear - the
    # audit's job here is to confirm shuffling doesn't manufacture a false
    # one either, which README Day 8 Findings records plainly.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, live=False, n_shuffles=100, seed=0)

    assert exit_code == 0


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, live=False, n_shuffles=50, seed=0)

    assert exit_code == 1


def test_run_with_too_few_resolved_headlines_reports_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            make_headline(
                "Cyient among 4 stocks showing White Marubozu Pattern",
                "1",
                datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            )
        ],
        in_path,
    )

    exit_code = run(in_path, live=False, n_shuffles=50, seed=0)

    assert exit_code == 1
