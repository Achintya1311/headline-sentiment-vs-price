import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import sentiment.audit as audit
import sentiment.prices as prices
from sentiment.headline import Headline
from sentiment.prices import Bar, save_fixture
from sentiment.vader_score import ScoredHeadline


def make_headline(link: str, published_at: datetime, title: str = "Infosys Share Price Highlights: x") -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def weekday_dates(start: str, n: int):
    """``n`` consecutive weekday dates (as datetime.date) from ``start`` (ISO)."""
    from datetime import date

    d = date.fromisoformat(start)
    dates = []
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def test_shuffle_timestamps_keeps_content_but_redistributes_timing():
    headlines = [make_headline(f"link-{i}", datetime(2026, 9, 28, 2, i, tzinfo=timezone.utc)) for i in range(5)]
    rng = random.Random(0)

    shuffled = audit.shuffle_timestamps(headlines, rng)

    assert [h.link for h in shuffled] == [h.link for h in headlines]
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert {h.published_at for h in shuffled} == {h.published_at for h in headlines}
    # a non-trivial shuffle: at least one headline's timestamp actually moved
    assert any(a.published_at != b.published_at for a, b in zip(headlines, shuffled))


def test_build_rows_drops_unresolved_and_unfetchable_headlines(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [Bar(date=__import__("datetime").date(2026, 9, 28), open=100.0, close=101.0)],
    )
    headlines = [
        make_headline("resolvable", datetime(2026, 9, 28, 2, 0, tzinfo=timezone.utc)),
        make_headline(
            "no-such-company", datetime(2026, 9, 28, 2, 0, tzinfo=timezone.utc), title="Totally unrelated headline"
        ),
        make_headline(
            "known-but-no-ticker",
            datetime(2026, 9, 28, 2, 0, tzinfo=timezone.utc),
            title="LTIMindtree Share Price Highlights: x",
        ),
    ]

    rows = audit.build_rows(headlines)

    assert len(rows) == 1


def test_real_fixture_correlation_matches_documented_day5_finding():
    # Same real, committed 50-headline fixture Day 5 already correlated -
    # this just re-derives the contemporaneous correlation through the
    # audit module's own build_rows/correlation path, as a check that the
    # audit's pairing logic agrees with sentiment.correlate's.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    from sentiment.headline import read_csv

    headlines = read_csv(fixture)

    result = audit.correlation(headlines)

    assert result is not None
    assert result.n == 23
    assert result.r == pytest.approx(-0.185, abs=0.001)


def test_shuffle_null_distribution_is_consistent_with_no_leak_on_the_real_fixture():
    # The real fixture's own correlation is already a null result (see
    # README Day 5 Findings), so the shuffled null distribution should not
    # systematically differ from it - there is nothing for a leak to reveal
    # here. This exercises the CLI's own statistic, not a synthetic signal.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    from sentiment.headline import read_csv

    headlines = read_csv(fixture)
    real = audit.correlation(headlines)
    assert real is not None

    null_rs = audit.shuffle_null_distribution(headlines, n_shuffles=200, seed=0)

    assert len(null_rs) > 150  # most shuffles should still resolve rows
    mean_null = sum(null_rs) / len(null_rs)
    # real |r| should not be some extreme outlier relative to the shuffled
    # null - a generous bound, since this is a sanity check against a
    # fixture with no real signal, not a power test (see the synthetic
    # test below for that).
    assert abs(real.r) <= 3 * mean_null + 0.2


def test_shuffling_destroys_a_real_injected_signal(tmp_path: Path, monkeypatch):
    """The leakage control's actual job: prove it can detect *and destroy* a
    genuine signal, not just that it passes on a fixture that already has
    none (see the two tests above).

    Built so that, unshuffled, each headline's VADER compound exactly
    determines the return of the one session its real timestamp aligns to
    (return = 0.01 * compound). Monkeypatching ``score_headline`` ties each
    headline's score to its own link rather than running real VADER, so the
    score travels with the headline's *content* exactly like production
    code - only ``published_at`` ever gets shuffled.
    """
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    n = 20
    dates = weekday_dates("2026-09-07", n)
    compounds = {f"link-{i}": (i - n / 2) / n for i in range(n)}  # spread of distinct scores

    # Each headline is pre-open (02:00 UTC = 07:30 IST) on its own distinct
    # trading day, so align_headline maps it to exactly that day - no two
    # headlines share a session to begin with.
    headlines = [
        make_headline(f"link-{i}", datetime(dates[i].year, dates[i].month, dates[i].day, 2, 0, tzinfo=timezone.utc))
        for i in range(n)
    ]

    bars = [
        Bar(date=d, open=100.0, close=100.0 * (1 + 0.01 * compounds[f"link-{i}"])) for i, d in enumerate(dates)
    ]
    save_fixture("INFY.NS", bars)

    def fake_score_headline(headline: Headline) -> ScoredHeadline:
        compound = compounds[headline.link]
        return ScoredHeadline(
            source=headline.source,
            title=headline.title,
            link=headline.link,
            published_at=headline.published_at.isoformat(),
            neg=0.0,
            neu=1.0 - abs(compound),
            pos=abs(compound),
            compound=compound,
            label="positive" if compound > 0 else "negative" if compound < 0 else "neutral",
        )

    monkeypatch.setattr(audit, "score_headline", fake_score_headline)

    real = audit.correlation(headlines)
    assert real is not None
    assert real.n == n
    # by construction the real pairing is a perfect line: return = 0.01 * compound
    assert real.r == pytest.approx(1.0, abs=1e-9)

    null_rs = audit.shuffle_null_distribution(headlines, n_shuffles=300, seed=0)
    assert len(null_rs) > 250
    mean_null = sum(null_rs) / len(null_rs)

    # the whole point of the control: shuffling timestamps (which reassigns
    # each headline's *session*, not its compound) must wreck the real r=1.0
    # relationship, on average, by a wide margin.
    assert mean_null < 0.5
    assert abs(real.r) > 2 * mean_null
