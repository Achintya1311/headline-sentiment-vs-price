from datetime import date, datetime, timedelta, timezone

import pytest

import sentiment.correlate as correlate
import sentiment.prices as prices
from sentiment.audit import contemporaneous_stat, is_significant, run_audit, shuffle_timestamps
from sentiment.headline import Headline
from sentiment.prices import Bar, save_fixture

# Eight distinct "Share Price Highlights" titles sentiment.tickers.resolve
# already maps to eight distinct, real tickers - reused as-is so this test
# exercises the real resolver, not a stand-in for it.
TICKERS = [
    ("Infosys Share Price Highlights: Infosys Stock Price History", "INFY.NS"),
    ("Wipro Share Price Highlights: Wipro Stock Price History", "WIPRO.NS"),
    ("Tech Mahindra Share Price Highlights: Tech Mahindra Stock Price History", "TECHM.NS"),
    ("HCL Tech Share Price Highlights: HCL Tech Stock Price History", "HCLTECH.NS"),
    ("Tata Steel Share Price Highlights: Tata Steel Stock Price History", "TATASTEEL.NS"),
    ("Bharti Airtel Share Price Highlights: Bharti Airtel Stock Price History", "BHARTIARTL.NS"),
    ("SBI Life Share Price Highlights: SBI Life Stock Price History", "SBILIFE.NS"),
    ("Nestle India Share Price Highlights: Nestle India Stock Price History", "NESTLEIND.NS"),
]
# Distinct, symmetric VADER compounds - real titles would never score exactly
# these, so the fake scorer below assigns them directly rather than relying
# on VADER's actual lexicon output (irrelevant to what this audit tests).
COMPOUNDS = [-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8]
SLOPE = 0.05


def trading_weekdays(start: date, count: int) -> list[date]:
    days: list[date] = []
    d = start
    while len(days) < count:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


DATES = trading_weekdays(date(2026, 9, 28), len(TICKERS))


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 0, 0, 0, tzinfo=timezone.utc),
    )


def jitter(i: int, j: int) -> float:
    """Small deterministic noise for ticker i's bar on another headline's
    date j, uncorrelated with ticker i's own compound - models "whatever
    this ticker's return happened to be on an unrelated day"."""
    return 0.001 * (((i * 3 + j * 7) % 9) - 4)


@pytest.fixture
def synthetic_fixtures(tmp_path, monkeypatch):
    """A fixture set engineered so that, aligned with each headline's real
    (pre-open) publication date, ticker i's contemporaneous return is an
    exact linear function of its own VADER compound (r = 1.0) - but on any
    *other* date, that same ticker's return is unrelated noise. Shuffling
    which headline gets which timestamp should therefore collapse the
    correlation, unless the pipeline isn't really using the timestamp.
    """
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    compound_by_title = {title: c for (title, _ticker), c in zip(TICKERS, COMPOUNDS)}

    class FakeScored:
        def __init__(self, compound: float) -> None:
            self.compound = compound

    def fake_score_headline(headline: Headline) -> FakeScored:
        return FakeScored(compound_by_title[headline.title])

    monkeypatch.setattr(correlate, "score_headline", fake_score_headline)

    for i, (_title, ticker) in enumerate(TICKERS):
        bars = []
        for j, d in enumerate(DATES):
            if j == i:
                close = 100.0 * (1 + SLOPE * COMPOUNDS[i])
            else:
                close = 100.0 * (1 + jitter(i, j))
            bars.append(Bar(date=d, open=100.0, close=close))
        save_fixture(ticker, bars)

    headlines = [
        make_headline(title, str(i), datetime(DATES[i].year, DATES[i].month, DATES[i].day, 2, 0, 0, tzinfo=timezone.utc))
        for i, (title, _ticker) in enumerate(TICKERS)
    ]
    return headlines


def test_real_alignment_recovers_the_engineered_signal(synthetic_fixtures):
    stat = contemporaneous_stat(synthetic_fixtures)
    assert stat is not None
    assert stat.n == len(TICKERS)
    assert stat.r == pytest.approx(1.0, abs=1e-9)
    assert is_significant(stat)


def test_shuffling_timestamps_destroys_the_engineered_signal(synthetic_fixtures):
    result = run_audit(synthetic_fixtures, trials=300, seed=0)

    assert result.real_significant is True
    assert result.real.r == pytest.approx(1.0, abs=1e-9)
    # The real, correctly-timestamped run is a clean r=1.0 - if the shuffled
    # control still looked "significant" nearly as often, the correlation
    # would not actually depend on getting timestamps right, which is
    # exactly the leak NEXT_STEPS.md's "Done when" test is built to catch.
    assert result.shuffled_significant_fraction < 0.20
    assert result.leak_suspected is False


def test_shuffle_timestamps_permutes_without_touching_other_fields(synthetic_fixtures):
    import random

    rng = random.Random(42)
    shuffled = shuffle_timestamps(synthetic_fixtures, rng)

    assert [h.title for h in shuffled] == [h.title for h in synthetic_fixtures]
    assert [h.link for h in shuffled] == [h.link for h in synthetic_fixtures]
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in synthetic_fixtures)
    # A genuine shuffle, not an accidental no-op, for this seed.
    assert [h.published_at for h in shuffled] != [h.published_at for h in synthetic_fixtures]


def test_run_audit_raises_when_too_few_headlines_resolve():
    with pytest.raises(ValueError):
        run_audit([])
