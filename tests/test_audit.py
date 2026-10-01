import datetime as dt
from datetime import datetime, time, timezone
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.audit import (
    contemporaneous_r,
    permutation_p_value,
    run,
    run_audit,
    shuffle_published_at,
)
from sentiment.headline import Headline, read_csv, write_csv
from sentiment.market_hours import is_trading_day, next_trading_day
from sentiment.prices import Bar, save_fixture
from sentiment.vader_score import score_headline

UTC = timezone.utc


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=UTC),
    )


def trading_days(start: dt.date, n: int) -> list[dt.date]:
    day = start if is_trading_day(start) else next_trading_day(start)
    days = [day]
    while len(days) < n:
        day = next_trading_day(day)
        days.append(day)
    return days


# One "shares <verb> ..." headline per ticker, phrased with real, varying
# sentiment words (not the saturated "Share Price Highlights" template,
# which scores an identical compound regardless of content - see README Day
# 5/6 - and so has zero variance to build a correlation out of).
SYNTHETIC_TITLES = {
    "PC Jeweller shares soar after record profit growth and a bullish upgrade": "PCJEWELLER.NS",
    "Fortis Healthcare shares surge on strong earnings beat and upgraded guidance": "FORTIS.NS",
    "BSE shares rally as volumes hit a record high": "BSE.NS",
    "PB Fintech shares climb on robust growth and positive analyst coverage": "POLICYBZR.NS",
    "NSE shares gain on steady outlook": "NSE.BO",
    "Suzlon Energy shares advance on healthy order book": "SUZLON.NS",
    "HDFC Bank shares plunge on weak results and a disappointing outlook": "HDFCBANK.NS",
    "Great Eastern Shipping shares tumble on falling freight rates and a profit warning": "GESHIP.NS",
    "Jefferies names Max Financial shares a laggard on soft guidance": "MFSL.NS",
}


def build_synthetic_leak_free_headlines(scale: float = 0.1) -> tuple[list[Headline], dict]:
    """Headlines and per-ticker bars engineered so that, under the TRUE
    alignment, contemporaneous_return == scale * compound exactly for every
    headline (a perfect linear relationship) - and == 0 on every other
    headline's trading day. Compound scores come from the real VADER scorer,
    not hand-picked values, so this only tests pairing/alignment, not
    whether VADER itself is predictable.

    Returns (headlines, bars_by_ticker) - the caller still has to
    monkeypatch ``prices.FIXTURE_DIR`` and ``save_fixture`` each ticker's
    bars before running the pipeline against them.
    """
    titles = list(SYNTHETIC_TITLES.items())
    days = trading_days(dt.date(2026, 9, 7), len(titles))

    headlines = []
    compounds = []
    for (title, ticker), day in zip(titles, days):
        published_at = datetime.combine(day, time(2, 0), tzinfo=UTC)  # 07:30 IST, pre-open
        h = make_headline(title, link=ticker, published_at=published_at)
        headlines.append(h)
        compounds.append(score_headline(h).compound)

    assert len(set(compounds)) > 1, "fixture headlines must carry varying sentiment to test a correlation"

    bars_by_ticker: dict[str, list[Bar]] = {}
    for (_, ticker), own_day, compound in zip(titles, days, compounds):
        bars = []
        for day in days:
            if day == own_day:
                bars.append(Bar(date=day, open=100.0, close=100.0 * (1 + scale * compound)))
            else:
                bars.append(Bar(date=day, open=100.0, close=100.0))
        bars_by_ticker[ticker] = bars

    return headlines, bars_by_ticker


def test_shuffle_published_at_preserves_the_timestamp_set_and_everything_else():
    headlines = [
        make_headline(f"Headline {i}", f"link-{i}", datetime(2026, 9, 7 + i, 2, 0, 0, tzinfo=UTC))
        for i in range(6)
    ]

    shuffled = shuffle_published_at(headlines, seed=1)

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    # a seed large enough relative to n should actually reorder something -
    # otherwise this "shuffle" would silently be a no-op control.
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_shuffle_published_at_is_deterministic_for_a_given_seed():
    headlines = [
        make_headline(f"Headline {i}", f"link-{i}", datetime(2026, 9, 7 + i, 2, 0, 0, tzinfo=UTC))
        for i in range(6)
    ]

    a = shuffle_published_at(headlines, seed=7)
    b = shuffle_published_at(headlines, seed=7)

    assert [h.published_at for h in a] == [h.published_at for h in b]


def test_permutation_p_value_is_small_when_real_r_is_extreme():
    assert permutation_p_value(0.95, [0.1, -0.2, 0.05, 0.3, -0.1]) == pytest.approx(1 / 6)


def test_permutation_p_value_is_large_when_real_r_is_typical_of_the_null():
    shuffled = [0.9, -0.9, 0.85, -0.85, 0.5, -0.5]
    assert permutation_p_value(0.05, shuffled) == pytest.approx(7 / 7)


def test_contemporaneous_r_needs_at_least_two_priced_headlines(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=dt.date(2026, 9, 7), open=100.0, close=101.0)])
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 7, 2, 0, 0, tzinfo=UTC),
        )
    ]

    assert contemporaneous_r(headlines) is None


def test_true_timing_recovers_a_near_perfect_correlation_but_shuffled_timing_destroys_it(
    tmp_path: Path, monkeypatch
):
    # This is the leakage test itself: engineer a pipeline run where sentiment
    # truly does predict contemporaneous return (by construction, under the
    # correct alignment), confirm the real pipeline finds it, then confirm
    # that randomising each headline's publish time - leaving the sentiment
    # scores, the tickers, and the set of timestamps all otherwise identical -
    # makes that relationship disappear. If it didn't disappear, the
    # "relationship" would have to be coming from something other than
    # correct headline-to-session timing, which is exactly the bug this
    # audit exists to catch.
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines, bars_by_ticker = build_synthetic_leak_free_headlines(scale=0.1)
    for ticker, bars in bars_by_ticker.items():
        save_fixture(ticker, bars)

    real_r = contemporaneous_r(headlines)
    assert real_r == pytest.approx(1.0, abs=1e-9)

    real_r_out, shuffled_rs, dropped = run_audit(headlines, n_shuffles=200, seed=0)
    assert real_r_out == pytest.approx(real_r)
    assert len(shuffled_rs) + dropped == 200
    assert len(shuffled_rs) > 0

    mean_abs_shuffled = sum(abs(r) for r in shuffled_rs) / len(shuffled_rs)
    assert mean_abs_shuffled < 0.5  # real_r=1.0 towers over the shuffled-timing null

    p = permutation_p_value(real_r, shuffled_rs)
    assert p < 0.05  # real r is an outlier against the shuffled-timing null, as it must be


def test_real_committed_fixture_is_not_an_outlier_against_its_own_shuffled_null():
    # The actual Day 8 "Done when" check: run the audit against the real,
    # committed headlines/prices (Day 5's already-null r=-0.185), not a
    # synthetic positive control. A genuinely leak-free pipeline should find
    # that real result unremarkable next to its own shuffled-timing null -
    # this is a stronger, CI-enforced version of "the signal must disappear"
    # than eyeballing a small CI once by hand.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    headlines = read_csv(fixture)

    real_r, shuffled_rs, dropped = run_audit(headlines, n_shuffles=100, seed=0)

    assert real_r is not None
    assert len(shuffled_rs) > 0
    p = permutation_p_value(real_r, shuffled_rs)
    assert p >= 0.05  # no evidence of leakage: consistent with Day 5's own null finding


def test_run_cli_against_committed_fixture_succeeds(tmp_path: Path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, n_shuffles=20, seed=0, live=False)

    assert exit_code == 0


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, n_shuffles=20, seed=0, live=False)

    assert exit_code == 1


def test_run_with_too_few_resolved_headlines_reports_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            make_headline(
                "LTIMindtree Share Price Highlights: LTIMindtree Stock Price History",
                "1",
                datetime(2026, 9, 28, 2, 0, 0, tzinfo=UTC),
            )
        ],
        in_path,
    )

    exit_code = run(in_path, n_shuffles=20, seed=0, live=False)

    assert exit_code == 1
