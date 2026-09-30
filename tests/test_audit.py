import random
from datetime import date, datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import (
    contemporaneous_r,
    run,
    run_shuffle_audit,
    shuffle_published_at,
)
from sentiment.headline import Headline, read_csv
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


def test_shuffle_published_at_permutes_times_keeps_titles_and_multiset():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 21 + i, 2, 0, 0, tzinfo=timezone.utc))
        for i in range(5)
    ]
    rng = random.Random(1)

    shuffled = shuffle_published_at(headlines, rng)

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # with a real shuffle at n=5, at least one headline should land on a time
    # that was not originally its own (this seed is checked to do so).
    assert any(a.published_at != b.published_at for a, b in zip(headlines, shuffled))


def test_contemporaneous_r_needs_at_least_two_rows():
    assert contemporaneous_r([]) is None
    assert contemporaneous_r([{"compound": 0.5, "contemporaneous_return": 0.01}]) is None
    rows = [
        {"compound": 0.5, "contemporaneous_return": 0.01},
        {"compound": -0.5, "contemporaneous_return": -0.01},
    ]
    assert contemporaneous_r(rows) is not None


# --- the leakage-detector correctness check -------------------------------
#
# Day 5/6/7 already found this repo's own fixture has no real sentiment/
# return signal, so a permutation test against *that* data can only ever
# report "nothing here, nothing to shuffle away" - it cannot by itself prove
# the shuffle actually destroys a genuine signal when one exists. This test
# builds a synthetic fixture with a strong, deliberately planted relationship
# (positive-sentiment headlines paired with big positive moves, negative
# with big negative moves, on 10 distinct trading days) and checks both
# halves of NEXT_STEPS.md's "Done when": the real r is large, and the
# shuffled-timestamp controls collapse it back toward zero.

_TICKER_RULES = [
    ("Infosys Share Price Highlights", "INFY.NS"),
    ("Wipro Share Price Highlights", "WIPRO.NS"),
    ("Tata Steel Share Price Highlights", "TATASTEEL.NS"),
    ("Tech Mahindra Share Price Highlights", "TECHM.NS"),
    ("Bharti Airtel Share Price Highlights", "BHARTIARTL.NS"),
    ("HUL Share Price Highlights", "HINDUNILVR.NS"),
    ("HCL Tech Share Price Highlights", "HCLTECH.NS"),
    ("HDFC Life Share Price Highlights", "HDFCLIFE.NS"),
    ("Eicher Motors Share Price Highlights", "EICHERMOT.NS"),
    ("Bajaj Finserv Share Price Highlights", "BAJAJFINSV.NS"),
]

_DATES = [
    date(2026, 9, 21),
    date(2026, 9, 22),
    date(2026, 9, 23),
    date(2026, 9, 24),
    date(2026, 9, 25),
    date(2026, 9, 28),
    date(2026, 9, 29),
    date(2026, 9, 30),
    date(2026, 10, 1),
    date(2026, 10, 2),
]

_POS_WORDS = "fantastic excellent outstanding wonderful superb amazing brilliant terrific"
_NEG_WORDS = "terrible awful disastrous horrible dreadful atrocious appalling horrific"


def _off_home_noise(i: int, j: int) -> float:
    """Deterministic pseudo-random return in [-0.08, 0.08] for ticker ``i`` on
    a date that is not its own home date ``j``. Needs real magnitude and a
    sign uncorrelated with ``i``'s sentiment direction - a *flat* off-home
    return would let the rare shuffle that happens to fix one headline back
    onto its own home date dominate the correlation on its own (one huge,
    correctly-signed outlier against nine near-zero points), which would
    make shuffled controls look artificially signal-bearing for a reason
    that has nothing to do with leakage."""
    h = (i * 2654435761 + j * 40503) % 1009
    return (h / 1009) * 0.16 - 0.08


def _build_synthetic_signal_fixture(tmp_path: Path, monkeypatch) -> list[Headline]:
    """10 headlines, each naming a distinct ticker and published pre-open on
    its own distinct home date. The first 5 are strongly positive headlines
    on a +10% day; the last 5 strongly negative on a -10% day. Every ticker
    also gets a bar for the other 9 dates with noise uncorrelated to its own
    sentiment (see ``_off_home_noise``), so a shuffled headline always finds
    a bar - just not the one its sentiment was written to predict."""
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    headlines = []
    for i, (prefix, ticker) in enumerate(_TICKER_RULES):
        home_return = 0.10 if i < 5 else -0.10
        bars = []
        for j, d in enumerate(_DATES):
            ret = home_return if j == i else _off_home_noise(i, j)
            bars.append(Bar(date=d, open=100.0, close=100.0 * (1 + ret)))
        save_fixture(ticker, bars)

        words = _POS_WORDS if i < 5 else _NEG_WORDS
        title = f"{prefix}: {words}"
        published_at = datetime(_DATES[i].year, _DATES[i].month, _DATES[i].day, 2, 0, 0, tzinfo=timezone.utc)
        headlines.append(make_headline(title, str(i), published_at))

    return headlines


def test_shuffle_audit_real_signal_is_strong_and_shuffled_controls_destroy_it(tmp_path, monkeypatch):
    headlines = _build_synthetic_signal_fixture(tmp_path, monkeypatch)

    result = run_shuffle_audit(headlines, n_shuffles=300, seed=0)

    # the planted relationship is real and large in the unshuffled data...
    assert result.real_r > 0.85
    assert result.real_n == 10

    # ...but decoupling sentiment from timing collapses it: shuffled
    # controls average out near zero and almost never reproduce something
    # as extreme as the real result.
    assert abs(result.shuffled_mean) < 0.2
    assert result.p_value < 0.05


def test_shuffle_audit_raises_with_fewer_than_two_resolved_headlines(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    ]
    save_fixture("INFY.NS", [Bar(date=date(2026, 9, 28), open=100.0, close=105.0)])

    try:
        run_shuffle_audit(headlines, n_shuffles=10, seed=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    exit_code = run(missing, live=False, n_shuffles=10, seed=0)
    assert exit_code == 1


def test_run_against_committed_fixture_does_not_crash():
    # Smoke test only - this repo's own real fixture already has no signal
    # (see README Day 5/6/7 Findings), so this just checks the CLI plumbs
    # the real 23-headline fixture through the shuffle audit end to end
    # without error, not that it finds anything.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    headlines = read_csv(fixture)
    assert len(headlines) > 0

    exit_code = run(fixture, live=False, n_shuffles=20, seed=0)
    assert exit_code == 0
