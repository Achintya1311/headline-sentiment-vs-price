import datetime as dt
import random
from datetime import datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import lagged_correlation, run_audit, shuffle_timestamps
from sentiment.headline import Headline, write_csv
from sentiment.prices import Bar, save_fixture
from sentiment.vader_score import score_text

# A spread of titles whose VADER compound runs from clearly negative to
# clearly positive, all matching sentiment.tickers.resolve's Infosys rule
# (it only requires the title to start with "Infosys Share Price
# Highlights" - what follows is free to vary the VADER score without
# changing which company/ticker the headline resolves to).
_TITLES = [
    "Infosys Share Price Highlights: disastrous terrible horrible awful collapse",
    "Infosys Share Price Highlights: very bad news today",
    "Infosys Share Price Highlights: disappointing weak results today",
    "Infosys Share Price Highlights: somewhat bad news today",
    "Infosys Share Price Highlights: okay news today",
    "Infosys Share Price Highlights: slightly good news today",
    "Infosys Share Price Highlights: good news reported",
    "Infosys Share Price Highlights: great news reported",
    "Infosys Share Price Highlights: wonderful fantastic great news",
    "Infosys Share Price Highlights: excellent amazing wonderful fantastic news",
]

_K = 0.5  # how strongly the synthetic next-day return tracks compound


def _make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
    )


def _trading_days(start: dt.date, n: int) -> list[dt.date]:
    """``n`` consecutive NSE trading days (weekdays) starting at ``start``."""
    days: list[dt.date] = []
    day = start
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def _build_synthetic_fixture(tmp_path: Path) -> Path:
    """A fixture with a genuine, by-construction signal: each headline's
    next-day (lagged) return is exactly ``_K * compound`` of that same
    headline, via a shared ticker whose daily bars are built to match.
    This is the positive control - the audit must detect this signal in the
    correctly timestamped data and watch it collapse under shuffling, or
    the audit itself is not testing anything.
    """
    compounds = [score_text(title)["compound"] for title in _TITLES]
    assert len(set(compounds)) == len(compounds), "titles must give distinct compounds for a clean correlation"

    # one trading day per headline, plus one extra day so the last
    # headline's lagged (next-day) return is defined too.
    days = _trading_days(dt.date(2030, 1, 7), len(_TITLES) + 1)

    bars = [Bar(date=days[0], open=100.0, close=100.0)]
    for i, compound in enumerate(compounds):
        prev_close = bars[-1].close
        next_close = prev_close * (1 + _K * compound)
        bars.append(Bar(date=days[i + 1], open=prev_close, close=next_close))
    save_fixture("INFY.NS", bars)

    headlines = [
        _make_headline(title, link=str(i), published_at=datetime(days[i].year, days[i].month, days[i].day, 2, 0, 0, tzinfo=timezone.utc))
        for i, title in enumerate(_TITLES)
    ]
    in_path = tmp_path / "synthetic_raw.csv"
    write_csv(headlines, in_path)
    return in_path


def test_shuffle_timestamps_permutes_without_changing_the_set(tmp_path: Path):
    headlines = [
        _make_headline("Infosys Share Price Highlights: a", "1", datetime(2030, 1, 7, 2, 0, 0, tzinfo=timezone.utc)),
        _make_headline("Infosys Share Price Highlights: b", "2", datetime(2030, 1, 8, 2, 0, 0, tzinfo=timezone.utc)),
        _make_headline("Infosys Share Price Highlights: c", "3", datetime(2030, 1, 9, 2, 0, 0, tzinfo=timezone.utc)),
    ]
    shuffled = shuffle_timestamps(headlines, random.Random(0))

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert {h.published_at for h in shuffled} == {h.published_at for h in headlines}


def test_shuffle_timestamps_actually_reassigns_at_least_one_pair():
    headlines = [
        _make_headline("x", str(i), datetime(2030, 1, 7 + i, 2, 0, 0, tzinfo=timezone.utc)) for i in range(8)
    ]
    # seed 0 on 8 distinct items should not land on the identity permutation.
    shuffled = shuffle_timestamps(headlines, random.Random(0))
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_lagged_correlation_reports_n_below_two_as_no_signal():
    r, n = lagged_correlation([{"compound": 0.1, "lagged_return": 0.01}])
    assert (r, n) == (0.0, 1)


def test_run_audit_detects_the_synthetic_signal_and_the_shuffle_destroys_it(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    in_path = _build_synthetic_fixture(tmp_path)

    result = run_audit(in_path, live=False, n_shuffles=200, seed=0)

    # Positive control: the correctly-timestamped pipeline must recover the
    # signal that was built into the fixture by construction.
    assert result.real_n == len(_TITLES)
    assert result.real_r > 0.9, f"expected a near-perfect real correlation, got {result.real_r!r}"

    # And the control this audit actually exists for: scrambling timestamps
    # must make that same signal disappear - the shuffled distribution
    # should sit far below the real statistic, centered near zero.
    assert result.n_shuffles_used > 0
    assert abs(result.shuffled_mean) < 0.3, f"shuffled mean r should collapse toward 0, got {result.shuffled_mean!r}"
    assert result.permutation_p_value < 0.05, "the real signal should be an outlier against the shuffled null"
    assert not result.passes, (
        "a pipeline with a real, alignment-dependent signal should fail this audit's "
        "'is the real result unremarkable against the shuffled null' check - the real "
        "result is NOT supposed to look like noise when there genuinely is a signal"
    )


def test_run_audit_on_the_real_fixture_finds_no_outlier_signal():
    """On this repo's actual committed fixture, Day 5/6 already found no
    predictive signal (lagged r ~ 0, saturated compound scores - see
    README Findings). The audit should agree: the real statistic should
    not stand out against the shuffled null, because there is no
    alignment-dependent signal here to begin with. This is the honest,
    weaker-than-ideal case the README's Day 8 section names - the audit
    passing here does not by itself prove leak-freedom, only that nothing
    in this fixture looks like it is leaking; the synthetic test above is
    what proves the audit can tell the difference.
    """
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    result = run_audit(fixture, live=False, n_shuffles=200, seed=0)

    assert result.real_n >= 2
    assert result.passes
