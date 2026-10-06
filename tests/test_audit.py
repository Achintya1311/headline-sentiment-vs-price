import random
from datetime import datetime, timezone
from pathlib import Path

import sentiment.audit as audit
import sentiment.prices as prices
from sentiment.headline import Headline, write_csv
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


def test_shuffle_published_at_permutes_times_but_keeps_every_other_field():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(10)
    ]
    rng = random.Random(0)

    shuffled = audit.shuffle_published_at(headlines, rng)

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # with 10 headlines a fixed, non-identity seed should not land on the
    # original order - otherwise the "shuffle" is not actually permuting.
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_contemporaneous_r_needs_at_least_two_rows():
    assert audit.contemporaneous_r([{"compound": 0.1, "contemporaneous_return": 0.01}]) is None

    rows = [
        {"compound": 0.0, "contemporaneous_return": 0.0},
        {"compound": 1.0, "contemporaneous_return": 1.0},
    ]
    result = audit.contemporaneous_r(rows)
    assert result is not None
    r_value, n = result
    assert n == 2
    assert r_value == 1.0


def test_lagged_r_skips_rows_with_no_lagged_return():
    rows = [
        {"compound": 0.0, "lagged_return": 0.0},
        {"compound": 1.0, "lagged_return": None},
        {"compound": 1.0, "lagged_return": 1.0},
    ]
    # only 2 rows actually have a lagged return -> n=2, not 3
    result = audit.lagged_r(rows)
    assert result is not None
    r_value, n = result
    assert n == 2
    assert r_value == 1.0


def test_event_diff_requires_both_groups_nonempty():
    threshold = 0.3
    event_diff = audit.make_event_diff(threshold)

    all_high = [{"compound": 0.9, "contemporaneous_return": 0.01}]
    assert event_diff(all_high) is None

    rows = [
        {"compound": 0.9, "contemporaneous_return": 0.02},
        {"compound": 0.1, "contemporaneous_return": 0.00},
    ]
    result = event_diff(rows)
    assert result is not None
    diff, n = result
    assert n == 2
    assert diff == 0.02 - 0.00


def test_run_shuffle_test_flags_a_real_pairing_no_shuffle_can_reproduce(monkeypatch):
    # Fake out build_rows_from_headlines so the test controls exactly what
    # the "observed" (real, correctly-aligned) call sees versus what every
    # shuffled call sees, without needing a fixture engineered to actually
    # leak. Observed: a perfect correlation. Every shuffle: a much weaker
    # one. A real leak should look exactly like this - a statistic no
    # scrambled pairing comes close to.
    call_count = {"n": 0}

    def fake_build_rows(headlines, live=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            rows = [{"compound": float(i), "contemporaneous_return": float(i)} for i in range(6)]
        else:
            scrambled_returns = [5.0, 3.0, 0.0, 4.0, 1.0, 2.0]
            rows = [{"compound": float(i), "contemporaneous_return": scrambled_returns[i]} for i in range(6)]
        return rows, []

    monkeypatch.setattr(audit, "build_rows_from_headlines", fake_build_rows)

    result = audit.run_shuffle_test([], "fake stat", audit.contemporaneous_r, n_shuffles=20, seed=0)

    assert result is not None
    assert result.observed == 1.0
    assert result.n_observed == 6
    assert result.n_null == 20
    assert result.p_value == 0.0
    assert result.leaks() is True


def test_run_shuffle_test_returns_none_when_observed_statistic_is_undefined(monkeypatch):
    monkeypatch.setattr(audit, "build_rows_from_headlines", lambda headlines, live=False: ([], []))

    result = audit.run_shuffle_test([], "fake stat", audit.contemporaneous_r, n_shuffles=10, seed=0)

    assert result is None


def test_run_against_committed_fixture_passes_with_no_leak_evidence():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = audit.run(fixture, n_shuffles=30, seed=0, event_threshold=0.3)

    # Day 5/6 already found this fixture's correlations indistinguishable
    # from zero - the shuffle control should agree that the real pairing is
    # not an outlier against random pairings.
    assert exit_code == 0


def test_run_reports_failure_when_nothing_has_enough_data(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=__import__("datetime").date(2026, 9, 28), open=100.0, close=105.0)])

    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            make_headline(
                "Infosys Share Price Highlights: Infosys Stock Price History",
                "1",
                datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            )
        ],
        in_path,
    )

    exit_code = audit.run(in_path, n_shuffles=10, seed=0, event_threshold=0.3)

    assert exit_code == 1


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = audit.run(missing, n_shuffles=10, seed=0, event_threshold=0.3)

    assert exit_code == 1
