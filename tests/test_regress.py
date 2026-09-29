import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.headline import Headline, write_csv
from sentiment.prices import Bar, save_fixture
from sentiment.regress import fit_and_evaluate, run, time_split, usable_rows


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_usable_rows_keeps_only_rows_with_a_lagged_return():
    rows = [
        {"lagged_return": 0.01},
        {"lagged_return": None},
        {"lagged_return": -0.02},
    ]
    assert usable_rows(rows) == [{"lagged_return": 0.01}, {"lagged_return": -0.02}]


def test_time_split_keeps_chronological_order_and_respects_fraction():
    rows = [{"i": i} for i in range(10)]
    train, test = time_split(rows, train_frac=0.7)
    assert train == rows[:7]
    assert test == rows[7:]


def test_time_split_enforces_a_minimum_train_size():
    rows = [{"i": i} for i in range(3)]
    train, test = time_split(rows, train_frac=0.1)
    assert len(train) >= 2  # MIN_TRAIN, even though 0.1 * 3 rounds to 0
    assert len(test) >= 1


def test_time_split_raises_when_too_few_usable_rows():
    with pytest.raises(ValueError):
        time_split([{"i": 0}, {"i": 1}], train_frac=0.7)  # 1 test row for 0.7*2 train


def test_fit_and_evaluate_recovers_a_real_out_of_sample_relationship():
    # Synthetic but noise-free: next_day_return = 0.01 * compound exactly,
    # so a genuine relationship should show up out of sample too, unlike the
    # committed fixture's degenerate case below.
    train = [{"compound": x, "lagged_return": 0.01 * x} for x in [0.1, 0.2, 0.3, 0.4, 0.5]]
    test = [{"compound": x, "lagged_return": 0.01 * x} for x in [0.6, 0.7]]

    result = fit_and_evaluate(train, test)

    assert result.slope == pytest.approx(0.01, abs=1e-9)
    assert result.test_r2 == pytest.approx(1.0)
    assert result.test_mae < result.baseline_test_mae


def test_fit_and_evaluate_zero_variance_predictor_gives_zero_oos_r2_not_a_crash():
    # Exactly the committed fixture's own finding (see README Day 6): every
    # usable headline shares the same compound score, so there is no slope
    # to fit and the out-of-sample R^2 must come back defined (0.0), not NaN.
    train = [{"compound": 0.296, "lagged_return": r} for r in [0.01, -0.02, 0.03, -0.01, 0.02]]
    test = [{"compound": 0.296, "lagged_return": r} for r in [-0.01, 0.04]]

    result = fit_and_evaluate(train, test)

    assert result.slope == 0.0
    assert result.test_r2 == pytest.approx(0.0)
    assert result.test_mae == pytest.approx(result.baseline_test_mae)


def test_run_writes_output_csv_with_train_and_test_splits(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    import datetime as dt

    bars = [
        Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0),
        Bar(date=dt.date(2026, 9, 29), open=105.0, close=103.0),
    ]
    save_fixture("INFY.NS", bars)

    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            f"link-{i}",
            datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc),
        )
        for i in range(4)
    ]
    in_path = tmp_path / "raw.csv"
    write_csv(headlines, in_path)
    out_path = tmp_path / "regression.csv"

    exit_code = run(in_path, out_path, live=False, train_frac=0.5)

    assert exit_code == 0
    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert [r["split"] for r in rows] == ["train", "train", "test", "test"]


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    out_path = tmp_path / "regression.csv"

    exit_code = run(missing, out_path, live=False, train_frac=0.7)

    assert exit_code == 1
    assert not out_path.exists()


def test_run_reports_failure_rather_than_crashing_when_too_few_usable_rows(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    import datetime as dt

    save_fixture("INFY.NS", [Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0)])  # no next session
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
    out_path = tmp_path / "regression.csv"

    exit_code = run(in_path, out_path, live=False, train_frac=0.7)

    assert exit_code == 1
    assert not out_path.exists()


def test_run_against_committed_fixture_matches_the_documented_degenerate_case(tmp_path: Path):
    # The real, honest Day 6 finding (see README): all 15 headlines with a
    # next-day return share the identical VADER compound score, so the fit
    # is mechanically valid but has nothing to explain, in train or test.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    out_path = tmp_path / "regression.csv"

    exit_code = run(fixture, out_path, live=False, train_frac=0.7)

    assert exit_code == 0
    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 15
    assert sum(1 for r in rows if r["split"] == "train") == 10
    assert sum(1 for r in rows if r["split"] == "test") == 5
