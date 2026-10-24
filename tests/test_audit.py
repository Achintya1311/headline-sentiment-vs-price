from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import sentiment.audit as audit
import sentiment.prices as prices
from sentiment.correlate import build_rows
from sentiment.headline import Headline, read_csv, write_csv
from sentiment.prices import Bar, save_fixture
from sentiment.stats import pearson_r


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_run_audit_too_few_resolved_headlines_raises(tmp_path):
    in_path = tmp_path / "raw.csv"
    write_csv(
        [make_headline("Nothing in the lookup table", "1", datetime(2026, 9, 28, 2, 0, tzinfo=timezone.utc))],
        in_path,
    )
    with pytest.raises(ValueError):
        audit.run_audit(read_csv(in_path))


def test_run_audit_against_real_fixture_matches_correlate_and_finds_no_leak():
    """Cross-check against the real committed fixture: the audit's own
    true-timestamp pass must reproduce exactly the contemporaneous r Day
    5's ``sentiment.correlate`` reports (same alignment, same prices, same
    headlines), and - the actual Day 8 deliverable - running the shuffle
    control on it must not flag a leak. A small n_perm keeps this fast;
    the committed fixture's real result is this repo's live correctness
    gate, so it is checked here, not only printed by the CLI."""
    rows, _, _ = build_rows(audit.DEFAULT_IN)
    expected_r = pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])

    headlines = read_csv(audit.DEFAULT_IN)
    result = audit.run_audit(headlines, n_perm=300, seed=0)

    assert result.real_r == pytest.approx(expected_r)
    assert result.n_real == len(rows)
    assert result.n_perm > 0
    assert 0.0 < result.p_value <= 1.0
    # The documented "done when": the real pairing must not look like a
    # leak against a null built from re-timed re-pairings.
    assert result.p_value >= audit.SIGNIFICANCE


def test_shuffle_destroys_a_true_timestamp_dependent_signal(tmp_path, monkeypatch):
    """Prove the audit has power: without this, a control that always says
    'no leak' - as it does on this repo's own null fixture above - would be
    vacuous. Build a case where the contemporaneous return equals the
    headline's compound *only* when each headline is paired with its own
    true publish day, monkeypatch ticker resolution and scoring to control
    both ends precisely, and confirm the shuffle control flags the true
    pairing as a significant outlier against the shuffled null."""
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    compounds = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]
    titles = [f"Synthetic headline {i}" for i in range(len(compounds))]
    compound_by_title = dict(zip(titles, compounds))

    ist_two_am = timezone(timedelta(hours=5, minutes=30))
    base_day = datetime(2026, 9, 7, 2, 0, 0, tzinfo=ist_two_am)  # Monday, pre-open
    weekdays = []
    d = base_day
    while len(weekdays) < len(compounds):
        if d.weekday() < 5:
            weekdays.append(d)
        d += timedelta(days=1)

    headlines = [make_headline(title, str(i), ts) for i, (title, ts) in enumerate(zip(titles, weekdays))]

    bars = [
        Bar(date=ts.date(), open=100.0, close=100.0 * (1 + compound))
        for ts, compound in zip(weekdays, compounds)
    ]
    save_fixture("FAKE.NS", bars)

    monkeypatch.setattr(audit, "resolve", lambda title: ("Fake Co", "FAKE.NS"))
    monkeypatch.setattr(
        audit, "score_headline", lambda h: SimpleNamespace(compound=compound_by_title[h.title])
    )

    result = audit.run_audit(headlines, n_perm=300, seed=1)

    assert result.real_r == pytest.approx(1.0)
    assert result.n_real == len(compounds)
    # Only the identity re-pairing can reproduce r=1 on this strictly
    # monotonic sequence, so the real pairing sits in the extreme tail of
    # the shuffled null almost by construction - exactly what a real
    # look-ahead leak would look like, and exactly what this control exists
    # to catch.
    assert result.p_value < audit.SIGNIFICANCE
    # The shuffled null itself is not just "also near 1" - shuffling really
    # did decouple compound from return for most re-pairings.
    assert abs(result.perm_mean) < 0.3


def test_cli_run_reports_pass_on_real_fixture(capsys):
    exit_code = audit.run(audit.DEFAULT_IN, live=False, n_perm=200, seed=0)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "no evidence of timestamp-dependent leakage" in captured.out
    assert "p-value" in captured.out


def test_cli_run_missing_input_file_errors(tmp_path, capsys):
    exit_code = audit.run(tmp_path / "missing.csv", live=False, n_perm=10, seed=0)
    assert exit_code == 1
    assert "no headlines file" in capsys.readouterr().err
