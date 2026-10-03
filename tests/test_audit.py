from datetime import datetime, timezone
from pathlib import Path

import sentiment.correlate as correlate
import sentiment.prices as prices
from sentiment.audit import run, run_shuffle_audit, shuffle_timestamps
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


def test_shuffle_timestamps_preserves_the_multiset_of_timestamps_and_headlines():
    headlines = [
        make_headline(f"headline {i}", str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(10)
    ]
    rng = __import__("random").Random(0)

    shuffled = shuffle_timestamps(headlines, rng)

    assert len(shuffled) == len(headlines)
    assert {h.published_at for h in shuffled} == {h.published_at for h in headlines}
    # every row keeps its own title/link - only published_at moves
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    # with 10 distinct timestamps, a real shuffle should not land back on the
    # identity permutation - this seed doesn't, which is what makes it useful
    # as a regression check that shuffling actually does something.
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_shuffle_timestamps_changes_session_dates_for_the_real_fixture():
    # Guards against a no-op shuffle bug: if shuffling the timestamps didn't
    # actually change anything downstream, the audit would be vacuous - it
    # would "pass" no matter what, by always comparing the real result to
    # itself.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    headlines = read_csv(fixture)
    rng = __import__("random").Random(0)

    shuffled = shuffle_timestamps(headlines, rng)

    from sentiment.market_hours import align_headline

    real_sessions = [align_headline(h.published_at).session_date for h in headlines]
    shuffled_sessions = [align_headline(h.published_at).session_date for h in shuffled]
    assert real_sessions != shuffled_sessions


def test_run_shuffle_audit_too_few_resolved_headlines_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = [
        make_headline(
            "LTIMindtree Share Price Highlights: LTIMindtree Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    ]

    import pytest

    with pytest.raises(ValueError, match="not enough"):
        run_shuffle_audit(headlines, n_shuffles=10, seed=0)


def test_run_shuffle_audit_against_the_committed_fixture_matches_known_correlation():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    headlines = read_csv(fixture)

    result = run_shuffle_audit(headlines, n_shuffles=200, seed=0)

    # README's Day 5 Findings: contemporaneous r=-0.185, 95% CI [-0.555,
    # +0.246], n=23 - the audit must reuse the exact same pipeline, not a
    # reimplementation that could silently drift from it.
    assert result.real.n == 23
    assert result.real.r == correlate_known_r(fixture)
    assert result.n_valid > 0
    # the real CI already includes zero (no signal to begin with), so the
    # honest expectation is a low false-positive rate under shuffling too -
    # not dramatically above the ~5% chance alone would produce.
    assert result.false_positive_rate <= 0.15


def correlate_known_r(fixture: Path) -> float:
    rows, _ = correlate.build_rows(fixture, live=False)
    from sentiment.stats import pearson_r

    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


def test_run_cli_against_the_committed_fixture_passes(capsys):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, n_shuffles=200, seed=0)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "PASS" in out
    assert "r=-0.185" in out


def test_run_missing_input_reports_failure(tmp_path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, n_shuffles=10, seed=0)

    assert exit_code == 1


def test_run_shuffle_audit_flags_a_pipeline_that_ignores_real_timing(tmp_path, monkeypatch):
    """The mechanism test: construct a pipeline where a headline's return is
    wired to which company it is about, not to the (correctly computed)
    session_date - the exact leak this audit exists to catch. Monkeypatching
    ``correlate.bar_on`` to ignore the session_date it's given simulates a
    bug where the return a headline gets never actually depended on real
    timing. Shuffling published_at should then change nothing, and the
    false-positive rate should come back at (or near) 100%, not ~5%."""
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    headlines = [
        make_headline("Fortis Healthcare shares in focus", "1", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)),
        make_headline("Great Eastern Shipping gains", "2", datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc)),
        make_headline("Jefferies names Max Financial a top pick", "3", datetime(2026, 9, 28, 4, 0, 0, tzinfo=timezone.utc)),
        make_headline("PC Jeweller shares rise 4%", "4", datetime(2026, 9, 29, 2, 0, 0, tzinfo=timezone.utc)),
        make_headline("BSE shares slide", "5", datetime(2026, 9, 29, 3, 0, 0, tzinfo=timezone.utc)),
        make_headline("Suzlon Energy shares fall 2%", "6", datetime(2026, 9, 29, 4, 0, 0, tzinfo=timezone.utc)),
    ]

    # Compound scores monotonically spread so a real, timing-correct
    # relationship would be a clean, strong correlation - not relying on
    # VADER's own scoring of these exact strings.
    compounds = {"1": 0.9, "2": 0.6, "3": 0.3, "4": -0.3, "5": -0.6, "6": -0.9}
    monkeypatch.setattr(
        correlate, "score_headline", lambda h: type("Scored", (), {"compound": compounds[h.link]})()
    )

    # Returns keyed by ticker alone, matching the compound ordering exactly -
    # and bar_on ignores the date it's asked for, so no matter what session
    # a shuffled timestamp aligns a headline to, it gets this same ticker's
    # fixed bar back. This is the bug: the return never depended on timing.
    ticker_return = {
        "FORTIS.NS": 0.09,
        "GESHIP.NS": 0.06,
        "MFSL.NS": 0.03,
        "PCJEWELLER.NS": -0.03,
        "BSE.NS": -0.06,
        "SUZLON.NS": -0.09,
    }
    fixed_bar_by_ticker = {t: Bar(date=__import__("datetime").date(2026, 9, 28), open=1.0, close=1.0 + r) for t, r in ticker_return.items()}

    monkeypatch.setattr(correlate, "load_bars", lambda ticker, live=False: [fixed_bar_by_ticker[ticker]])
    monkeypatch.setattr(correlate, "bar_on", lambda bars, on: bars[0])
    monkeypatch.setattr(correlate, "next_session_bar", lambda bars, after: None)

    result = run_shuffle_audit(headlines, n_shuffles=50, seed=0)

    assert result.real.ci_low > 0 or result.real.ci_high < 0  # the real relationship is "significant"
    assert result.false_positive_rate >= 0.9  # and shuffling timing did essentially nothing to it
