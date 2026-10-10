import random
from datetime import datetime, timezone
from pathlib import Path

from sentiment.audit import (
    permutation_audit,
    run,
    shuffle_timestamps,
    synthetic_signal_correlation,
)
from sentiment.headline import Headline

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_shuffle_timestamps_is_a_permutation_of_the_same_values():
    headlines = [
        make_headline(f"headline {i}", str(i), datetime(2026, 9, 28, 1, i, 0, tzinfo=timezone.utc))
        for i in range(10)
    ]
    shuffled = shuffle_timestamps(headlines, random.Random(0))

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # with 10 distinct timestamps and a fixed seed, at least one headline
    # must land on a different timestamp than it started with.
    assert any(a.published_at != b.published_at for a, b in zip(headlines, shuffled))


def test_shuffle_timestamps_leaves_content_fields_untouched():
    headlines = [
        make_headline("Infosys beats estimates", "link-1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("Wipro misses guidance", "link-2", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)),
    ]
    shuffled = shuffle_timestamps(headlines, random.Random(1))

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    assert [h.source for h in shuffled] == [h.source for h in headlines]


def test_synthetic_signal_is_recovered_almost_exactly_without_shuffling():
    # a real, deterministic sentiment -> next-session-return relationship,
    # correctly aligned - the audit's positive control for "there is
    # something here for a leak-free pipeline to find".
    r = synthetic_signal_correlation(shuffle=False, seed=0)
    assert r > 0.999


def test_synthetic_signal_collapses_when_timestamps_are_shuffled():
    # same dataset, but each headline's content is now paired with a
    # different day's true return via a shuffled publish time. If this
    # stayed near 1.0, align_headline (or the pairing code above it) would
    # be leaking which day the news really arrived on - exactly the bug
    # NEXT_STEPS.md's "Done when" section exists to catch.
    r = synthetic_signal_correlation(shuffle=True, seed=0)
    assert abs(r) < 0.5


def test_permutation_audit_against_real_fixture_finds_no_leak_signal():
    # the actual CI gate: on the committed fixture, the real contemporaneous
    # correlation must not be an outlier against a null distribution built
    # by shuffling timestamps. A result here failing (p < 0.05, suggesting
    # the real number is weirdly extreme relative to randomly-timed noise)
    # would mean something in resolve/align/price is leaking and needs to be
    # fixed before any of Day 5-7's numbers can be trusted.
    from sentiment.headline import read_csv

    headlines = read_csv(FIXTURE)
    result = permutation_audit(headlines, n_shuffles=150, seed=0)

    assert result.real_r is not None
    assert result.real_n == 23
    assert len(result.null_rs) == 150
    assert result.p_value is not None
    assert result.p_value >= 0.05, (
        f"real r={result.real_r:+.3f} is an outlier against the shuffled-timestamp null "
        f"(p={result.p_value:.3f}) - investigate for a timestamp leak before trusting this result"
    )


def test_permutation_audit_skips_shuffles_that_lose_all_priced_rows(monkeypatch):
    # a shuffle that rolls every headline's session past the edge of the
    # one-month price fixture must be dropped, not counted as r=0 - that
    # would quietly bias the null distribution towards zero.
    import sentiment.audit as audit_module

    headlines = [
        make_headline("Infosys Share Price Highlights: Infosys Stock Price History", "1",
                      datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)),
        make_headline("Wipro Share Price Highlights: Wipro Stock Price History", "2",
                      datetime(2026, 9, 28, 2, 30, 0, tzinfo=timezone.utc)),
    ]

    calls = {"n": 0}
    real_fn = audit_module._contemporaneous_r

    def flaky(hs, live):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_fn(hs, live)
        return None, 0  # every shuffle "fails" to price

    monkeypatch.setattr(audit_module, "_contemporaneous_r", flaky)
    result = permutation_audit(headlines, n_shuffles=5, seed=0)

    assert result.null_rs == []
    assert result.p_value is None


def test_run_against_committed_fixture_prints_a_verdict_and_succeeds(capsys):
    exit_code = run(FIXTURE, n_shuffles=20, seed=0, live=False)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "real (unshuffled) contemporaneous" in out
    assert "permutation p-value" in out
    assert "positive control" in out


def test_run_missing_input_reports_failure(tmp_path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, n_shuffles=20, seed=0, live=False)

    assert exit_code == 1
