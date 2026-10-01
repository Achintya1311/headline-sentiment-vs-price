from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from sentiment.audit import run, run_shuffle_audit, shuffle_published_at
from sentiment.headline import Headline
from sentiment.market_hours import Timing, align_headline

IST = ZoneInfo("Asia/Kolkata")

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"


def make_headline(i: int, published_at: datetime) -> Headline:
    return Headline(
        source="synthetic",
        title=f"synthetic headline {i}",
        link=f"link-{i}",
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=published_at,
    )


def make_intraday_headlines(dates: list[date]) -> list[Headline]:
    """Headlines published mid-session (14:00 IST) on each of ``dates`` - under
    Day 4's real leak-free alignment these all roll forward to the *next*
    trading day, never the same day."""
    headlines = []
    for i, d in enumerate(dates):
        local = datetime(d.year, d.month, d.day, 14, 0, 0, tzinfo=IST)
        headlines.append(make_headline(i, local.astimezone(timezone.utc)))
    return headlines


def make_leaky_build_rows(headlines: list[Headline]):
    """A deliberately broken headline-to-row builder: pairs each headline
    with its *own calendar day's* return - exactly the look-ahead mistake
    Day 4's ``align_headline`` exists to prevent (a mid-session headline
    getting credited with a return that partly happened before it existed).

    Each headline's ``compound`` is fixed to its original identity (via
    ``title``, looked up in a closure over the *original*, unshuffled
    headlines) and does not change when ``published_at`` is shuffled - same
    as the real pipeline, where VADER scores headline text, never a date.
    Only the *return* side depends on the (possibly shuffled) timestamp,
    through the leaky same-day lookup.
    """
    dates = [h.published_at.astimezone(IST).date() for h in headlines]
    values = [-0.06 + 0.01 * i for i in range(len(headlines))]
    synthetic_returns = dict(zip(dates, values))
    compound_by_title = {h.title: v * 10 for h, v in zip(headlines, values)}

    def leaky_build_rows(hs: list[Headline], live: bool = False):
        rows = []
        for h in hs:
            d = h.published_at.astimezone(IST).date()
            ret = synthetic_returns.get(d)
            if ret is None:
                continue
            rows.append({"compound": compound_by_title[h.title], "contemporaneous_return": ret})
        return rows, []

    return leaky_build_rows


def test_shuffle_published_at_keeps_the_same_multiset_of_timestamps():
    dates = [date(2026, 9, d) for d in range(1, 21)]
    headlines = make_intraday_headlines(dates)

    import random

    shuffled = shuffle_published_at(headlines, random.Random(0))

    assert len(shuffled) == len(headlines)
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # titles/links stay attached to their original position - only timing moves
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]


def test_shuffle_published_at_actually_reassigns_timestamps():
    dates = [date(2026, 9, d) for d in range(1, 21)]
    headlines = make_intraday_headlines(dates)

    import random

    shuffled = shuffle_published_at(headlines, random.Random(0))

    # With 20 headlines, a seeded shuffle landing on the identity permutation
    # is astronomically unlikely - at least one headline's timing must move.
    assert any(a.published_at != b.published_at for a, b in zip(headlines, shuffled))


def test_shuffle_published_at_keeps_published_raw_consistent_with_published_at():
    dates = [date(2026, 9, d) for d in range(1, 6)]
    headlines = make_intraday_headlines(dates)

    import random

    shuffled = shuffle_published_at(headlines, random.Random(1))

    for h in shuffled:
        assert h.published_raw == h.published_at.isoformat()


def test_run_shuffle_audit_inconclusive_with_too_few_rows():
    headlines = make_intraday_headlines([date(2026, 9, 21)])

    def one_row_builder(hs, live=False):
        return [{"compound": 0.1, "contemporaneous_return": 0.01}], []

    result = run_shuffle_audit(headlines, build_rows_fn=one_row_builder, n_shuffles=10, seed=0)

    assert result.inconclusive
    assert result.passed is None
    assert result.p_value is None


def test_run_shuffle_audit_passes_on_the_real_committed_fixture():
    # The honest Day 5 finding this audit is meant to stress-test: a weak,
    # not-significant contemporaneous correlation (r=-0.185, n=23, 95% CI
    # crossing zero - see README). A real, timing-dependent effect this
    # small should not stand out against a shuffled-timestamp null, and it
    # doesn't: the real pairing is unremarkable next to 300 random re-timings.
    from sentiment.headline import read_csv

    headlines = read_csv(FIXTURE)
    result = run_shuffle_audit(headlines, n_shuffles=300, seed=0)

    assert not result.inconclusive
    assert result.real_n == 23
    assert result.real_r == pytest.approx(-0.185, abs=0.001)
    assert result.passed is True
    assert result.p_value > 0.05


def test_run_shuffle_audit_detects_a_deliberately_leaky_alignment():
    # Proof the audit has power, not just a test that always passes because
    # the real fixture has nothing to lose. These headlines are published
    # intraday (14:00 IST) - Day 4's real alignment would roll every one of
    # them to the *next* trading day - but ``leaky_build_rows`` ignores that
    # and pairs each with its own calendar day's return instead, the exact
    # look-ahead bug Day 4 exists to prevent. Built so the un-shuffled
    # pairing recovers a planted signal perfectly (r=1.0); a genuine
    # timing-independent artifact like that should not survive the shuffle.
    dates = [date(2026, 9, d) for d in [14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 28, 29]]
    headlines = make_intraday_headlines(dates)
    leaky_build_rows = make_leaky_build_rows(headlines)

    result = run_shuffle_audit(headlines, build_rows_fn=leaky_build_rows, n_shuffles=1000, seed=0)

    assert result.real_r == pytest.approx(1.0)
    assert result.passed is False
    assert result.p_value <= 0.05
    # the shuffled null is centered well below the planted, leak-inflated r
    assert result.mean_abs_shuffled_r < 0.4


def test_the_leaky_scenarios_headlines_would_not_replicate_under_real_alignment():
    # Sanity check that the synthetic leak above is actually exercising the
    # mistake Day 4 prevents, not some unrelated artifact: every headline in
    # it is intraday, and the real ``align_headline`` rolls every one of
    # them to the *next* trading day, not the same-day pairing the leaky
    # builder uses.
    dates = [date(2026, 9, d) for d in [14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 28, 29]]
    headlines = make_intraday_headlines(dates)

    for h, d in zip(headlines, dates):
        alignment = align_headline(h.published_at)
        assert alignment.timing is Timing.INTRADAY
        assert alignment.session_date > d
        assert alignment.leak_free()


def test_run_cli_passes_against_the_real_committed_fixture(capsys):
    exit_code = run(FIXTURE, n_shuffles=200, seed=0, live=False)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS" in captured.out


def test_run_cli_missing_input_reports_failure(tmp_path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, n_shuffles=10, seed=0, live=False)

    assert exit_code == 1
