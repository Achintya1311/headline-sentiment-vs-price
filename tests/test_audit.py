import random
from datetime import datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import (
    contemporaneous_r,
    run,
    run_shuffle_audit,
    shuffle_published_at,
)
from sentiment.headline import Headline
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


def test_shuffle_published_at_preserves_everything_but_the_timestamp():
    headlines = [
        make_headline("A", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("B", "2", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)),
        make_headline("C", "3", datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc)),
    ]
    shuffled = shuffle_published_at(headlines, random.Random(1))

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    # same multiset of timestamps, just reassigned.
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)


def test_shuffle_published_at_actually_moves_at_least_one_timestamp():
    # not a strict requirement of every single shuffle, but with 10 distinct
    # timestamps and a real RNG, an unperturbed shuffle would be suspicious.
    headlines = [
        make_headline(str(i), str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc)) for i in range(10)
    ]
    shuffled = shuffle_published_at(headlines, random.Random(42))
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_contemporaneous_r_needs_at_least_two_rows():
    assert contemporaneous_r([{"compound": 0.1, "contemporaneous_return": 0.01}]) is None
    assert contemporaneous_r([]) is None


def test_run_shuffle_audit_against_the_real_pipeline_passes(tmp_path: Path, monkeypatch):
    # Real build_rows_from_headlines, real market_hours alignment. The
    # headline text is written so VADER gives each one a clearly different
    # compound score (same ticker throughout, so resolution doesn't need its
    # own fixture juggling) - 3 published pre-open on the 28th (-> session
    # 28, a +10% day) and 3 published intraday on the 28th (-> rolled to the
    # 29th, a -18% day), which by this test's construction ties compound
    # sign to return sign in the *unshuffled* data. Shuffling published_at
    # should scatter that pairing back toward zero, because the only thing
    # connecting compound to contemporaneous_return here is which of those
    # two sessions a headline's timestamp lands it on.
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "PCJEWELLER.NS",
        [
            Bar(date=__import__("datetime").date(2026, 9, 28), open=100.0, close=110.0),
            Bar(date=__import__("datetime").date(2026, 9, 29), open=110.0, close=90.0),
        ],
    )

    headlines = [
        make_headline(
            "PC Jeweller shares surge as profit soars on strong festive demand",
            "1",
            datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc),  # pre-open -> session 28
        ),
        make_headline(
            "PC Jeweller shares rally on robust quarterly growth outlook",
            "2",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),  # pre-open -> session 28
        ),
        make_headline(
            "PC Jeweller shares climb after award winning performance",
            "3",
            datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc),  # pre-open -> session 28
        ),
        make_headline(
            "PC Jeweller shares plunge after fraud investigation disaster",
            "4",
            datetime(2026, 9, 28, 10, 0, 0, tzinfo=timezone.utc),  # intraday -> session 29
        ),
        make_headline(
            "PC Jeweller shares crash amid regulatory crackdown fears",
            "5",
            datetime(2026, 9, 28, 11, 0, 0, tzinfo=timezone.utc),  # intraday -> session 29
        ),
        make_headline(
            "PC Jeweller shares tumble on weak disappointing results",
            "6",
            datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc),  # intraday -> session 29
        ),
    ]

    result = run_shuffle_audit(headlines, n_shuffles=200, seed=0)

    assert result.real_n == 6
    assert abs(result.real_r) > 0.5  # the constructed pre-open/intraday split is a strong pairing
    assert result.shuffled_std > 0.0  # shuffling genuinely changed the alignment
    assert result.passes()  # ... and that pairing doesn't survive randomising it


def test_run_shuffle_audit_flags_a_build_function_that_ignores_the_shuffle():
    # Simulates the classic leak bug this audit exists to catch: a build
    # path that doesn't actually use the (shuffled) timestamp it was given,
    # so the "real" correlation survives every shuffle unchanged.
    headlines = [
        make_headline("A", "1", datetime(2026, 9, 28, h, 0, 0, tzinfo=timezone.utc)) for h in range(1, 9)
    ]
    compounds = [0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2]
    returns = [0.09, -0.08, 0.07, -0.06, 0.05, -0.04, 0.03, -0.02]  # perfectly tracks compound

    def leaky_build_fn(hs, live):
        # ignores hs entirely - returns the same compound/return pairing no
        # matter what order or timestamps the caller passed in.
        rows = [{"compound": c, "contemporaneous_return": r} for c, r in zip(compounds, returns)]
        return rows, []

    result = run_shuffle_audit(headlines, build_fn=leaky_build_fn, n_shuffles=50, seed=0)

    assert result.shuffled_std < 1e-9
    assert not result.passes()


def test_run_shuffle_audit_flags_a_leak_that_survives_shuffling_with_noise():
    # A subtler leak: the build function's output return is mostly driven by
    # compound directly (not by the shuffled timestamp), with a little
    # timestamp-dependent jitter mixed in so std > 0 but the null never gets
    # near zero - std==0 alone would not catch this one.
    headlines = [
        make_headline("A", str(i), datetime(2026, 9, 28, (i % 12) + 1, 0, 0, tzinfo=timezone.utc))
        for i in range(12)
    ]
    compounds = [(-1) ** i * (i + 1) / 12 for i in range(12)]

    def leaky_build_fn(hs, live):
        rows = []
        for h, c in zip(hs, compounds):
            jitter = (h.published_at.hour % 3) * 0.001  # tiny, timestamp-dependent
            rows.append({"compound": c, "contemporaneous_return": 2 * c + jitter})
        return rows, []

    result = run_shuffle_audit(headlines, build_fn=leaky_build_fn, n_shuffles=200, seed=0)

    assert result.shuffled_std > 0.0  # shuffling does perturb the output a little
    lo, hi = result.null_ci()
    assert lo > 0.0  # but the null distribution never reaches zero
    assert not result.passes()


def test_run_shuffle_audit_raises_when_too_few_resolved_headlines():
    headlines = [make_headline("A", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc))]

    def empty_build_fn(hs, live):
        return [], []

    try:
        run_shuffle_audit(headlines, build_fn=empty_build_fn, n_shuffles=5, seed=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_run_against_committed_fixture_passes_and_returns_zero():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    exit_code = run(fixture, n_shuffles=100, seed=0, live=False)
    assert exit_code == 0


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    exit_code = run(missing, n_shuffles=10, seed=0, live=False)
    assert exit_code == 1
