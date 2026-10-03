import random
from datetime import date, datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import run, run_shuffle_audit, shuffle_timestamps
from sentiment.headline import Headline, read_csv
from sentiment.prices import Bar, save_fixture
from sentiment.vader_score import score_text


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_shuffle_timestamps_preserves_the_set_of_timestamps():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 21 + i, 2, 0, 0, tzinfo=timezone.utc))
        for i in range(5)
    ]
    rng = random.Random(0)
    shuffled = shuffle_timestamps(headlines, rng)

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)


def test_shuffle_timestamps_actually_reassigns_at_least_one_timestamp():
    # Not a hard guarantee for every seed/size (a shuffle can land on the
    # identity permutation), but true for this seed with 5 items - included
    # so a future refactor that silently makes shuffle_timestamps a no-op
    # (e.g. always returning the input unchanged) gets caught.
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 21 + i, 2, 0, 0, tzinfo=timezone.utc))
        for i in range(5)
    ]
    rng = random.Random(0)
    shuffled = shuffle_timestamps(headlines, rng)

    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_shuffle_timestamps_leaves_everything_but_published_at_alone():
    headlines = [make_headline("Infosys Share Price Highlights", "1", datetime(2026, 9, 21, 2, 0, 0, tzinfo=timezone.utc))]
    rng = random.Random(0)
    shuffled = shuffle_timestamps(headlines, rng)

    assert shuffled[0].title == headlines[0].title
    assert shuffled[0].link == headlines[0].link
    assert shuffled[0].source == headlines[0].source


def test_run_shuffle_audit_against_committed_fixture_finds_no_leakage_signal():
    # Day 5/6 already found the real pipeline's contemporaneous correlation
    # on this fixture is close to zero (r=-0.185). A real statistic with
    # nothing to leak should sit comfortably inside its own shuffled null -
    # a low p-value here would mean something changed underneath this test
    # worth looking at.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    headlines = read_csv(fixture)

    result = run_shuffle_audit(headlines, live=False, n_shuffles=100, seed=0)

    assert result.real_r is not None
    assert result.real_n == 23
    assert abs(result.real_r - (-0.185)) < 0.01
    assert result.p_value is not None
    assert result.p_value >= 0.05


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    exit_code = run(missing, live=False, n_shuffles=10, seed=0)
    assert exit_code == 1


def test_run_with_too_few_resolved_headlines_reports_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    from sentiment.headline import write_csv

    in_path = tmp_path / "raw.csv"
    write_csv(
        [make_headline("Cyient among 4 stocks showing White Marubozu Pattern", "1", datetime(2026, 9, 21, 2, 0, 0, tzinfo=timezone.utc))],
        in_path,
    )

    exit_code = run(in_path, live=False, n_shuffles=10, seed=0)
    assert exit_code == 1


def test_run_against_committed_fixture_succeeds(capsys):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    exit_code = run(fixture, live=False, n_shuffles=20, seed=0)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "permutation p-value" in out


# --- Synthetic case: proves the shuffle control has power -------------------
#
# The real fixture's correlation is already null (see above), so a pass
# there only shows the control agrees there is nothing to destroy - it does
# not show the control is *capable* of catching a real signal. This builds a
# small synthetic universe where each headline's title (hence its VADER
# compound, computed for real, not hand-picked) is deliberately paired with
# a price move that is an exact linear function of that compound - but only
# on the one calendar date each headline is genuinely published pre-open on.
# On every other date, the same ticker's return is an unrelated filler value.
# The real (unshuffled) pipeline sees r approx 1.0 by construction; shuffling
# published_at among the headlines reassigns which date each ticker's return
# is read from, which - whenever the shuffle isn't the identity - swaps the
# engineered return for an unrelated filler one and should collapse r.

SYNTHETIC_TITLES = [
    "Fortis Healthcare shares in focus as Supreme Court allows forensic audit to proceed",
    "Great Eastern Shipping shares gain 3% as Nomura retains Buy rating",
    "Jefferies names Max Financial as top pick, sees 41% upside",
    "PC Jeweller shares drop after weak quarterly results",
    "Suzlon Energy shares fall 2% on profit booking",
    "BSE shares rally on record trading volumes",
]
SYNTHETIC_TICKERS = ["FORTIS.NS", "GESHIP.NS", "MFSL.NS", "PCJEWELLER.NS", "SUZLON.NS", "BSE.NS"]
# 6 weekdays, Mon 21 Sep - Mon 28 Sep 2026 (skipping the weekend).
SYNTHETIC_DATES = [date(2026, 9, d) for d in (21, 22, 23, 24, 25, 28)]
SLOPE = 0.05
# Filler returns used on every "wrong" date, deliberately not ordered by
# compound so an accidental correlation doesn't sneak back in.
FILLER = [-0.02, 0.01, -0.015, 0.025, -0.005, 0.02]


def _build_synthetic_fixture(tmp_path: Path, monkeypatch) -> list[Headline]:
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    compounds = [score_text(title)["compound"] for title in SYNTHETIC_TITLES]
    assert len(set(compounds)) == len(compounds), "synthetic titles must score distinct compounds"

    headlines = [
        make_headline(title, str(i), datetime(d.year, d.month, d.day, 2, 0, 0, tzinfo=timezone.utc))
        for i, (title, d) in enumerate(zip(SYNTHETIC_TITLES, SYNTHETIC_DATES))
    ]

    for i, ticker in enumerate(SYNTHETIC_TICKERS):
        bars = []
        for j, d in enumerate(SYNTHETIC_DATES):
            if i == j:
                ret = SLOPE * compounds[i]
            else:
                ret = FILLER[(i + j) % len(FILLER)]
            bars.append(Bar(date=d, open=100.0, close=100.0 * (1 + ret)))
        save_fixture(ticker, bars)

    return headlines


def test_synthetic_engineered_correlation_is_strong_before_shuffling(tmp_path: Path, monkeypatch):
    headlines = _build_synthetic_fixture(tmp_path, monkeypatch)
    result = run_shuffle_audit(headlines, live=False, n_shuffles=1, seed=0)

    assert result.real_n == len(SYNTHETIC_TITLES)
    assert result.real_r is not None
    assert abs(result.real_r) > 0.95


def test_shuffle_control_destroys_the_synthetic_correlation(tmp_path: Path, monkeypatch):
    headlines = _build_synthetic_fixture(tmp_path, monkeypatch)
    result = run_shuffle_audit(headlines, live=False, n_shuffles=200, seed=0)

    assert result.real_r is not None and abs(result.real_r) > 0.95
    assert result.shuffled_rs
    mean_abs_shuffled = sum(abs(r) for r in result.shuffled_rs) / len(result.shuffled_rs)
    # The engineered signal only survives a shuffle that happens to be the
    # identity permutation (1 in 6! = 720 shuffles) - across 200 random
    # shuffles the average |r| should collapse well below the real value.
    assert mean_abs_shuffled < 0.5
    # And the permutation p-value should flag the real statistic as sitting
    # in the tail of its own shuffled null, not inside it.
    assert result.p_value is not None
    assert result.p_value < 0.1
