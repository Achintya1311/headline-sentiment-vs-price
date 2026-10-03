import random
from datetime import datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import contemporaneous_r, run, run_audit, shuffle_published_at
from sentiment.headline import Headline, write_csv
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


def test_shuffle_preserves_headline_identity_and_timestamp_multiset():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 1 + i, 2, 0, 0, tzinfo=timezone.utc))
        for i in range(6)
    ]
    rng = random.Random(0)

    shuffled = shuffle_published_at(headlines, rng)

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # a permutation, not an identity mapping, for this seed and n
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_shuffle_is_deterministic_for_a_given_seed():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 1 + i, 2, 0, 0, tzinfo=timezone.utc))
        for i in range(5)
    ]

    first = shuffle_published_at(headlines, random.Random(42))
    second = shuffle_published_at(headlines, random.Random(42))

    assert [h.published_at for h in first] == [h.published_at for h in second]


def test_contemporaneous_r_degrades_gracefully_below_two_rows(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    in_path = tmp_path / "raw.csv"
    write_csv([], in_path)

    r, n = contemporaneous_r(in_path)

    assert r == 0.0
    assert n == 0


def test_run_audit_flags_a_genuine_timestamp_dependent_correlation(tmp_path: Path, monkeypatch):
    """Proves the audit has power: when sentiment genuinely only lines up
    with returns through the *correct* headline-to-session pairing, shuffling
    timestamps must destroy it, and the audit must say so (not just always
    report "pass" the way a check with no teeth would on this repo's actual,
    already-null result).

    Six headlines, all matching ``sentiment.tickers``' "HDFC Bank " rule so
    they all resolve to the same ticker, each published pre-open on a
    distinct trading day whose fixture bar return matches the headline's
    own wording (clearly positive or clearly negative). Compound and
    same-day return correlate strongly only because each headline sits on
    *its own* correct date; reassigning dates at random (what the shuffle
    does) pairs each headline's wording with an unrelated day's return.
    """
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    titles = [
        "HDFC Bank shares surge as profit beats estimates, outlook excellent",
        "HDFC Bank stock jumps on strong quarter, analysts delighted",
        "HDFC Bank shares rise as results impress investors",
        "HDFC Bank shares fall as results disappoint, outlook weak",
        "HDFC Bank stock slumps on dismal quarter, analysts alarmed",
        "HDFC Bank shares tumble as investors flee on terrible guidance",
    ]
    # Mon 7, Tue 8, Wed 9, Thu 10, Fri 11, then skip the weekend -> Mon 14
    dates = [
        __import__("datetime").date(2026, 9, 7),
        __import__("datetime").date(2026, 9, 8),
        __import__("datetime").date(2026, 9, 9),
        __import__("datetime").date(2026, 9, 10),
        __import__("datetime").date(2026, 9, 11),
        __import__("datetime").date(2026, 9, 14),
    ]

    compounds = [score_text(t)["compound"] for t in titles]
    assert compounds[0] > 0 and compounds[1] > 0 and compounds[2] > 0
    assert compounds[3] < 0 and compounds[4] < 0 and compounds[5] < 0

    # Build one bar per date whose sign matches that date's headline's
    # sentiment, ranked by compound so the real-timestamp correlation is
    # strong and monotonic rather than merely same-signed.
    order = sorted(range(6), key=lambda i: compounds[i])
    returns_by_rank = [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]
    returns = [0.0] * 6
    for rank, idx in enumerate(order):
        returns[idx] = returns_by_rank[rank]

    bars = []
    for d, ret in zip(dates, returns):
        open_px = 100.0
        bars.append(Bar(date=d, open=open_px, close=open_px * (1 + ret)))
    save_fixture("HDFCBANK.NS", bars)

    headlines = [
        make_headline(title, str(i), datetime(d.year, d.month, d.day, 2, 0, 0, tzinfo=timezone.utc))
        for i, (title, d) in enumerate(zip(titles, dates))
    ]
    in_path = tmp_path / "raw.csv"
    write_csv(headlines, in_path)

    result = run_audit(in_path, n_shuffles=200, seed=0)

    assert result.real_n == 6
    assert result.real_r > 0.9  # near-perfect by construction when correctly aligned
    assert not result.passes()  # the audit must flag this as inconsistent with the shuffled null
    assert abs(result.shuffled_mean) < abs(result.real_r)


def test_run_audit_does_not_flag_the_real_committed_fixture():
    """The honest current state (see README Day 5-8 Findings): the real
    pipeline's contemporaneous correlation is already a null result, and
    this shuffle control finds nothing in it that looks like a leak."""
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    result = run_audit(fixture, n_shuffles=100, seed=0)

    assert result.real_n == 23
    assert result.passes()


def test_cli_run_exits_zero_on_the_real_committed_fixture(tmp_path: Path, capsys):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, n_shuffles=50, seed=0, live=False)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "PASS" in out


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, n_shuffles=50, seed=0, live=False)

    assert exit_code == 1
