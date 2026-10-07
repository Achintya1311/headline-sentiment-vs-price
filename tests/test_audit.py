import random
from datetime import date, datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import PermutationResult, permutation_test, run, shuffle_headline_times
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


def test_shuffle_preserves_the_timestamp_multiset_and_everything_else():
    headlines = [
        make_headline("Infosys Share Price Highlights", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("Wipro Share Price Highlights", "2", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)),
        make_headline("Tech Mahindra Share Price Highlights", "3", datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc)),
    ]

    shuffled = shuffle_headline_times(headlines, random.Random(0))

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # titles/links stay with their original headline - only the timestamp moves
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    # this seed actually moves at least one timestamp - a no-op shuffle would
    # silently defeat the whole audit (see sentiment/audit.py's docstring)
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_shuffle_is_deterministic_given_the_same_seed():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(10)
    ]

    a = shuffle_headline_times(headlines, random.Random(42))
    b = shuffle_headline_times(headlines, random.Random(42))

    assert [h.published_at for h in a] == [h.published_at for h in b]


def test_permutation_test_returns_none_when_nothing_resolves(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = [
        make_headline(
            "Cyient among 4 stocks showing White Marubozu Pattern",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    ]

    assert permutation_test(headlines, n_perm=50) is None


def test_permutation_test_on_the_real_fixture_does_not_look_like_a_leak():
    # The real, committed 50-headline fixture: Day 5 already found a null
    # contemporaneous correlation (r=-0.185, 95% CI containing zero). The
    # leakage audit should agree there's nothing here that depends on
    # getting the timestamp right - shuffling it should not make the real
    # result look like an outlier.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    from sentiment.headline import read_csv

    headlines = read_csv(fixture)
    result = permutation_test(headlines, n_perm=500, seed=0)

    assert result is not None
    assert result.real_n == 23
    assert result.p_value > 0.05


def test_permutation_test_flags_an_engineered_strong_relationship(tmp_path: Path, monkeypatch):
    # Positive control: if sentiment and contemporaneous return really were
    # tightly (and only) related through the *correct* timestamp alignment,
    # the shuffle control must be able to tell - otherwise the audit has no
    # power and a "PASS" on the real fixture would mean nothing. Eight
    # headlines, four resolved on Monday's session and four on Tuesday's
    # (via a pre-open vs. intraday publish time), each engineered so its
    # VADER compound sign matches the session it actually resolves to and
    # nothing else - shuffling which timestamp a headline carries will often
    # swap a headline onto the *other* day's unrelated/flat return.
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    mon, tue = date(2026, 9, 28), date(2026, 9, 29)
    scraped = datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc)

    def bars(mon_return: float, tue_return: float) -> list[Bar]:
        mon_open = 100.0
        mon_close = mon_open * (1 + mon_return)
        tue_close = mon_close * (1 + tue_return)
        return [Bar(date=mon, open=mon_open, close=mon_close), Bar(date=tue, open=mon_close, close=tue_close)]

    positive_words = "excellent amazing wonderful fantastic surge record profit"
    negative_words = "terrible horrible awful disaster crash plunge slump"

    # (company, ticker, mon_return, tue_return, published_at, words)
    rigged = [
        ("Infosys", "INFY.NS", 0.08, 0.0, datetime(2026, 9, 28, 2, 0, tzinfo=timezone.utc), positive_words),
        ("Wipro", "WIPRO.NS", 0.08, 0.0, datetime(2026, 9, 28, 2, 10, tzinfo=timezone.utc), positive_words),
        ("HCL Tech", "HCLTECH.NS", -0.08, 0.0, datetime(2026, 9, 28, 2, 20, tzinfo=timezone.utc), negative_words),
        ("Sun Pharma", "SUNPHARMA.NS", -0.08, 0.0, datetime(2026, 9, 28, 2, 30, tzinfo=timezone.utc), negative_words),
        ("Bharti Airtel", "BHARTIARTL.NS", 0.0, 0.08, datetime(2026, 9, 28, 10, 0, tzinfo=timezone.utc), positive_words),
        ("Tech Mahindra", "TECHM.NS", 0.0, 0.08, datetime(2026, 9, 28, 10, 10, tzinfo=timezone.utc), positive_words),
        ("SBI Life", "SBILIFE.NS", 0.0, -0.08, datetime(2026, 9, 28, 10, 20, tzinfo=timezone.utc), negative_words),
        ("Nestle India", "NESTLEIND.NS", 0.0, -0.08, datetime(2026, 9, 28, 10, 30, tzinfo=timezone.utc), negative_words),
    ]

    headlines = []
    for company, ticker, mon_ret, tue_ret, published_at, words in rigged:
        save_fixture(ticker, bars(mon_ret, tue_ret))
        title = f"{company} Share Price Highlights: {words}"
        headlines.append(make_headline(title, ticker, published_at))

    result = permutation_test(headlines, n_perm=1000, seed=0)

    assert result is not None
    assert result.real_n == 8
    assert abs(result.real_r) > 0.9  # engineered to be (almost) perfect
    assert result.p_value <= 0.05  # the shuffle control catches it

    in_path = tmp_path / "raw.csv"
    from sentiment.headline import write_csv

    write_csv(headlines, in_path)
    exit_code = run(in_path, n_perm=1000, seed=0, alpha=0.05, live=False)
    assert exit_code == 1  # "FAIL" - a real pipeline this leaky should not pass CI


def test_run_against_committed_fixture_passes():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    exit_code = run(fixture, n_perm=500, seed=0, alpha=0.05, live=False)
    assert exit_code == 0


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    exit_code = run(missing, n_perm=50, seed=0, alpha=0.05, live=False)
    assert exit_code == 1
