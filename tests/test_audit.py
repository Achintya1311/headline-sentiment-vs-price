import datetime as dt
import random
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.audit import contemporaneous_r, run, run_shuffle_audit, shuffle_timestamps
from sentiment.headline import Headline, read_csv
from sentiment.prices import Bar, save_fixture
from sentiment.vader_score import score_headline

RETURN_PER_COMPOUND = 0.05

# Six distinct trading weekdays (Mon-Fri then the following Monday) - one per
# planted headline, and the full pool a timestamp shuffle can reassign any of
# them to.
SESSION_DATES = [
    dt.date(2026, 9, 7),
    dt.date(2026, 9, 8),
    dt.date(2026, 9, 9),
    dt.date(2026, 9, 10),
    dt.date(2026, 9, 11),
    dt.date(2026, 9, 14),
]

# Each title matches a distinct real pattern in sentiment/tickers.py, so each
# headline resolves to its own ticker - six separate price series, not one.
TITLES_AND_TICKERS = [
    ("PC Jeweller shares soar after blockbuster profit beat and sharply raised guidance", "PCJEWELLER.NS"),
    ("Fortis Healthcare shares plunge as fraud investigation widens and outlook slashed", "FORTIS.NS"),
    ("BSE shares jump on record trading volumes and strong profit growth outlook", "BSE.NS"),
    ("PB Fintech shares tumble after regulatory crackdown and heavy quarterly losses", "POLICYBZR.NS"),
    ("Suzlon Energy shares rally as profit triples and fresh orders surge sharply", "SUZLON.NS"),
    ("HDFC Bank shares slide after rating downgrade and mounting bad loan concerns", "HDFCBANK.NS"),
]


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=published_at,
    )


def build_planted_signal_headlines() -> tuple[list[Headline], list[float]]:
    """Six headlines, each pre-open on its own distinct trading day for its
    own ticker, with a synthetic price fixture where *only* the correct
    (unshuffled) session's return is an exact linear function of that
    headline's real VADER compound - every other candidate session (the
    ones a timestamp shuffle could reassign a headline to) is flat at 0%.
    A real, noise-free compound/return relationship exists only through the
    correct alignment, so the real run's r should be ~1 and every shuffled
    run's should collapse toward 0."""
    headlines = []
    compounds = []
    for i, (title, _ticker) in enumerate(TITLES_AND_TICKERS):
        published_at = datetime(
            SESSION_DATES[i].year, SESSION_DATES[i].month, SESSION_DATES[i].day, 2, 0, 0, tzinfo=timezone.utc
        )
        h = make_headline(title, f"link-{i}", published_at)
        headlines.append(h)
        compounds.append(score_headline(h).compound)

    for i, (_title, ticker) in enumerate(TITLES_AND_TICKERS):
        bars = []
        for d in SESSION_DATES:
            ret = RETURN_PER_COMPOUND * compounds[i] if d == SESSION_DATES[i] else 0.0
            bars.append(Bar(date=d, open=100.0, close=100.0 * (1 + ret)))
        save_fixture(ticker, bars)

    return headlines, compounds


def test_contemporaneous_r_needs_at_least_two_rows():
    assert contemporaneous_r([]) is None
    assert contemporaneous_r([{"compound": 0.1, "contemporaneous_return": 0.01}]) is None


def test_contemporaneous_r_matches_a_known_linear_relationship():
    rows = [{"compound": x, "contemporaneous_return": 0.02 * x} for x in [-0.5, -0.2, 0.1, 0.4, 0.6]]
    assert contemporaneous_r(rows) == pytest.approx(1.0)


def test_shuffle_timestamps_permutes_without_changing_which_headline_is_which():
    headlines = [
        make_headline("A", "1", datetime(2026, 9, 7, 2, 0, tzinfo=timezone.utc)),
        make_headline("B", "2", datetime(2026, 9, 8, 2, 0, tzinfo=timezone.utc)),
        make_headline("C", "3", datetime(2026, 9, 9, 2, 0, tzinfo=timezone.utc)),
    ]

    shuffled = shuffle_timestamps(headlines, random.Random(1))

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)


def test_shuffle_audit_detects_a_real_planted_signal_and_shuffling_destroys_it(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines, compounds = build_planted_signal_headlines()
    # The six titles must not all score identically, or there is no real
    # relationship for the audit to find in the first place.
    assert len({round(c, 4) for c in compounds}) > 1

    result = run_shuffle_audit(headlines, n_shuffles=200, seed=0)

    assert result.real_n == 6
    # contemporaneous_return is an exact linear function of compound on the
    # correct (unshuffled) alignment - a perfect fit.
    assert result.real_r == pytest.approx(1.0, abs=1e-6)
    mean_abs_shuffled = sum(abs(r) for r in result.shuffled_rs) / len(result.shuffled_rs)
    assert mean_abs_shuffled < 0.5  # a real signal does not survive a random time-pairing
    assert result.p_value < 0.05


def test_shuffle_audit_on_the_real_fixture_is_null_and_unremarkable():
    # Mirrors Day 5's own finding (README): no real signal here for a leak
    # to fake, so the audit's honest answer is "nothing stands out."
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    headlines = read_csv(fixture)

    result = run_shuffle_audit(headlines, n_shuffles=50, seed=0)

    assert result.real_n == 23
    assert result.real_r == pytest.approx(-0.185, abs=0.01)
    assert result.p_value > 0.05


def test_run_shuffle_audit_raises_when_fewer_than_two_resolved_headlines(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0)])
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    ]

    with pytest.raises(ValueError):
        run_shuffle_audit(headlines, n_shuffles=10, seed=0)


def test_cli_run_against_committed_fixture_succeeds(capsys):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, n_shuffles=20, live=False, seed=0)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "real contemporaneous r=" in out
    assert "permutation p-value" in out


def test_cli_run_missing_input_reports_failure(tmp_path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, n_shuffles=10, live=False, seed=0)

    assert exit_code == 1
