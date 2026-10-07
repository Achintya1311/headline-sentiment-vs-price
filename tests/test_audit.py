import datetime as dt
import random
from datetime import datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import (
    SIGNAL_THRESHOLD,
    AuditResult,
    contemporaneous_r,
    run,
    run_audit,
    shuffle_published_at,
)
from sentiment.headline import Headline
from sentiment.prices import Bar, save_fixture
from sentiment.stats import PearsonResult
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


# --- shuffle_published_at -------------------------------------------------


def test_shuffle_published_at_is_a_permutation_of_the_same_timestamps():
    headlines = [
        make_headline(f"Infosys Share Price Highlights {i}", str(i), datetime(2026, 9, 21 + i, 2, 0, tzinfo=timezone.utc))
        for i in range(8)
    ]
    rng = random.Random(0)

    shuffled = shuffle_published_at(headlines, rng)

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # With 8 distinct timestamps a fixed seed should actually move at least
    # one of them - a shuffle that is secretly a no-op would defeat the
    # whole audit silently.
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_shuffle_published_at_preserves_headline_count_for_single_timestamp():
    headlines = [make_headline("Infosys Share Price Highlights", "1", datetime(2026, 9, 28, 2, 0, tzinfo=timezone.utc))]
    rng = random.Random(0)

    shuffled = shuffle_published_at(headlines, rng)

    assert shuffled == headlines


# --- AuditResult decision logic -------------------------------------------


def test_signal_survived_shuffle_is_false_when_there_is_no_real_signal():
    real = PearsonResult(r=-0.185, n=23, ci_low=-0.555, ci_high=0.246)
    result = AuditResult(real=real, shuffled_rs=[0.0] * 50)

    assert abs(real.r) < SIGNAL_THRESHOLD
    assert result.had_signal_to_lose is False
    assert result.signal_survived_shuffle is False


def test_signal_survived_shuffle_is_false_when_shuffle_destroys_a_real_signal():
    real = PearsonResult(r=0.9, n=10, ci_low=0.6, ci_high=0.98)
    shuffled_rs = [0.05, -0.1, 0.02, 0.0, -0.03, 0.08]

    result = AuditResult(real=real, shuffled_rs=shuffled_rs)

    assert result.had_signal_to_lose is True
    assert result.mean_abs_shuffled < abs(real.r) / 2
    assert result.signal_survived_shuffle is False


def test_signal_survived_shuffle_is_true_when_shuffle_reproduces_a_real_signal():
    # The leak signature: a real signal exists and scrambling timestamps
    # barely changes it - whatever drives the correlation does not depend
    # on correct timing.
    real = PearsonResult(r=0.9, n=10, ci_low=0.6, ci_high=0.98)
    shuffled_rs = [0.85, 0.88, 0.92, 0.80, 0.86, 0.90]

    result = AuditResult(real=real, shuffled_rs=shuffled_rs)

    assert result.had_signal_to_lose is True
    assert result.signal_survived_shuffle is True


# --- the real, leak-free pipeline on a synthetic injected signal ----------


def _build_injected_signal_fixture(tmp_path: Path, n: int = 10):
    """Headlines for a single company (INFY.NS), each pre-open on its own
    trading day, whose title's sentiment is deliberately chosen so that the
    day it is aligned to has a return proportional to that headline's own
    real VADER compound score. A leak-free pipeline that honours the real
    timestamp should recover a strong contemporaneous correlation; shuffling
    the timestamps should sever the headline-to-day pairing and collapse it.
    """
    # All start with "Infosys Share Price Highlights" so sentiment.tickers
    # resolves every one of them to INFY.NS (see sentiment/tickers.py) - the
    # trailing text is free to vary and still drives VADER's compound score.
    texts = [
        "Infosys Share Price Highlights: wins landmark deal, upgrades outlook, beats on margins",
        "Infosys Share Price Highlights: delivers stellar record profit, raises guidance sharply",
        "Infosys Share Price Highlights: announces strong buyback, stock soars on robust growth",
        "Infosys Share Price Highlights: wins big contract, surges on excellent upgrade news",
        "Infosys Share Price Highlights: posts modest results in a muted quarter",
        "Infosys Share Price Highlights: shares steady amid flat trading session",
        "Infosys Share Price Highlights: slips on weak demand, disappointing outlook cut",
        "Infosys Share Price Highlights: plunges after shocking profit warning, downgrade fears",
        "Infosys Share Price Highlights: collapses on fraud allegations, investors flee in panic",
        "Infosys Share Price Highlights: crashes on disastrous guidance slash, outlook grim",
    ][:n]

    headlines = []
    bars = []
    base_day = dt.date(2026, 9, 1)
    trading_day = base_day
    for i, text in enumerate(texts):
        while trading_day.weekday() >= 5:
            trading_day += dt.timedelta(days=1)
        compound = score_text(text)["compound"]
        # Pre-open (02:00 UTC ~ 07:30 IST) so align_headline keeps the
        # headline on this exact trading day.
        published_at = datetime(trading_day.year, trading_day.month, trading_day.day, 2, 0, tzinfo=timezone.utc)
        headlines.append(make_headline(text, str(i), published_at))
        # Open-to-close return engineered as a direct, strong function of
        # this headline's own real compound score.
        open_price = 100.0
        close_price = open_price * (1 + 0.20 * compound)
        bars.append(Bar(date=trading_day, open=open_price, close=close_price))
        trading_day += dt.timedelta(days=1)

    save_fixture("INFY.NS", bars)
    return headlines


def test_leak_free_pipeline_recovers_an_injected_signal(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = _build_injected_signal_fixture(tmp_path)

    real = contemporaneous_r(headlines)

    assert real is not None
    assert real.n == len(headlines)
    assert real.r > 0.9  # engineered as (almost) a perfect linear relationship


def test_shuffling_timestamps_collapses_the_injected_signal(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = _build_injected_signal_fixture(tmp_path)

    result = run_audit(headlines, n_shuffles=200, seed=0)

    assert result is not None
    assert result.had_signal_to_lose is True
    # The literal "Done when" requirement: shuffle the timestamps and the
    # signal must disappear.
    assert result.mean_abs_shuffled < 0.3
    assert result.signal_survived_shuffle is False


def test_cli_run_passes_on_the_synthetic_injected_signal_fixture(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = _build_injected_signal_fixture(tmp_path)
    in_path = tmp_path / "raw.csv"
    from sentiment.headline import write_csv

    write_csv(headlines, in_path)

    exit_code = run(in_path, n_shuffles=200, seed=0, live=False)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "PASS" in out


def test_run_against_the_real_committed_fixture_finds_no_signal_to_test(capsys):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, n_shuffles=100, seed=0, live=False)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "no real signal to test" in out


def test_run_missing_input_reports_failure(tmp_path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, n_shuffles=10, seed=0, live=False)

    assert exit_code == 1
