import random
from datetime import date, datetime, timezone

import sentiment.correlate as correlate
import sentiment.prices as prices
from sentiment.audit import (
    CorrelationAudit,
    audit_correlation,
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


def test_shuffle_published_at_permutes_timestamps_keeps_everything_else(monkeypatch):
    rng = random.Random(0)
    headlines = [
        make_headline("PC Jeweller shares surge", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("Fortis Healthcare shares plunge", "2", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)),
        make_headline("BSE shares rally", "3", datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc)),
    ]

    shuffled = shuffle_published_at(headlines, rng)

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    for h in shuffled:
        assert h.published_raw == h.published_at.isoformat()


def test_shuffle_published_at_actually_reorders_on_a_nontrivial_seed():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(10)
    ]
    rng = random.Random(1)
    shuffled = shuffle_published_at(headlines, rng)
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_audit_correlation_reports_none_when_too_few_resolved_headlines(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = [make_headline("some unrelated headline", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc))]

    result = audit_correlation(headlines, "contemporaneous_return", "contemporaneous", n_shuffles=10)

    assert result.observed is None
    assert result.permutation_p is None
    assert result.passes  # nothing to audit counts as passing, not failing


# Six single-company headlines whose VADER sentiment cleanly tracks the sign
# of that company's return on its OWN publish day - but each company also
# has a committed (flat, 0%) bar for every other headline's day, so a
# timestamp shuffle that reassigns a headline to a different real day still
# finds a price to pair it with, just not the one its content was about.
_LEAK_FREE_HEADLINES = [
    ("PC Jeweller shares surge as blockbuster fantastic excellent profit beats estimates",
     "PCJEWELLER.NS", date(2026, 9, 28), 0.10),
    ("Fortis Healthcare shares plunge amid disastrous terrible dismal results",
     "FORTIS.NS", date(2026, 9, 29), -0.10),
    ("BSE shares rally on outstanding superb robust performance",
     "BSE.NS", date(2026, 9, 30), 0.08),
    ("PB Fintech shares tumble on disappointing horrible weak guidance",
     "POLICYBZR.NS", date(2026, 10, 1), -0.08),
    ("Suzlon Energy shares jump on terrific excellent stellar order win",
     "SUZLON.NS", date(2026, 10, 2), 0.06),
    ("Great Eastern Shipping shares slump on awful bleak grim outlook",
     "GESHIP.NS", date(2026, 10, 5), -0.06),  # Monday - Oct 3/4 is a weekend
]

_ALL_DATES = [d for _, _, d, _ in _LEAK_FREE_HEADLINES]


def _pre_open_utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 2, 0, 0, tzinfo=timezone.utc)  # 07:30 IST, before the 09:15 open


def _setup_leak_free_fixtures(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = []
    for i, (title, ticker, own_date, pct) in enumerate(_LEAK_FREE_HEADLINES):
        base = 100.0
        bars = [
            Bar(date=d, open=base, close=base * (1 + pct) if d == own_date else base)
            for d in _ALL_DATES
        ]
        save_fixture(ticker, bars)
        headlines.append(make_headline(title, str(i), _pre_open_utc(own_date)))
    return headlines


def test_audit_correlation_passes_on_a_genuine_time_dependent_relationship(tmp_path, monkeypatch):
    headlines = _setup_leak_free_fixtures(tmp_path, monkeypatch)

    result = audit_correlation(
        headlines, "contemporaneous_return", "contemporaneous", n_shuffles=200, seed=0
    )

    assert isinstance(result, CorrelationAudit)
    assert result.observed is not None
    assert result.passes


def test_audit_correlation_catches_a_planted_return_leak(tmp_path, monkeypatch):
    """Plant the exact bug class this audit exists to catch: the return
    looked up for a ticker stops actually depending on the headline's
    aligned session_date (i.e. on published_at) and instead always comes
    back as that ticker's one big-move day, regardless of which date
    alignment computed. Once that is true, shuffling published_at is a
    no-op for the correlation's y-values - every shuffled trial reproduces
    the same "significant" result as the real run. The permutation p-value
    alone would not catch this (a no-op shuffle is never an outlier against
    itself) - the false-positive-rate metric is what flags it, which is
    exactly why the audit reports both."""
    headlines = _setup_leak_free_fixtures(tmp_path, monkeypatch)

    def leaky_bar_on(bars, on):
        # Ignore the (possibly shuffled-session) date entirely and always
        # hand back the one day this ticker actually moved on.
        return max(bars, key=lambda b: abs(b.close - b.open))

    monkeypatch.setattr(correlate, "bar_on", leaky_bar_on)

    result = audit_correlation(
        headlines, "contemporaneous_return", "contemporaneous", n_shuffles=200, seed=0
    )

    assert result.observed is not None
    assert result.false_positive_rate is not None
    assert result.false_positive_rate > 0.9
    assert not result.passes
