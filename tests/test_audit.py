import datetime as dt
import random
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sentiment.audit as audit
import sentiment.prices as prices
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


# -- shuffle_timestamps: pure permutation mechanics -------------------------


def test_shuffle_timestamps_is_a_permutation_not_a_no_op():
    timestamps = [datetime(2026, 9, 28, h, 0, 0, tzinfo=timezone.utc) for h in range(2, 10)]
    rng = random.Random(0)

    shuffled = audit.shuffle_timestamps(timestamps, rng)

    assert sorted(shuffled) == sorted(timestamps)  # same multiset of values
    assert shuffled != timestamps  # overwhelmingly likely to reorder 8 distinct values


# -- contemporaneous_pairs: proves the shuffle argument actually flows -----
# through to the computed return, not just the ticker/compound side. A bug
# where alignment silently ignored its timestamp argument would make this
# fail directly - the one failure mode a permutation p-value cannot see on
# its own (see audit.py's module docstring).


def test_contemporaneous_pairs_uses_the_timestamp_argument_not_the_headlines_own(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [
            Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0),  # +10%
            Bar(date=dt.date(2026, 9, 29), open=100.0, close=90.0),  # -10%
        ],
    )
    original_ts = datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)  # pre-open -> session 28th
    resolved = [audit.ResolvedHeadline(title="x", ticker="INFY.NS", compound=0.5, published_at=original_ts)]

    _compounds, real_returns = audit.contemporaneous_pairs(resolved, [original_ts])
    assert real_returns == [pytest.approx(0.10)]

    other_ts = datetime(2026, 9, 28, 11, 0, 0, tzinfo=timezone.utc)  # post-close -> session 29th
    _compounds, shuffled_returns = audit.contemporaneous_pairs(resolved, [other_ts])
    assert shuffled_returns == [pytest.approx(-0.10)]


# -- PermutationAuditResult: pure stats, no pipeline needed -----------------


def test_p_value_and_leak_free_when_observed_is_unremarkable():
    # every shuffle beats the observed |r| -> p = 1.0, clearly not an outlier
    result = audit.PermutationAuditResult(observed_r=0.1, observed_n=10, null_abs_r=[0.9] * 10, n_shuffles=10)
    assert result.p_value == pytest.approx(1.0)
    assert result.leak_free(alpha=0.05) is True


def test_p_value_and_leak_free_when_observed_is_an_extreme_outlier():
    # no shuffle comes close to the observed |r| -> p = 1/(n+1), a significant outlier
    result = audit.PermutationAuditResult(observed_r=0.9, observed_n=10, null_abs_r=[0.01] * 99, n_shuffles=99)
    assert result.p_value == pytest.approx(1 / 100)
    assert result.leak_free(alpha=0.05) is False


# -- the real "Done when" gate: runs in CI, not once by hand ----------------


def test_permutation_audit_on_the_real_fixture_finds_no_significant_signal():
    headlines = read_csv(audit.DEFAULT_IN)

    result = audit.run_permutation_audit(headlines, n_shuffles=500, seed=0)

    assert result.observed_n == 23  # same scope Day 5 resolved
    # matches the Day 5 Findings section's documented contemporaneous r
    assert result.observed_r == pytest.approx(-0.185, abs=0.01)
    assert result.leak_free(alpha=0.05)


def test_run_cli_against_real_fixture_passes_and_returns_zero(capsys):
    exit_code = audit.run(audit.DEFAULT_IN, live=False, n_shuffles=300, seed=0, alpha=0.05)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "PASS" in out


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = audit.run(missing, live=False, n_shuffles=10, seed=0, alpha=0.05)

    assert exit_code == 1


# -- proof the test has teeth: a genuine, timing-dependent signal is -------
# flagged as significant, not rubber-stamped as "no signal" the way the
# real near-null fixture is. Mirrors Day 6's synthetic-signal check, which
# proved that regression pipeline wasn't just always returning zero.
#
# 12 "Share Price Highlights" headlines (a real tickers.py pattern), 6 with
# a compound score wired to +1.0 and 6 to -1.0 via a monkeypatched scorer -
# isolating the permutation logic under test from VADER's actual text
# scoring, which Day 2/3 already cover. The +1.0 group is timestamped
# pre-open Monday (aligns to Monday's session) and the -1.0 group post-close
# Monday (rolls to Tuesday); Monday's fixture bar is +10%, Tuesday's -10%,
# so the real, correctly-aligned pairing is a perfect r = +1.0. Shuffling
# published_at reassigns which group's return each headline gets, almost
# always destroying that perfect split (only 2 of the 12!/(6!*6!) * ... exact
# relabelings reproduce it), so the real r=1.0 should be a rare, significant
# outlier against the shuffled null.

_GROUP_PLUS = [
    ("SBI Life Share Price Highlights: SBI Life Stock Price History", "SBILIFE.NS"),
    ("Nestle India Share Price Highlights: Nestle India Stock Price History", "NESTLEIND.NS"),
    ("Sun Pharma Share Price Highlights: Sun Pharma Stock Price History", "SUNPHARMA.NS"),
    ("Grasim Inds Share Price Highlights: Grasim Inds. Stock Price History", "GRASIM.NS"),
    ("Tech Mahindra Share Price Highlights: Tech Mahindra Stock Price History", "TECHM.NS"),
    ("Wipro Share Price Highlights: Wipro Stock Price History", "WIPRO.NS"),
]
_GROUP_MINUS = [
    ("Bharti Airtel Share Price Highlights: Bharti Airtel Stock Price History", "BHARTIARTL.NS"),
    ("Tata Steel Share Price Highlights: Tata Steel Stock Price History", "TATASTEEL.NS"),
    ("HUL Share Price Highlights: HUL Stock Price History", "HINDUNILVR.NS"),
    ("Infosys Share Price Highlights: Infosys Stock Price History", "INFY.NS"),
    ("HCL Tech Share Price Highlights: HCL Tech Stock Price History", "HCLTECH.NS"),
    ("HDFC Life Share Price Highlights: HDFC Life Stock Price History", "HDFCLIFE.NS"),
]


class _FakeScored:
    def __init__(self, compound: float) -> None:
        self.compound = compound


def test_permutation_audit_flags_a_genuine_timing_dependent_signal_as_significant(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    for _title, ticker in _GROUP_PLUS + _GROUP_MINUS:
        save_fixture(
            ticker,
            [
                Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0),  # Monday: +10%
                Bar(date=dt.date(2026, 9, 29), open=100.0, close=90.0),  # Tuesday: -10%
            ],
        )

    compound_by_title = {title: +1.0 for title, _t in _GROUP_PLUS}
    compound_by_title.update({title: -1.0 for title, _t in _GROUP_MINUS})
    monkeypatch.setattr(audit, "score_headline", lambda h: _FakeScored(compound_by_title[h.title]))

    headlines = [
        # pre-open Monday -> aligns to Monday (28th), the +10% session
        make_headline(title, f"plus-{i}", datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc))
        for i, (title, _t) in enumerate(_GROUP_PLUS)
    ] + [
        # post-close Monday -> rolls to Tuesday (29th), the -10% session
        make_headline(title, f"minus-{i}", datetime(2026, 9, 28, 11, i, 0, tzinfo=timezone.utc))
        for i, (title, _t) in enumerate(_GROUP_MINUS)
    ]

    result = audit.run_permutation_audit(headlines, n_shuffles=500, seed=0)

    assert result.observed_n == 12
    assert result.observed_r == pytest.approx(1.0)
    assert result.p_value < 0.05
    assert result.leak_free(alpha=0.05) is False  # correctly flagged, not rubber-stamped
