"""S20: every named-stock CLI prints PLAN.md rule 1's disclaimer as its first stdout line."""
import subprocess
import sys

import pytest

import sentiment

REG50 = "fixtures/headlines/headlines_regression_50.csv"


def cli_args(tmp_path):
    return {
        "sentiment.correlate": ["--in", REG50, "--out", str(tmp_path / "c.csv")],
        "sentiment.regress": ["--in", REG50, "--out", str(tmp_path / "r.csv")],
        "sentiment.robustness": ["--in", REG50, "--n-perm", "50"],
        "sentiment.audit": ["--in", REG50, "--n-perm", "50"],
        "sentiment.leakage_audit": ["--in", REG50, "--trials", "3"],
        "sentiment.overlay": ["--in", REG50, "--out-dir", str(tmp_path / "charts")],
        "sentiment.contract": ["--ticker", "INFY.NS", "--out", str(tmp_path / "k.json")],
        "sentiment.align": ["--out", str(tmp_path / "a.csv")],
        "sentiment.score": ["--out", str(tmp_path / "s.csv")],
    }


MODULES = [
    "sentiment.align",
    "sentiment.audit",
    "sentiment.contract",
    "sentiment.correlate",
    "sentiment.leakage_audit",
    "sentiment.overlay",
    "sentiment.regress",
    "sentiment.robustness",
    "sentiment.score",
]


def test_disclaimer_is_plan_rule_1_line():
    assert sentiment.DISCLAIMER == (
        "Educational research on historical data. Not investment advice. "
        "Not a SEBI-registered research analyst."
    )


@pytest.mark.parametrize("module", MODULES)
def test_cli_prints_disclaimer_first(module, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", module, *cli_args(tmp_path)[module]],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == sentiment.DISCLAIMER
