"""README's grown-corpus line names the corpus its numbers come from (hub R15).

The daily collector grows headlines_raw.csv, so an unpinned "n=42" goes stale
within a day. The line must name the commit it ran at and that commit's
headline count; where the commit is in local history, the count must match.
"""

import csv
import io
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORPUS = "fixtures/headlines/headlines_raw.csv"


def _grown_line() -> str:
    lines = [l for l in (ROOT / "README.md").read_text().splitlines() if l.startswith("**Grown corpus")]
    assert len(lines) == 1, "README needs exactly one **Grown corpus line"
    return lines[0]


def test_line_names_commit_and_headline_count():
    line = _grown_line()
    assert re.search(r"at `[0-9a-f]{7,40}`", line), "name the commit the numbers ran at"
    assert re.search(r"\b\d+ headlines\b", line)


def test_count_matches_the_named_commit():
    line = _grown_line()
    sha = re.search(r"at `([0-9a-f]{7,40})`", line)
    count = re.search(r"\b(\d+) headlines\b", line)
    if not (sha and count):
        pytest.fail("line names no commit or count")
    shown = subprocess.run(["git", "show", f"{sha.group(1)}:{CORPUS}"], cwd=ROOT, capture_output=True, text=True)
    if shown.returncode:
        pytest.skip(f"{sha.group(1)} not in this (shallow) checkout")
    assert sum(1 for _ in csv.DictReader(io.StringIO(shown.stdout))) == int(count.group(1))
