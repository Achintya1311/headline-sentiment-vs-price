"""The data workflows test the repo before they commit (hub CRITIQUES.md P88).

Their GITHUB_TOKEN pushes trigger no ci.yml run, so a data change that broke
a test would otherwise sit on main unseen until the next human push.
"""

from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
DATA_WORKFLOWS = ["collect-headlines.yml", "fetch-prices.yml", "finbert-sentiment-contract.yml"]


@pytest.mark.parametrize("name", DATA_WORKFLOWS)
def test_pytest_runs_before_the_commit(name):
    text = (WORKFLOWS / name).read_text()
    assert "pytest -q" in text, f"{name} never runs the tests"
    assert text.index("pytest -q") < text.index("git commit"), f"{name} commits before testing"


@pytest.mark.parametrize("name", DATA_WORKFLOWS)
def test_install_is_the_readme_install(name):
    # The README install (P86): the package itself plus requirements.txt.
    assert "pip install -r requirements.txt -e ." in (WORKFLOWS / name).read_text()


def test_only_the_finbert_workflow_installs_torch():
    for name in DATA_WORKFLOWS:
        has_torch = "download.pytorch.org" in (WORKFLOWS / name).read_text()
        assert has_torch == (name == "finbert-sentiment-contract.yml"), name
