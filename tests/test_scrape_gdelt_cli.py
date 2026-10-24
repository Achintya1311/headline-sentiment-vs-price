from pathlib import Path

from sentiment.headline import read_csv
from sentiment.scrape_gdelt import run


def test_offline_run_against_the_committed_infosys_fixture(tmp_path: Path):
    out = tmp_path / "headlines.csv"
    exit_code = run(["Infosys"], live=False, out=out)

    assert exit_code == 0
    headlines = read_csv(out)
    assert len(headlines) > 0
    assert all(h.source != "gdelt" for h in headlines)  # domain-named sources


def test_offline_run_skips_a_company_with_no_committed_fixture(tmp_path: Path, capsys):
    out = tmp_path / "headlines.csv"
    exit_code = run(["Some Company With No Fixture"], live=False, out=out)

    assert exit_code == 0  # a missing fixture is a skip, not a run failure
    err = capsys.readouterr().err
    assert "SKIPPED" in err
    assert not out.exists() or read_csv(out) == []


def test_offline_run_merges_with_an_existing_headlines_csv(tmp_path: Path):
    out = tmp_path / "headlines.csv"
    run(["Infosys"], live=False, out=out)
    first_count = len(read_csv(out))

    exit_code = run(["Infosys"], live=False, out=out)
    assert exit_code == 0
    assert len(read_csv(out)) == first_count  # same fixture again: dedup, not duplicated
