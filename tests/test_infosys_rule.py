"""Day 198 (hub CRITIQUES.md P134): every single-company "Infosys ..."
headline resolves to INFY.NS, not only the "Share Price Highlights"
template. Multi-company market wraps that merely list Infosys stay
unresolved."""

import csv
from pathlib import Path

from sentiment.tickers import resolve

RAW = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"


def test_infosys_results_and_move_headlines_resolve_to_infy():
    for title in [
        "Infosys Q2 results preview: Will the results provide oomph to surging IT stocks?",
        "Infosys shares jump 3% ahead of Q2 earnings announcement. What to expect?",
        "Infosys dividend alert! IT major announces Rs 26 per share dividend. Check record date",
        "Infosys upbeat on deals, cautious on growth as competition intensifies",
        "Infosys Share Price Live Updates: Current Trading Status of Infosys",
    ]:
        assert resolve(title) == ("Infosys", "INFY.NS"), title


def test_a_word_that_only_starts_with_infosys_is_not_matched():
    assert resolve("InfosysBPM spins off unit") is None


def test_market_wrap_listing_infosys_stays_unresolved():
    title = "Market wrap: Infosys, HCL Tech, Bajaj Finance, BSE top gainers and losers on Nifty and Sensex on Wednesday"
    assert resolve(title) is None


def test_every_title_starting_with_infosys_in_the_corpus_resolves():
    with RAW.open(newline="", encoding="utf-8") as f:
        titles = [row["title"] for row in csv.DictReader(f)]
    infosys = [t for t in titles if t.startswith("Infosys ")]
    assert len(infosys) >= 9
    assert all(resolve(t) == ("Infosys", "INFY.NS") for t in infosys)
    wraps = [t for t in titles if "Infosys" in t and t.startswith("Market wrap")]
    assert wraps and all(resolve(t) is None for t in wraps)
