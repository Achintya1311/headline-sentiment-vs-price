from sentiment.tickers import resolve


def test_share_price_highlights_template_resolves_to_its_ticker():
    title = "Infosys Share Price Highlights: Infosys Stock Price History"
    assert resolve(title) == ("Infosys", "INFY.NS")


def test_shares_move_headline_resolves_to_its_ticker():
    title = "Suzlon Energy shares fall 2% to near six-month low, down 15% in one month"
    assert resolve(title) == ("Suzlon Energy", "SUZLON.NS")


def test_known_company_with_no_resolvable_ticker_returns_none_ticker():
    title = "LTIMindtree Share Price Highlights: LTIMindtree Stock Price History"
    assert resolve(title) == ("LTIMindtree", None)


def test_screener_list_headline_is_not_resolved_to_the_named_example():
    # names Cyient only as one of several stocks in a screen, not the subject
    title = "Cyient among 4 stocks showing White Marubozu Pattern"
    assert resolve(title) is None


def test_multi_company_market_wrap_is_not_resolved():
    title = "Market wrap: Infosys, Dr Reddy's Labs, Tata Motors PV, Power Grid top gainers and losers"
    assert resolve(title) is None


def test_index_level_headline_is_not_resolved():
    title = "Sensex crashes over 1,000 points, Nifty50 below 22,850; erases nearly ₹6 lakh cr in an hour"
    assert resolve(title) is None


def test_resolves_all_and_only_the_expected_headlines_in_the_committed_fixture():
    # pinned to the frozen 50-row regression snapshot, not the live
    # headlines_raw.csv the daily scrape workflow keeps appending to -
    # otherwise this count drifts every time that workflow commits (D10).
    from pathlib import Path

    from sentiment.headline import read_csv

    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_regression_50.csv"
    headlines = read_csv(fixture)
    resolved = [h.title for h in headlines if resolve(h.title) is not None]
    assert len(resolved) == 24
    with_ticker = [h.title for h in headlines if (m := resolve(h.title)) is not None and m[1] is not None]
    assert len(with_ticker) == 23


def test_known_tickers_is_sorted_deduplicated_and_excludes_unresolved_companies():
    from sentiment.tickers import known_tickers

    tickers = known_tickers()
    assert tickers == sorted(set(tickers))
    assert None not in tickers
    assert "INFY.NS" in tickers
    # LTIMindtree has no resolvable Yahoo ticker (see _RULES) - must not appear.
    assert "LTIM.NS" not in tickers
    assert "LTIMINDTREE.NS" not in tickers


def test_known_companies_is_sorted_deduplicated_and_excludes_unresolved_companies():
    from sentiment.tickers import known_companies

    companies = known_companies()
    assert companies == sorted(set(companies))
    assert "Infosys" in companies
    # LTIMindtree has no resolvable Yahoo ticker - sentiment.scrape_gdelt
    # must never spend a query on it.
    assert "LTIMindtree" not in companies
