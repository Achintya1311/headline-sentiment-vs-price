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
    from pathlib import Path

    from sentiment.headline import read_csv

    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    headlines = read_csv(fixture)
    resolved = [h.title for h in headlines if resolve(h.title) is not None]
    assert len(resolved) == 24
    with_ticker = [h.title for h in headlines if (m := resolve(h.title)) is not None and m[1] is not None]
    assert len(with_ticker) == 23
