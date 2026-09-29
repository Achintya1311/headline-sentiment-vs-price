"""Map a scraped headline's title to a single NSE ticker, if it names one (Day 5).

This is a closed lookup, not NLP: the RSS fixture is a mixed feed where most
headlines are either about one company (a "Share Price Highlights" liveblog
title, or a "<Company> shares rise/fall N%" move headline) or about several
at once (a screener list - "X among N stocks showing..." - or a market-wide
wrap). Only the single-company kind can be honestly paired with one ticker's
return; a screener-list headline names a company only as one example among
several, and resolving it to that one ticker would overstate what the
headline is actually about. So the table below is deliberately narrow: it
matches the specific single-company patterns present in
``fixtures/headlines/headlines_raw.csv`` and nothing else, and new headline
shapes need a new rule added on purpose rather than a fuzzy match picking
one up by accident.
"""

from __future__ import annotations

import re

# (compiled pattern, company name, Yahoo Finance ticker or None if unresolved)
# None means: this headline is about exactly one company, but that company's
# Yahoo ticker could not be found (see README Findings) - excluded from the
# correlation/event-study CLI rather than guessed.
_RULES: list[tuple[re.Pattern[str], str, str | None]] = [
    (re.compile(r"^SBI Life Share Price Highlights"), "SBI Life", "SBILIFE.NS"),
    (re.compile(r"^Nestle India Share Price Highlights"), "Nestle India", "NESTLEIND.NS"),
    (re.compile(r"^Sun Pharma Share Price Highlights"), "Sun Pharma", "SUNPHARMA.NS"),
    (re.compile(r"^Grasim Inds Share Price Highlights"), "Grasim Inds", "GRASIM.NS"),
    (re.compile(r"^Tech Mahindra Share Price Highlights"), "Tech Mahindra", "TECHM.NS"),
    (re.compile(r"^Wipro Share Price Highlights"), "Wipro", "WIPRO.NS"),
    (re.compile(r"^Bharti Airtel Share Price Highlights"), "Bharti Airtel", "BHARTIARTL.NS"),
    (re.compile(r"^Tata Steel Share Price Highlights"), "Tata Steel", "TATASTEEL.NS"),
    (re.compile(r"^LTIMindtree Share Price Highlights"), "LTIMindtree", None),
    (re.compile(r"^HUL Share Price Highlights"), "HUL", "HINDUNILVR.NS"),
    (re.compile(r"^Infosys Share Price Highlights"), "Infosys", "INFY.NS"),
    (re.compile(r"^HCL Tech Share Price Highlights"), "HCL Tech", "HCLTECH.NS"),
    (re.compile(r"^HDFC Life Share Price Highlights"), "HDFC Life", "HDFCLIFE.NS"),
    (re.compile(r"^Eicher Motors Share Price Highlights"), "Eicher Motors", "EICHERMOT.NS"),
    (re.compile(r"^Bajaj Finserv Share Price Highlights"), "Bajaj Finserv", "BAJAJFINSV.NS"),
    (re.compile(r"^PC Jeweller shares"), "PC Jeweller", "PCJEWELLER.NS"),
    (re.compile(r"^Fortis Healthcare shares"), "Fortis Healthcare", "FORTIS.NS"),
    (re.compile(r"^BSE shares"), "BSE", "BSE.NS"),
    (re.compile(r"^PB Fintech shares"), "PB Fintech", "POLICYBZR.NS"),
    (re.compile(r"^NSE shares"), "NSE", "NSE.BO"),
    (re.compile(r"^Suzlon Energy shares"), "Suzlon Energy", "SUZLON.NS"),
    (re.compile(r"^HDFC Bank "), "HDFC Bank", "HDFCBANK.NS"),
    (re.compile(r"^Great Eastern Shipping "), "Great Eastern Shipping", "GESHIP.NS"),
    (re.compile(r"Jefferies names Max Financial"), "Max Financial", "MFSL.NS"),
]


def resolve(title: str) -> tuple[str, str | None] | None:
    """Return ``(company, ticker)`` if ``title`` names exactly one company this
    table knows, else ``None``. ``ticker`` is ``None`` if the company is known
    but its Yahoo ticker could not be resolved (see README Findings)."""
    for pattern, company, ticker in _RULES:
        if pattern.search(title):
            return company, ticker
    return None
