# Headline sentiment vs price

Scores financial headlines and tests them against subsequent returns, with a leakage control that has to fail before any result is believed.

**Status:** Not started · Next: Day 1 - RSS scraper respecting robots.txt, timestamped headlines

## What this is

Scrapes headlines with timestamps, scores them with VADER as a baseline and FinBERT as the finance-tuned upgrade, then tests correlation and predictive power out of sample.

The hard part is not the model, it is the timestamp alignment. A headline stamped 09:20 is not tradable at the 09:15 open, and getting that wrong is how these projects quietly lie. It gets its own stage and its own control test.

## Correctness gate

Shuffled-timestamp control: randomise headline times and the signal must disappear. Runs in CI, not once by hand.

This is the test that decides whether the repo is finished. A result that has not passed it is a draft.

## Data sources

Every source is free. Nothing in this project requires a paid tier, a subscription, or a funded account.

- Public RSS feeds (Moneycontrol, Economic Times, Reuters) - scraped politely, robots.txt respected. As of Day 1, only Economic Times is actually reachable and permitted; see Findings.
- yfinance - price history
- VADER (nltk) and FinBERT (HuggingFace, local CPU) - both free

## How to run

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Day 1: scrape headlines. Offline by default - reads the committed
# fixtures/rss/*.xml snapshots, so nothing blocks on network access.
python -m sentiment.scrape
python -m sentiment.scrape --live                       # real fetch, robots.txt + rate limit enforced
python -m sentiment.scrape --feed economic_times_markets --live

# Day 2+ (not built yet):
python -m sentiment.analyse --ticker EXAMPLE.NS --fixtures fixtures/headlines_sample.parquet
```

No API key is needed for Day 1 - RSS feeds are public. `.env.example` is for a later day's price data.

## Findings

**Day 1 - RSS scraper.** Three feeds registered, one actually reachable from this sandbox:

- `economic_times_markets` (economictimes.indiatimes.com): robots.txt allows the RSS path, live fetch succeeds. 50 headlines scraped and snapshotted to `fixtures/rss/economic_times_markets.xml` - this is real, current data, not synthetic.
- `moneycontrol_business` (moneycontrol.com): fetching moneycontrol's own `robots.txt` returns HTTP 403. Per RFC 9309, a 401/403 on robots.txt itself means "assume full disallow" - `RobotsCache` does exactly that and the feed is skipped, never fetched. No fixture exists for this feed because it has never been legitimately scraped.
- `reuters_business` (feeds.reuters.com): unreachable from this sandbox (proxy returns a 502 on the CONNECT tunnel). `RobotsCache` fails closed on any connection error while reading robots.txt, so this also comes back "denied" rather than attempting the feed anyway. No fixture exists for the same reason.

Storage is CSV, not the parquet the scaffold's "How to run" section assumed - neither `pyarrow` nor `fastparquet` is available in this sandbox and nothing else in the portfolio uses parquet either, so `fixtures/headlines/headlines_raw.csv` is the real format going forward. The `sentiment.analyse --fixtures ...parquet` line above is Day 2+ scaffold text, not yet true.

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- News feeds are edited and deleted, so coverage is partial and not reproducible from the live web. Everything used is snapshotted.
- Coverage is currently one source (Economic Times markets RSS). Moneycontrol blocks robots.txt itself with a 403 and Reuters' feed host is unreachable from this sandbox's network - both are recorded, not silently dropped. A single-source sample cannot support a claim about "headline sentiment" broadly; later days' correlation and event-study work needs to be read against that, or coverage needs to widen first.
- VADER is a general-purpose lexicon and scores plainly bullish financial phrasing as neutral. Treated as a baseline, not a finance model.
- Correlation over a short window with many tested horizons manufactures significance. The hypothesis is fixed before the data is touched.

## Where this sits

Part of a nine-repo research pipeline. Stock Stalker screens the NSE universe; this repo publishes a versioned artifact it reads back:

```json
{
  "sentiment": {
    "score_7d": 0.31,
    "headline_count_7d": 12,
    "model": "finbert"
  }
}
```

Communication is by file contract, not imports, so either side can be refactored without breaking the other.

## Exam mapping

Series XV ch.12.10 (behavioural biases), ch.4.6 (behavioural approach to equity investing)

---

CLI only, by design. No dashboard, no server. Charts and documents are written to `outputs/`.
