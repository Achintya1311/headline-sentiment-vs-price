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

- Public RSS feeds (Moneycontrol, Economic Times, Reuters) - scraped politely, robots.txt respected
- yfinance - price history
- VADER (nltk) and FinBERT (HuggingFace, local CPU) - both free

## How to run

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m sentiment.analyse --ticker EXAMPLE.NS --fixtures fixtures/headlines_sample.parquet
```

Runs offline against committed fixtures by default. Live data needs a key in `.env` (see `.env.example`); the fixture path is the default so nothing blocks on network access.

## Findings

Nothing yet. This section fills in as the work lands, including the results that do not flatter the method.

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- News feeds are edited and deleted, so coverage is partial and not reproducible from the live web. Everything used is snapshotted.
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
