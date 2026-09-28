# Headline sentiment vs price

Scores financial headlines and tests them against subsequent returns, with a leakage control that has to fail before any result is believed.

**Status:** Last checkpoint 2026-09-28 · Next: Day 3 - FinBERT scorer, plus an agreement analysis against VADER

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

# Day 2: VADER baseline scorer + evaluation harness. Fully offline - the
# VADER lexicon is vendored under sentiment/lexicons/, not downloaded.
python -m sentiment.score                                # scores fixtures/headlines/headlines_raw.csv
python -m sentiment.score --in path/to.csv --out path/to_scored.csv
python -m sentiment.evaluate                              # accuracy vs fixtures/eval/vader_eval_set.csv

# Day 3: FinBERT scorer + VADER/FinBERT agreement analysis. Needs torch,
# installed separately as a CPU-only wheel (see requirements.txt), and a
# network connection the first time only, to download the model:
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m sentiment.finbert                               # scores fixtures/headlines/headlines_raw.csv with FinBERT
python -m sentiment.agreement                              # VADER vs FinBERT: corpus agreement + eval-set accuracy

# Day 4+ (not built yet):
python -m sentiment.analyse --ticker EXAMPLE.NS --fixtures fixtures/headlines_sample.parquet
```

No API key is needed through Day 3 - RSS feeds are public, the VADER lexicon is vendored, and FinBERT (`ProsusAI/finbert`) is a public HuggingFace model. `.env.example` is for a later day's price data.

## Findings

**Day 1 - RSS scraper.** Three feeds registered, one actually reachable from this sandbox:

- `economic_times_markets` (economictimes.indiatimes.com): robots.txt allows the RSS path, live fetch succeeds. 50 headlines scraped and snapshotted to `fixtures/rss/economic_times_markets.xml` - this is real, current data, not synthetic.
- `moneycontrol_business` (moneycontrol.com): fetching moneycontrol's own `robots.txt` returns HTTP 403. Per RFC 9309, a 401/403 on robots.txt itself means "assume full disallow" - `RobotsCache` does exactly that and the feed is skipped, never fetched. No fixture exists for this feed because it has never been legitimately scraped.
- `reuters_business` (feeds.reuters.com): unreachable from this sandbox (proxy returns a 502 on the CONNECT tunnel). `RobotsCache` fails closed on any connection error while reading robots.txt, so this also comes back "denied" rather than attempting the feed anyway. No fixture exists for the same reason.

Storage is CSV, not the parquet the scaffold's "How to run" section assumed - neither `pyarrow` nor `fastparquet` is available in this sandbox and nothing else in the portfolio uses parquet either, so `fixtures/headlines/headlines_raw.csv` is the real format going forward. The `sentiment.analyse --fixtures ...parquet` line above is Day 3+ scaffold text, not yet true.

**Day 2 - VADER baseline scorer and evaluation harness.** `sentiment.vader_score` wraps NLTK's `SentimentIntensityAnalyzer` against a vendored copy of the lexicon (`sentiment/lexicons/vader_lexicon.txt`, ~92 KB, see `NOTICE.md` there) so scoring never needs `nltk.download()` at run time - verified by moving the sandbox's own downloaded copy aside and re-running the full suite and both CLIs, which behaved identically. `python -m sentiment.score` scores all 50 committed headlines (standard VADER thresholds: compound ≥ 0.05 positive, ≤ -0.05 negative, else neutral) - 29 positive / 16 neutral / 5 negative, mean compound +0.17. That skew is itself a finding, not a market signal: the evaluation harness below shows VADER over-reads routine corporate-action language ("approves", "announces") as positive.

`python -m sentiment.evaluate` runs the scorer against a 24-row hand-labeled set (`fixtures/eval/vader_eval_set.csv`) split into three categories and reports accuracy overall and per category:

| Category | Accuracy | What it tests |
|---|---|---|
| `general` (14) | 14/14 (100%) | headlines whose tone lives in everyday sentiment words - VADER's home turf |
| `neutral_factual` (4) | 3/4 (75%) | routine, tone-free corporate notices |
| `finance_jargon` (6) | 0/6 (0%) | tone that lives in finance-specific phrasing a general lexicon can't read |

Overall: 17/24 (70.8%). The `finance_jargon` column is the headline result, and it is exactly the caveat NEXT_STEPS.md already flagged: `"Company beats earnings estimates, raises full-year guidance"` scores `0.0` (neutral) despite being unambiguously bullish, and `"Manufacturer recalls product over safety defect"` scores *positive* because "safety" is a positive lexicon word with no notion that "safety defect" reverses it. This is the gap Day 3's FinBERT scorer and the VADER/FinBERT agreement analysis exist to close - not a bug in this scorer, VADER working exactly as documented.

**Day 3 - FinBERT scorer and VADER/FinBERT agreement analysis.** `sentiment/finbert_score.py` wraps `ProsusAI/finbert` (a BERT model fine-tuned on the Financial PhraseBank) via HuggingFace `transformers` + CPU-only `torch`. Unlike VADER's vendored lexicon, the ~440 MB model weights are not committed to this repo - they are pulled from the HuggingFace Hub on first use and cached under `~/.cache/huggingface`, so Day 3 needs network access once, then runs offline. `python -m sentiment.finbert` scores all 50 committed headlines: 15 positive / 25 neutral / 10 negative, mean derived score (positive − negative probability) +0.072 - far more neutral-heavy than VADER's 29/16/5, because FinBERT does not treat routine "Share Price Highlights" roundup headlines as sentiment-bearing the way VADER's lexicon does.

`python -m sentiment.agreement` runs both comparisons NEXT_STEPS.md asked for:

- **Corpus agreement** (50 scraped headlines, no ground truth): VADER and FinBERT agree on only **38%** of labels. The single biggest driver is mechanical, not a modeling disagreement: 13 headlines follow the pattern `"<Company> Share Price Highlights: <Company> Stock Price History"`, and VADER scores every one of them positive (compound +0.296, apparently from "Highlights") while FinBERT correctly reads them as neutral (no sentiment content at all). Strip that pattern out and the disagreement is still substantial - the two models are measuring different things, which is exactly why an agreement number alone would be misleading without looking at *where* they disagree.
- **Eval-set accuracy** (24 hand-labeled headlines, ground truth from a human): FinBERT scores 91.7% overall vs VADER's 70.8%, and the category breakdown shows why - `finance_jargon` accuracy goes from VADER's documented 0/6 (0%) to FinBERT's 6/6 (100%), closing the exact gap Day 2 found (`"Company beats earnings estimates, raises full-year guidance"` now reads positive; `"Manufacturer recalls product over safety defect"` now reads negative). That is not a clean sweep, though: FinBERT drops slightly below VADER on `general` (92.9% vs 100%), misreading `"Bank shares rally on rate cut hopes"` as negative - an honest miss on a headline VADER's everyday-sentiment lexicon gets right, worth naming rather than glossing over the one category where a finance-tuned model loses to a general-purpose one.

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
| 2026-09-28 | `aa60916` | Day 2: VADER baseline scorer (sentiment.vader_score, sentiment.score CLI) plus an evaluation harness (sentiment.evaluate) against a 24-row hand-labeled set. The VADER lexicon is vendored under sentiment/lexicons/ (verified offline by moving the sandbox's downloaded nltk_data aside and re-running everything) rather than fetched at run time. Honest headline result: VADER is 14/14 on headlines carrying everyday sentiment words but 0/6 on finance-specific phrasing ('beats earnings, raises guidance' scores neutral; a safety-defect recall scores positive because 'safety' alone is a positive lexicon word) - 17/24 (70.8%) overall, exactly the gap NEXT_STEPS.md flagged and Day 3's FinBERT scorer exists to close. Verified: 36/36 tests pass (16 new), both CLIs run by hand end to end against the real 50-headline fixture. Both this repo's and the hub's local main branches were found stuck in stale detached HEAD (the same recurring issue every prior day has hit) and fixed before committing. | Day 3 - FinBERT scorer, plus an agreement analysis against VADER |
| 2026-09-28 | `f3b6ec4` | Day 1: RSS scraper (sentiment.scrape) with a robots.txt cache (fail-closed on a 403 or unreachable robots.txt, per RFC 9309) and a per-host rate limiter that also respects a feed's own Crawl-delay. Ran it live against all 3 registered feeds: economic_times_markets is genuinely allowed and reachable (50 real headlines scraped, snapshotted to fixtures/rss/economic_times_markets.xml with source, title, link, and both the raw and UTC-normalized published timestamp); moneycontrol_business is honestly robots-denied (moneycontrol.com's own robots.txt 403s); reuters_business is unreachable from this sandbox's network (proxy 502 on the CONNECT tunnel), which the robots cache also treats as denied rather than guessing. Headlines are stored as CSV, not the parquet the scaffolded README assumed - neither pyarrow nor fastparquet is available here and no other repo in the portfolio uses parquet either, so README now says so. Offline default reads committed fixtures/rss/*.xml so tests and CI need no network; --live re-fetches for real. 20/20 tests pass (robots allow/deny/fail-closed semantics, rate-limiter timing, RSS parsing incl. malformed items, CSV dedup/merge-across-runs, and the CLI both offline and live-with-mocked-fetch); python -m sentiment.scrape and --live were both run by hand end to end. | Day 2 - VADER baseline scorer and an evaluation harness |
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- News feeds are edited and deleted, so coverage is partial and not reproducible from the live web. Everything used is snapshotted.
- Coverage is currently one source (Economic Times markets RSS). Moneycontrol blocks robots.txt itself with a 403 and Reuters' feed host is unreachable from this sandbox's network - both are recorded, not silently dropped. A single-source sample cannot support a claim about "headline sentiment" broadly; later days' correlation and event-study work needs to be read against that, or coverage needs to widen first.
- VADER is a general-purpose lexicon and scores plainly bullish financial phrasing as neutral. Treated as a baseline, not a finance model - Day 2's evaluation harness measures this directly (0/6 on the `finance_jargon` category) rather than asserting it.
- The eval set is 24 hand-labeled headlines I wrote, not an independent or blind-labeled benchmark, and it is small - a category accuracy of 0/6 or 6/6 is a signal, not a precise estimate. It exists to catch regressions and to give Day 3's FinBERT comparison a documented baseline to beat, not to be a rigorous scorer benchmark on its own. The same 24 headlines were also used to pick which model to trust more, so this is not a held-out test set for anything built on top of FinBERT later.
- FinBERT's model weights are not vendored like the VADER lexicon - they are fetched from the HuggingFace Hub on first use (~440 MB) and cached locally. If this sandbox (or CI) ever runs with no network and no pre-populated cache, `sentiment.finbert` and `sentiment.agreement` will fail; unlike Day 1/2's fixture-first offline guarantee, Day 3 needs network at least once.
- FinBERT is not strictly better than VADER: it closes the `finance_jargon` gap completely (0% -> 100%) but loses ground on `general` (100% -> 92.9%), misreading "Bank shares rally on rate cut hopes" as negative. Neither model is a free upgrade over the other across every category.
- The 38% corpus-level agreement rate is descriptive, not a quality metric - the unlabeled 50-headline corpus has no ground truth, so "the models disagree" says nothing about which one is right. A large share of the disagreement traces to a single repeated headline template ("Share Price Highlights"), so the number would move a lot with a different or larger sample.
- Neither the 50 scraped headlines nor the eval set have been checked against actual subsequent price moves - that correlation and event-study work is Day 5/6, deliberately kept separate from scorer accuracy.
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
