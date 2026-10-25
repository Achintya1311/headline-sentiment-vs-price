# Fixtures

Educational research on historical data. Not investment advice. Not a SEBI-registered research analyst.

## Third-party headline text

These files hold news headlines exactly as their publishers wrote them:

- `headlines/headlines_raw.csv` - Economic Times markets RSS (plus GDELT rows), added to daily by `.github/workflows/collect-headlines.yml`
- `headlines/headlines_regression_50.csv` - a frozen 50-headline copy that the tests pin; it reproduces the Day 5 result
- `rss/economic_times_markets.xml` - the raw RSS snapshot the scraper reads offline
- `gdelt/*.json` - raw GDELT DOC API responses

They are collected for one purpose: to measure the tone of a headline and compare it with the stock's later return. The project does not write, choose or endorse them. Any rating, target, return figure or buy/sell wording inside a headline is the publisher's, not this project's, and is not a recommendation. The text is kept unedited because the research measures it as published; changing it would change the result.

## Other fixtures

- `prices/*.json` - daily price bars from Yahoo Finance's free chart endpoint, so the analysis runs offline
- `eval/vader_eval_set.csv` - hand-labelled tone set for the scorer evaluation
- `sentiment/INFY.json` - the weekly measured tone score written by `finbert-sentiment-contract.yml` (a headline count and an average score, not a call)
