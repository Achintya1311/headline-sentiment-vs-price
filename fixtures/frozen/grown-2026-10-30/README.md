# Frozen inputs for the README's grown-corpus line

`headlines_raw.csv` is `fixtures/headlines/headlines_raw.csv` at commit `67baa03` (448 headlines). `prices/` is `fixtures/prices/` at commit `f17fc93` (the 27 Oct 2026 refresh). Both are copied unchanged so that

    python -m sentiment.correlate --in fixtures/frozen/grown-2026-10-30/headlines_raw.csv --prices-dir fixtures/frozen/grown-2026-10-30/prices

reproduces README:10 without git history (hub `CRITIQUES.md` P161). The live corpus and price fixtures keep growing; these files never change.
