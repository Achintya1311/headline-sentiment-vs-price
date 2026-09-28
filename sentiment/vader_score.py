"""VADER baseline sentiment scorer (Day 2).

Scores headline text with NLTK's VADER lexicon, vendored under
``sentiment/lexicons/`` so scoring works offline against the committed copy
rather than requiring a network fetch through ``nltk.download()`` at run
time - see ``sentiment/lexicons/NOTICE.md``.

VADER is a general-purpose lexicon, not a finance model: it scores plainly
bullish or bearish financial phrasing as neutral when the sentiment lives in
domain jargon rather than everyday sentiment words (``"beats earnings,
raises guidance"`` -> ``0.0`` compound). Day 3's FinBERT scorer and the
agreement analysis against this baseline exist because of exactly that gap;
``sentiment/evaluate.py`` measures it directly instead of asserting it.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sentiment.headline import Headline

LEXICON_DIR = Path(__file__).resolve().parent / "lexicons"
LEXICON_FILE = LEXICON_DIR / "vader_lexicon.txt"

# Standard VADER thresholds (per the project's own README/paper): a compound
# score at or above 0.05 counts positive, at or below -0.05 negative,
# anything in between neutral.
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


@lru_cache(maxsize=1)
def get_analyzer() -> SentimentIntensityAnalyzer:
    """Build a VADER analyzer from the vendored lexicon, never a network fetch.

    Cached because building it re-parses a ~7500-line lexicon file; every
    caller in a process shares the one instance.
    """
    if not LEXICON_FILE.exists():
        raise FileNotFoundError(
            f"vendored VADER lexicon missing at {LEXICON_FILE}; see sentiment/lexicons/NOTICE.md"
        )
    if str(LEXICON_DIR) not in nltk.data.path:
        nltk.data.path.append(str(LEXICON_DIR))
    return SentimentIntensityAnalyzer(lexicon_file=f"file:{LEXICON_FILE}")


def label_for_compound(compound: float) -> str:
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def score_text(text: str) -> dict[str, float]:
    """Return VADER's raw ``{neg, neu, pos, compound}`` scores for ``text``."""
    return get_analyzer().polarity_scores(text)


@dataclass(frozen=True)
class ScoredHeadline:
    source: str
    title: str
    link: str
    published_at: str
    neg: float
    neu: float
    pos: float
    compound: float
    label: str


def score_headline(headline: Headline) -> ScoredHeadline:
    scores = score_text(headline.title)
    return ScoredHeadline(
        source=headline.source,
        title=headline.title,
        link=headline.link,
        published_at=headline.published_at.isoformat(),
        neg=scores["neg"],
        neu=scores["neu"],
        pos=scores["pos"],
        compound=scores["compound"],
        label=label_for_compound(scores["compound"]),
    )


def score_headlines(headlines: list[Headline]) -> list[ScoredHeadline]:
    return [score_headline(h) for h in headlines]


SCORED_FIELDNAMES = ["source", "title", "link", "published_at", "neg", "neu", "pos", "compound", "label"]


def write_scored_csv(scored: list[ScoredHeadline], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORED_FIELDNAMES)
        writer.writeheader()
        for row in scored:
            writer.writerow(asdict(row))


def read_scored_csv(path: str | Path) -> list[ScoredHeadline]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            ScoredHeadline(
                source=row["source"],
                title=row["title"],
                link=row["link"],
                published_at=row["published_at"],
                neg=float(row["neg"]),
                neu=float(row["neu"]),
                pos=float(row["pos"]),
                compound=float(row["compound"]),
                label=row["label"],
            )
            for row in reader
        ]
