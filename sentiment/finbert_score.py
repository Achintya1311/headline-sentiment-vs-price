"""FinBERT sentiment scorer (Day 3).

Scores headline text with ``ProsusAI/finbert``, a BERT model fine-tuned on
financial text (Malo et al., Financial PhraseBank), via HuggingFace
``transformers``. Unlike ``sentiment/vader_score.py``'s vendored lexicon,
the model weights (~440 MB) are too large to commit alongside this
portfolio's small fixture files, so they are downloaded from the
HuggingFace Hub on first use and cached under ``~/.cache/huggingface`` -
no API key needed, the model is public and free, but a network connection
is required at least once. See the README's Limitations section.

FinBERT is a 3-way classifier (positive/negative/neutral) trained on
financial language, so - unlike VADER - there is no hand-picked compound
threshold: the label is whichever class the model assigns the highest
probability. ``compound`` here is a derived quantity (``positive -
negative``) kept only so magnitude comparisons against VADER's compound
score are possible; it is not a model output.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from sentiment.headline import Headline

MODEL_NAME = "ProsusAI/finbert"


@lru_cache(maxsize=1)
def get_pipeline():
    """Load the FinBERT tokenizer + model once per process and cache it.

    Imports ``torch``/``transformers`` lazily so importing this module (or
    the rest of the ``sentiment`` package) never pays their startup cost
    unless FinBERT scoring is actually used.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model, torch


def score_texts(texts: list[str]) -> list[dict[str, float]]:
    """Batch-score ``texts`` and return one ``{positive, negative, neutral}`` dict per text.

    Batching all headlines through the model in one forward pass is
    materially faster on CPU than scoring one at a time (this is why the
    signature takes a list rather than mirroring ``vader_score.score_text``
    exactly).
    """
    if not texts:
        return []

    tokenizer, model, torch = get_pipeline()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    id2label = model.config.id2label
    results = []
    for row in probs:
        by_label = {id2label[i]: float(row[i]) for i in range(len(id2label))}
        results.append(
            {
                "positive": by_label["positive"],
                "negative": by_label["negative"],
                "neutral": by_label["neutral"],
            }
        )
    return results


def score_text(text: str) -> dict[str, float]:
    """Single-text convenience wrapper around ``score_texts``."""
    return score_texts([text])[0]


def label_for_scores(scores: dict[str, float]) -> str:
    """FinBERT's own argmax class - no manual threshold, unlike VADER's compound cutoffs."""
    return max(("positive", "negative", "neutral"), key=lambda label: scores[label])


@dataclass(frozen=True)
class FinbertScoredHeadline:
    source: str
    title: str
    link: str
    published_at: str
    positive: float
    negative: float
    neutral: float
    compound: float
    label: str


def score_headlines(headlines: list[Headline]) -> list[FinbertScoredHeadline]:
    """Score every headline's title with FinBERT, batched in one forward pass."""
    if not headlines:
        return []

    scores = score_texts([h.title for h in headlines])
    scored = []
    for headline, s in zip(headlines, scores):
        scored.append(
            FinbertScoredHeadline(
                source=headline.source,
                title=headline.title,
                link=headline.link,
                published_at=headline.published_at.isoformat(),
                positive=s["positive"],
                negative=s["negative"],
                neutral=s["neutral"],
                compound=s["positive"] - s["negative"],
                label=label_for_scores(s),
            )
        )
    return scored


def score_headline(headline: Headline) -> FinbertScoredHeadline:
    return score_headlines([headline])[0]


SCORED_FIELDNAMES = ["source", "title", "link", "published_at", "positive", "negative", "neutral", "compound", "label"]


def write_scored_csv(scored: list[FinbertScoredHeadline], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORED_FIELDNAMES)
        writer.writeheader()
        for row in scored:
            writer.writerow(asdict(row))


def read_scored_csv(path: str | Path) -> list[FinbertScoredHeadline]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            FinbertScoredHeadline(
                source=row["source"],
                title=row["title"],
                link=row["link"],
                published_at=row["published_at"],
                positive=float(row["positive"]),
                negative=float(row["negative"]),
                neutral=float(row["neutral"]),
                compound=float(row["compound"]),
                label=row["label"],
            )
            for row in reader
        ]
