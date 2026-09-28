"""Raw headline record and CSV storage (Day 1).

A ``Headline`` is exactly what was scraped, before any scoring: source,
title, link, and two timestamps. ``published_at`` is parsed from the feed's
own ``pubDate`` and kept timezone-aware; ``scraped_at`` is when this process
fetched it, in UTC. Day 4 does timestamp-vs-market-hours alignment - this
module only has to preserve what the feed actually said, faithfully.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Headline:
    source: str
    title: str
    link: str
    published_at: datetime
    published_raw: str
    scraped_at: datetime

    def __post_init__(self) -> None:
        if self.published_at.tzinfo is None:
            raise ValueError(f"published_at must be timezone-aware: {self.published_at!r}")
        if self.scraped_at.tzinfo is None:
            raise ValueError(f"scraped_at must be timezone-aware: {self.scraped_at!r}")

    def dedup_key(self) -> tuple[str, str]:
        """A headline is the same headline if the same source republished the same link."""
        return (self.source, self.link)

    def to_row(self) -> dict[str, str]:
        row = asdict(self)
        row["published_at"] = self.published_at.astimezone(timezone.utc).isoformat()
        row["scraped_at"] = self.scraped_at.astimezone(timezone.utc).isoformat()
        return row


FIELDNAMES = ["source", "title", "link", "published_at", "published_raw", "scraped_at"]


def write_csv(headlines: list[Headline], path: str | Path) -> None:
    """Overwrite ``path`` with ``headlines``, sorted oldest-published first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(headlines, key=lambda h: h.published_at)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for h in ordered:
            writer.writerow(h.to_row())


def read_csv(path: str | Path) -> list[Headline]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            Headline(
                source=row["source"],
                title=row["title"],
                link=row["link"],
                published_at=datetime.fromisoformat(row["published_at"]),
                published_raw=row["published_raw"],
                scraped_at=datetime.fromisoformat(row["scraped_at"]),
            )
            for row in reader
        ]


def dedupe(headlines: list[Headline]) -> list[Headline]:
    """Drop repeats of the same (source, link), keeping the first-seen scrape."""
    seen: set[tuple[str, str]] = set()
    kept: list[Headline] = []
    for h in headlines:
        key = h.dedup_key()
        if key in seen:
            continue
        seen.add(key)
        kept.append(h)
    return kept


def merge_and_write(existing_path: str | Path, new_headlines: list[Headline]) -> list[Headline]:
    """Append ``new_headlines`` to whatever is already at ``existing_path`` and rewrite it.

    This is how snapshots accumulate over multiple scrape runs without losing
    headlines an outlet later deletes or rewrites (see the feed-survivorship
    trap in NEXT_STEPS.md) - once a headline is on disk it stays, even if a
    later scrape of the same feed no longer sees it.
    """
    existing_path = Path(existing_path)
    prior = read_csv(existing_path) if existing_path.exists() else []
    merged = dedupe(prior + new_headlines)
    write_csv(merged, existing_path)
    return merged
