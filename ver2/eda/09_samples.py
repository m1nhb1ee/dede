"""Qualitative samples for human eyeballing -- the last sanity check.

Numbers do not catch silently broken labels, weird formatting, or templated
spam. Reading 50 random posts per class does. This script dumps:

  - outputs/samples/random_label0.md
  - outputs/samples/random_label1.md
  - outputs/samples/long_label1.md           (long body, label=1)
  - outputs/samples/short_label1.md          (short body, label=1)
  - outputs/samples/no_mh_keywords_label1.md (label=1 with no obvious MH terms)
  - outputs/samples/with_mh_keywords_label0.md (label=0 mentioning MH terms)

Each file is human-readable markdown so the user can review in any editor.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    COL_BODY, COL_ID, COL_LABEL, COL_SUBR, COL_TIME, COL_TITLE, COL_UPVOTES,
    OUT_DIR, SEED,
)
from utils import char_len, load_raw, section, step, update_stats

SAMPLES_DIR = OUT_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

PER_BUCKET = 25

# Reuse the same MH keyword set used in 06_text_signals.py
MH_KEYWORDS = {
    "depress", "depressed", "depression", "anxious", "anxiety",
    "hopeless", "worthless", "useless", "empty", "numb",
    "suicide", "suicidal", "kill", "die", "death",
    "therapy", "therapist", "medication", "meds", "pills",
}


def _has_mh(text: str) -> bool:
    if not isinstance(text, str): return False
    low = text.lower()
    return any(k in low for k in MH_KEYWORDS)


def _render_sample(df: pd.DataFrame, title: str, note: str) -> str:
    lines = [f"# {title}\n", f"_{note}_\n", f"_Showing {len(df)} samples._\n", ""]
    for _, r in df.iterrows():
        sub = r.get(COL_SUBR, "?")
        lbl = r.get(COL_LABEL, "?")
        rid = r.get(COL_ID, "?")
        up  = r.get(COL_UPVOTES, "?")
        body = str(r.get(COL_BODY, "") or "")
        ttl  = str(r.get(COL_TITLE, "") or "")
        if len(body) > 1200: body = body[:1200] + "  ...[truncated]"
        lines.append(f"---")
        lines.append(f"**id**: `{rid}` | **subreddit**: r/{sub} | "
                     f"**label**: {lbl} | **upvotes**: {up}")
        lines.append(f"\n**Title:** {ttl}\n")
        lines.append(f"**Body:** {body}\n")
    return "\n".join(lines)


def _write(name: str, content: str) -> Path:
    path = SAMPLES_DIR / f"{name}.md"
    path.write_text(content, encoding="utf-8")
    step(f"wrote {path.name}")
    return path


def main() -> None:
    section("09 | Qualitative samples for human review")

    df = load_raw()
    df = df.dropna(subset=[COL_LABEL]).copy()
    df[COL_LABEL] = df[COL_LABEL].astype(int)
    df["_body_len"] = char_len(df[COL_BODY])
    n = len(df)
    step(f"{n:,} rows with valid label")

    payload: dict = {"per_bucket": PER_BUCKET, "files": []}

    # ── Random samples per class ──────────────────────────────────────────
    for lbl in sorted(df[COL_LABEL].unique()):
        sub = df[df[COL_LABEL] == lbl].sample(
            n=min(PER_BUCKET, int((df[COL_LABEL] == lbl).sum())),
            random_state=SEED,
        )
        p = _write(f"random_label{lbl}",
                   _render_sample(sub, f"Random sample, label={lbl}",
                                  "Uniformly random from this class."))
        payload["files"].append(str(p))

    # ── Long + short label=1 (length extremes) ────────────────────────────
    pos = df[df[COL_LABEL] == 1]
    if not pos.empty:
        long_pos = pos.nlargest(PER_BUCKET, "_body_len")
        short_pos = pos[pos["_body_len"] > 5].nsmallest(PER_BUCKET, "_body_len")
        payload["files"].append(str(_write(
            "long_label1",
            _render_sample(long_pos, "Longest label=1 posts",
                           "If labels are noisy, long posts let you see whether "
                           "the assignment makes sense."),
        )))
        payload["files"].append(str(_write(
            "short_label1",
            _render_sample(short_pos, "Shortest label=1 posts",
                           "Short positive posts are most likely to be mislabeled "
                           "or to rely on context the model cannot see."),
        )))

    # ── Label-mismatch eyeballs: positive with no obvious MH keyword ──────
    if not pos.empty:
        no_kw = pos[~((pos[COL_TITLE].fillna("") + " " + pos[COL_BODY].fillna("")
                     ).apply(_has_mh))].head(500)
        if len(no_kw):
            sub = no_kw.sample(n=min(PER_BUCKET, len(no_kw)), random_state=SEED)
            payload["files"].append(str(_write(
                "no_mh_keywords_label1",
                _render_sample(sub, "label=1 WITHOUT any common MH keyword",
                               "These are the rows that need ngu nghia / non-keyword "
                               "reasoning. If they look genuinely depressive, the text "
                               "model can earn its keep. If they look unrelated, "
                               "labels may be subreddit-derived rather than content-derived."),
            )))

    # ── Label-mismatch eyeballs: label=0 mentioning MH terms ──────────────
    neg = df[df[COL_LABEL] == 0]
    if not neg.empty:
        with_kw = neg[(neg[COL_TITLE].fillna("") + " " + neg[COL_BODY].fillna("")
                       ).apply(_has_mh)].head(500)
        if len(with_kw):
            sub = with_kw.sample(n=min(PER_BUCKET, len(with_kw)), random_state=SEED)
            payload["files"].append(str(_write(
                "with_mh_keywords_label0",
                _render_sample(sub, "label=0 mentioning MH terms",
                               "These are the hardest negatives. Often they are "
                               "discussions ABOUT depression rather than expressions "
                               "OF it -- exactly what a good model should distinguish."),
            )))

    update_stats("samples", payload)
    section(f"Done. {len(payload['files'])} sample files in {SAMPLES_DIR}\n"
            "Run 10_report.py next to assemble the final report.")


if __name__ == "__main__":
    main()
