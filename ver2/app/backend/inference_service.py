"""Inference service for one social-media post.

The service owns all model logic used by the app:
scrape/translate/clean -> Stage 1 text model -> Stage 2 LightGBM.
"""

from __future__ import annotations

import html
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ver2.preprocess.config import (  # noqa: E402
    ABSOLUTIST_WORDS,
    FIRST_PERSON_WORDS,
    MH_KEYWORDS,
    NEGATIVE_WORDS,
    REMOVED_TOKENS,
    SECOND_PERSON_WORDS,
    URL_TOKEN,
    USER_TOKEN,
    SUB_TOKEN,
)
from ver2.stage1.src.data import encode_head_tail  # noqa: E402
from ver2.stage1.src.model import MentalRoBERTaWithCustomHead  # noqa: E402
from ver2.stage1.src.utils import load_backbone, load_tokenizer  # noqa: E402
from ver2.stage2.src.config import FEATURE_COLS, MODEL_PATH  # noqa: E402


DEFAULT_STAGE1_CKPT = ROOT / "ver2" / "stage1" / "outputs" / "checkpoints" / "full_trainable.safetensors"
DEFAULT_MODEL_DIR = ROOT / "models" / "mental-roberta-base"

_WORD_RE = re.compile(r"[a-zA-Z']+")
_RE_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(\s*https?://[^\)]+\)")
_RE_REF_LINK = re.compile(r"\[([^\]]+)\]\[\d+\]")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_USER_MENTION = re.compile(r"(?:^|(?<=\s))/?u/[A-Za-z0-9_\-]+", flags=re.IGNORECASE)
_RE_SUB_MENTION = re.compile(r"(?:^|(?<=\s))/?r/[A-Za-z0-9_\-]+", flags=re.IGNORECASE)
_RE_MD_BOLD_ITALIC = re.compile(r"\*+([^*\n]+?)\*+")
_RE_MD_STRIKE = re.compile(r"~~([^~\n]+?)~~")
_RE_MD_INLINE_CODE = re.compile(r"`([^`\n]+?)`")
_RE_MD_HEADER = re.compile(r"(?m)^#{1,6}\s*")
_RE_MD_BLOCKQUOTE = re.compile(r"(?m)^>+\s?")
_RE_MD_HRULE = re.compile(r"(?m)^[-*_]{3,}\s*$")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_RE_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\ufeff]")
_RE_AMP_ENTITY = re.compile(r"&\w+;")


@dataclass
class PostInput:
    title: str
    body: str
    upvotes: float
    num_comments: float
    created_utc: int
    source_url: str | None = None


class InferenceService:
    def __init__(
        self,
        stage1_ckpt: Path = DEFAULT_STAGE1_CKPT,
        model_dir: Path | None = DEFAULT_MODEL_DIR,
        stage2_model: Path = MODEL_PATH,
    ) -> None:
        self.stage1_ckpt = stage1_ckpt
        self.model_dir = model_dir if model_dir and model_dir.exists() else None
        self.stage2_model_path = stage2_model
        self._lock = Lock()
        self._loaded = False
        self._tok = None
        self._stage1 = None
        self._device = None
        self._stage2 = None

    @property
    def model_loaded(self) -> bool:
        return self._loaded

    def predict(
        self,
        *,
        url: str | None,
        title: str,
        body: str,
        upvotes: float,
        num_comments: float,
        created_utc: int | None,
        translate: bool,
    ) -> dict[str, Any]:
        post = self.prepare_post(
            url=url,
            title=title,
            body=body,
            upvotes=upvotes,
            num_comments=num_comments,
            created_utc=created_utc,
            translate=translate,
        )
        self.load_models()
        p_text = self.predict_stage1(post.title, post.body)
        features = self.build_stage2_features(post, p_text)
        p_final = float(self._stage2.predict_proba(features[FEATURE_COLS])[:, 1][0])
        return {
            "p_text_stage1": p_text,
            "p_final_depression_risk": p_final,
            "predicted_label_at_0_5": int(p_final >= 0.5),
            "title_en_clean": post.title,
            "body_en_clean": post.body,
            "source_url": post.source_url,
            "note": "Model risk score only; not a medical diagnosis.",
        }

    def prepare_post(
        self,
        *,
        url: str | None,
        title: str,
        body: str,
        upvotes: float,
        num_comments: float,
        created_utc: int | None,
        translate: bool,
    ) -> PostInput:
        if url:
            title, body = scrape_url(url)
            if not (title or body):
                raise ValueError("Could not extract text from URL. Paste the post text manually.")
            source_url = url
        else:
            source_url = None

        title_en = translate_text(title, enabled=translate)
        body_en = translate_text(body, enabled=translate)
        clean_title = "" if is_removed(title_en) else clean_text(title_en)
        clean_body = "" if is_removed(body_en) else clean_text(body_en)
        if not (clean_title or clean_body):
            raise ValueError("Post is empty after cleaning.")

        return PostInput(
            title=clean_title,
            body=clean_body,
            upvotes=max(float(upvotes), 0.0),
            num_comments=max(float(num_comments), 0.0),
            created_utc=int(created_utc or time.time()),
            source_url=source_url,
        )

    def load_models(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if not self.stage1_ckpt.exists():
                raise FileNotFoundError(f"Stage 1 checkpoint not found: {self.stage1_ckpt}")
            if not self.stage2_model_path.exists():
                raise FileNotFoundError(f"Stage 2 model not found: {self.stage2_model_path}")

            tok, _ = load_tokenizer(str(self.model_dir) if self.model_dir else None)
            backbone, _ = load_backbone(str(self.model_dir) if self.model_dir else None)
            stage1 = MentalRoBERTaWithCustomHead(backbone, use_lora=True, num_classes=1)
            state = load_state_dict(self.stage1_ckpt)
            missing, unexpected = stage1.load_state_dict(state, strict=False)
            trained_missing = [k for k in missing if ("lora_" in k) or k.startswith("head.")]
            if trained_missing:
                raise RuntimeError(f"Stage 1 checkpoint is missing trained keys: {trained_missing[:8]}")
            if unexpected:
                print(f"[WARN] Stage 1 unexpected keys: {len(unexpected)}", flush=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            stage1.to(device)
            stage1.eval()

            self._tok = tok
            self._stage1 = stage1
            self._device = device
            self._stage2 = joblib.load(self.stage2_model_path)
            self._loaded = True

    def predict_stage1(self, title: str, body: str) -> float:
        enc = encode_head_tail(title, body, self._tok, max_length=512)
        batch = {
            "input_ids": torch.tensor([enc["input_ids"]], dtype=torch.long, device=self._device),
            "attention_mask": torch.tensor([enc["attention_mask"]], dtype=torch.long, device=self._device),
        }
        with torch.no_grad():
            use_amp = self._device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = self._stage1(**batch)
            logit = out["logits"].detach().float().cpu().numpy().reshape(-1)[0]
        return float(1.0 / (1.0 + math.exp(-float(logit))))

    def build_stage2_features(self, post: PostInput, p_text: float) -> pd.DataFrame:
        title = post.title or ""
        body = post.body or ""
        combined = f"{title} {body}".strip()
        words = _WORD_RE.findall(combined.lower())
        n_words = len(words)
        word_set = set(words)
        title_len = len(title)
        body_len = len(body)

        bins = [-1, 0, 50, 200, 500, 1000, 2000, 5000, 10_000_000]
        body_bucket = int(pd.cut(pd.Series([body_len]), bins=bins, labels=False).iloc[0])

        dt = pd.to_datetime(int(post.created_utc), unit="s", utc=True)
        hour = int(dt.hour)
        dow = int(dt.weekday())
        num_sentences = max(int(len(re.findall(r"[.!?]+", combined))), 1)
        n_upper = len(re.findall(r"[A-Z]", combined))
        n_alpha = len(re.findall(r"[A-Za-z]", combined))

        row: dict[str, Any] = {
            "p_text": p_text,
            "body_len_chars": body_len,
            "body_length_bucket": body_bucket,
            "title_len_chars": title_len,
            "body_to_title_ratio": body_len / (title_len + 1),
            "upvotes_log": np.log1p(post.upvotes),
            "num_comments_log": np.log1p(post.num_comments),
            "comments_per_upvote": post.num_comments / (post.upvotes + 1.0),
            "has_mh_keyword": any(k in combined.lower() for k in MH_KEYWORDS),
            "num_first_person": sum(1 for w in words if w in FIRST_PERSON_WORDS),
            "num_negative_words": sum(1 for w in words if w in NEGATIVE_WORDS),
            "num_exclamations": combined.count("!"),
            "num_questions": combined.count("?"),
            "num_caps_words": len(re.findall(r"\b[A-Z]{2,}\b", combined)),
            "num_ellipsis": combined.count("..."),
            "num_words": n_words,
            "num_absolutist": sum(1 for w in words if w in ABSOLUTIST_WORDS),
            "num_second_person": sum(1 for w in words if w in SECOND_PERSON_WORDS),
            "type_token_ratio": (len(word_set) / n_words) if n_words else 0.0,
            "avg_word_len": (sum(len(w) for w in words) / n_words) if n_words else 0.0,
            "num_sentences": num_sentences,
            "uppercase_ratio": n_upper / (n_alpha + 1),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "is_weekend": dow >= 5,
            "is_night_us_eastern": hour in {4, 5, 6, 7, 8, 9},
            "has_title": bool(title),
            "has_body": bool(body),
        }

        w = max(float(n_words), 1.0)
        row["first_person_rate"] = row["num_first_person"] / w
        row["negative_word_rate"] = row["num_negative_words"] / w
        row["exclamation_rate"] = row["num_exclamations"] / w
        row["question_rate"] = row["num_questions"] / w
        row["caps_word_rate"] = row["num_caps_words"] / w
        row["ellipsis_rate"] = row["num_ellipsis"] / w
        row["absolutist_rate"] = row["num_absolutist"] / w
        row["second_person_rate"] = row["num_second_person"] / w
        row["avg_sentence_len"] = n_words / max(float(num_sentences), 1.0)

        df = pd.DataFrame([row], columns=FEATURE_COLS)
        for c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype("int8")
        return df.astype("float32")


def clean_text(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    if "&" in s:
        s = html.unescape(s)
    s = _RE_ZERO_WIDTH.sub("", s)
    s = _RE_MARKDOWN_LINK.sub(r"\1", s)
    s = _RE_REF_LINK.sub(r"\1", s)
    s = _RE_URL.sub(URL_TOKEN, s)
    s = _RE_USER_MENTION.sub(USER_TOKEN, s)
    s = _RE_SUB_MENTION.sub(SUB_TOKEN, s)
    s = _RE_MD_BOLD_ITALIC.sub(r"\1", s)
    s = _RE_MD_STRIKE.sub(r"\1", s)
    s = _RE_MD_INLINE_CODE.sub(r"\1", s)
    s = _RE_MD_HEADER.sub("", s)
    s = _RE_MD_BLOCKQUOTE.sub("", s)
    s = _RE_MD_HRULE.sub("", s)
    s = _RE_AMP_ENTITY.sub(" ", s)
    s = _RE_MULTI_NEWLINE.sub("\n\n", s)
    s = _RE_MULTI_SPACE.sub(" ", s)
    return s.strip()


def is_removed(s: str) -> bool:
    return not isinstance(s, str) or s.strip().lower() in REMOVED_TOKENS


def load_state_dict(path: Path):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(str(path), map_location="cpu")


def scrape_url(url: str) -> tuple[str, str]:
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("URL scraping needs requests: pip install requests") from e

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    page = resp.text

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        title = _meta_regex(page, "og:title") or _tag_regex(page, "title")
        body = _meta_regex(page, "og:description") or _meta_regex(page, "description")
        return html.unescape(title or ""), html.unescape(body or "")

    soup = BeautifulSoup(page, "html.parser")

    def meta(name: str) -> str:
        tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        return str(tag.get("content", "")).strip() if tag else ""

    title = meta("og:title") or (soup.title.get_text(" ", strip=True) if soup.title else "")
    body = meta("og:description") or meta("description")
    if not body:
        article = soup.find("article")
        body = article.get_text(" ", strip=True) if article else ""
    return title, body


def _meta_regex(page: str, name: str) -> str:
    pat = re.compile(
        rf'<meta[^>]+(?:property|name)=["\']{re.escape(name)}["\'][^>]+content=["\']([^"\']*)["\']',
        re.IGNORECASE,
    )
    m = pat.search(page)
    return m.group(1).strip() if m else ""


def _tag_regex(page: str, tag: str) -> str:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", page, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def translate_text(text: str, enabled: bool) -> str:
    if not enabled or not text.strip():
        return text
    try:
        from deep_translator import GoogleTranslator
    except ImportError as e:
        raise RuntimeError(
            "Translation needs deep-translator. Install it with: "
            "pip install deep-translator, or disable translation."
        ) from e
    return GoogleTranslator(source="auto", target="en").translate(text)
