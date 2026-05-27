# EDA Report -- Depression Detection Dataset
_Generated: 2026-05-23T21:49:58_

---

## TL;DR -- Headline Insights

- [WARNING] 96.1% English (detector=langdetect). Some non-English contamination -- consider filtering or treat as noise.
- [WARNING] **5 subreddits** (of 24) have extreme label rate (>=98% or <=2%) with n>=1000. The `subreddit` feature alone can trivially predict label on these. **Action:** run an LGBM ablation WITHOUT subreddit to measure true text+metadata contribution.
- [WARNING] Moderate imbalance -- minority class 19.4%. Class weighting recommended.
- [WARNING] `max_length=256` covers only 89.9%. Use 512 (96.6% coverage) or head+tail truncation.
- [WARNING] Label rate drifts by **0.73** between earliest and latest year. Time-based split (train on older, test on newer) will give more honest generalization estimate than random split.
- [INFO] First-person pronoun use is 1.44x more common in label=1 (matches clinical depression literature -- self-focus marker).
- [INFO] Most informative single feature for label is **subreddit (raw)** (MI = 0.7107 bits). Sanity-check whether this matches expectations -- if it's `subreddit` by a large margin, leakage.
- [INFO] Hour-of-day x day-of-week label rate varies by 0.090 across the grid -- temporal features (cyclical encoding) are worth including in LGBM.
- [OK] Body duplicates negligible (0.35%).
- [OK] Strong text signal: MH-keyword presence is **4.3x** more common in label=1 than label=0. Text model has clear signal to learn.

---

## Recommended Actions (consolidated)

### Before training (data cleaning)
1. Drop rows where `label` is null or `id` is duplicated.
2. **Deduplicate by normalized body** (lowercased, whitespace-collapsed) BEFORE the train/val/test split -- not just by `id`.
3. Filter to English-only if non-English share is significant.
4. Mark `has_body=False` for empty / `[removed]` / `[deleted]` bodies; keep the row (title alone is usable). Drop rows with both title and body empty.
5. Replace URLs / `u/xxx` / `r/xxx` with sentinel tokens (`[URL]`, `[USER]`, `[SUB]`).
6. Bucket rare subreddits (n < 50) into `_rare_` BEFORE feeding LGBM.
7. Clip `upvotes` and `num_comments` at p99.5; store both raw and `log1p`.

### Stage 1 (MentalRoBERTa) decisions
- `max_length = 512` or head+tail truncation strategy.
- Use **Focal Loss** (gamma=2, alpha aligned to positive rate) OR weighted BCE with `pos_weight = neg/pos`.
- Label smoothing 0.05 (helps with noisy Reddit labels).
- Apply the custom head from tech_plan section 5.3 (layer-avg + dual pool + residual FFN + multi-sample dropout).

### Stage 2 (LightGBM) decisions
- `subreddit` as native categorical feature (no one-hot).
- Cyclical encoding (sin/cos) for `hour_utc`, `dow`, `month`.
- Run **ablation without `subreddit`** to measure the true contribution of text + temporal + engagement features.
- `class_weight='balanced'` if imbalanced.

### Splitting
- Strongly consider a **time-based split** (train on older years, test on newer) given the label-rate drift over time.
- Stratify by `label` at minimum; ideally by `(label, subreddit_bucket)`.
- Apply deduplication BEFORE splitting.

---

## Detailed evidence

### 1. Dataset overview
- Rows: **2,470,778**
- Columns: 8  (`id, subreddit, title, body, upvotes, created_utc, num_comments, label`)
- CSV: `C:\Project\Personal Project\DeDe\eng dataset\dataset_raw\reddit_depression_dataset.csv`  (1.17 GB)
- Parquet cache: `C:\Project\Personal Project\DeDe\ver2\eda\outputs\cache\raw.parquet`

### 2. Missingness and ID duplicates

| column | missing | % |
|---|---:|---:|
| id | 4 | 0.0002% |
| subreddit | 20 | 0.0008% |
| title | 23 | 0.0009% |
| body | 461,051 | 18.6602% |
| upvotes | 63 | 0.0025% |
| created_utc | 106 | 0.0043% |
| num_comments | 113,977 | 4.613% |
| label | 106 | 0.0043% |

- Duplicate `id`: 3 (0.0001%)

### 3. Label distribution
- Unique classes: 2 | binary: True
- class **0** -> 1,990,261 (80.5555%)
- class **1** -> 480,411 (19.4445%)

### 4. Subreddit
- Unique subreddits: 24
- Rare (<50 posts): 19 subreddits, 63 rows (0.0025%)
- Positives concentration: top **1** subs cover 50% of label=1 | top 2 cover 80% | top 2 cover 95%.
- **Leaky subreddits** (label rate >=98% or <=2%, n>=1000): 5 found. Top 10:

| subreddit | n | label_rate |
|---|---:|---:|
| teenagers | 1,956,489 | 0.000 |
| depression | 290,049 | 1.000 |
| SuicideWatch | 190,362 | 1.000 |
| happy | 24,609 | 0.000 |
| DeepThoughts | 9,163 | 0.000 |

### 5. Text length and emptiness
- Body empty / [removed] / [deleted]: 462,124 (18.7036%)
- Title empty: 23 (0.0009%)

**Token coverage of (title, body) pairs** (MentalRoBERTa tokenizer):

| max_length | % posts fitting (no truncation) |
|---:|---:|
| 128 | 78.45% |
| 256 | 89.934% |
| 384 | 94.401% |
| 512 | 96.602% |

### 6. Numeric features
- **upvotes**: median=7.0, p95=53.0, p99=521.0, max=128866.0, skew=37.45, neg=0
- **num_comments**: median=7.0, p95=41.0, p99=116.0, max=21131.0, skew=69.83, neg=3

### 7. Temporal
- Date range: 2008-01-26T02:17:12+00:00 -> 2022-12-31T23:58:35+00:00
- Null/invalid timestamps: 106 null, 0 invalid
- Label-rate drift across years (max-min): **0.7316**

### 8. Feature <-> label correlations (Spearman)

| feature | Pearson | Spearman |
|---|---:|---:|
| upvotes | -0.0186 | +0.0602 |
| num_comments | -0.0380 | -0.1973 |
| body_len | +0.3227 | +0.4549 |
| title_len | -0.0211 | -0.0314 |

### 9. Text signals (lexical distinctiveness per class)

| metric | label=0 | label=1 |
|---|---:|---:|
| has_mh_kw_pct | 19.9311 | 84.8122 |
| first_person_rate | 0.073953 | 0.106756 |
| negation_rate | 0.010501 | 0.016592 |
| neg_affect_rate | 0.006023 | 0.01075 |
| mean_words | 52.85 | 198.43 |
| mean_exclaim | 0.3333 | 0.1787 |
| mean_question | 0.5087 | 0.8489 |
| mean_caps_words | 0.8166 | 0.7174 |

**Top-15 most distinctive unigrams for label=1:**

| term | z-score | n(label=1) | n(label=0) |
|---|---:|---:|---:|
| `myself` | +121.46 | 32,258 | 8,247 |
| `feel` | +113.63 | 41,707 | 16,970 |
| `life` | +108.53 | 35,688 | 13,725 |
| `i'm` | +96.25 | 62,065 | 37,982 |
| `depression` | +92.56 | 15,770 | 1,964 |
| `don't` | +83.93 | 37,921 | 21,060 |
| `i've` | +82.81 | 23,009 | 9,596 |
| `can't` | +78.39 | 18,469 | 7,012 |
| `anymore` | +71.16 | 11,656 | 3,219 |
| `depressed` | +65.39 | 8,698 | 1,870 |
| `suicide` | +64.43 | 7,724 | 1,123 |
| `job` | +59.95 | 8,694 | 2,581 |
| `years` | +58.07 | 15,006 | 7,524 |
| `die` | +56.68 | 8,563 | 2,869 |
| `everything` | +56.53 | 11,587 | 5,102 |

**Top-15 most distinctive unigrams for label=0:**

| term | z-score | n(label=0) | n(label=1) |
|---|---:|---:|---:|
| `guys` | +81.27 | 15,826 | 2,200 |
| `girl` | +66.60 | 13,541 | 3,259 |
| `school` | +56.81 | 23,393 | 10,627 |
| `gonna` | +52.62 | 9,034 | 2,360 |
| `https` | +52.29 | 7,018 | 342 |
| `crush` | +51.72 | 7,324 | 286 |
| `girls` | +51.40 | 6,625 | 1,084 |
| `reddit` | +49.02 | 7,670 | 1,951 |
| `class` | +46.36 | 6,791 | 1,705 |
| `post` | +46.19 | 9,961 | 3,566 |
| `amp` | +46.17 | 8,863 | 2,916 |
| `com` | +46.07 | 5,086 | 698 |
| `edit` | +44.03 | 5,678 | 1,283 |
| `teacher` | +42.22 | 4,141 | 416 |
| `sub` | +39.95 | 3,901 | 584 |

**Top-15 most distinctive bigrams for label=1:**

| bigram | z-score | n(label=1) | n(label=0) |
|---|---:|---:|---:|
| `kill myself` | +51.42 | 5,228 | 400 |
| `i'm tired` | +34.84 | 2,344 | 355 |
| `don't what` | +32.24 | 3,558 | 1,427 |
| `i'm not` | +32.13 | 6,968 | 4,073 |
| `feel i'm` | +29.87 | 1,930 | 436 |
| `don't feel` | +29.50 | 2,087 | 570 |
| `hate myself` | +29.49 | 1,785 | 352 |
| `fuck life` | +28.66 | 1,645 | 118 |
| `suicidal thoughts` | +28.32 | 1,524 | 176 |
| `killing myself` | +26.90 | 1,456 | 102 |
| `every day` | +24.09 | 2,658 | 1,293 |
| `i've tried` | +23.95 | 1,100 | 156 |
| `depression anxiety` | +22.92 | 1,016 | 155 |
| `myself i'm` | +21.75 | 919 | 144 |
| `life i'm` | +21.40 | 910 | 160 |

### 10. Data quality
- Language detection (detector=`langdetect`, sample n=50,000): **96.102%** English, 3.898% non-English
- Exact body duplicates: 8,526 rows (0.3451%) in 3,493 groups. Biggest group = 519 copies.
- Title == body: 1,828 (0.074%)
- URL-only posts: 1 (0.0%)
- Emoji-heavy posts: 5,626 (0.2277%)
- Extreme-long posts (>20K chars): 530 (0.0215%)

### 11. Cross-feature interactions

**Mutual information I(feature; label), bits:**

| feature | MI (bits) |
|---|---:|
| subreddit (raw) | 0.7107 |
| body_length_bucket | 0.1705 |
| year | 0.0781 |
| has_body | 0.0326 |
| num_comments_bucket | 0.0300 |
| upvotes_bucket | 0.0101 |
| dow | 0.0011 |
| hour_utc | 0.0005 |

**Label rate by body-length bucket:**

| bucket | n | label rate |
|---|---:|---:|
| empty | 461,029 | 0.0405 |
| 1-50 | 436,608 | 0.0332 |
| 51-200 | 669,820 | 0.0918 |
| 201-500 | 411,253 | 0.2837 |
| 501-1k | 251,900 | 0.4751 |
| 1k-2k | 154,727 | 0.5943 |
| 2k-5k | 72,389 | 0.6748 |
| 5k+ | 12,946 | 0.6648 |

### 12. Generated artifacts
- Flat metrics: `outputs\summary_stats.csv`
- Plots: `outputs\plots/*.png` (34 files)
- Distinctive terms CSVs: `outputs/cache/distinctive_*.csv`
- Duplicate samples: `outputs/cache/duplicate_body_samples.csv`
- Qualitative samples: `outputs/samples/*.md` (6 files)

---

_End of report. Open `outputs/samples/*.md` to eyeball labels manually._
