# Tech Plan: 2-Stage Depression Detection Pipeline

> **Mục tiêu**: Xây dựng hệ thống dự đoán xác suất trầm cảm từ post Reddit, kết hợp tín hiệu ngữ nghĩa (text) và tín hiệu hành vi (metadata) qua kiến trúc 2 tầng (Stage 1: MentalRoBERTa + LoRA, Stage 2: LightGBM).
>
> **Hardware** (UPDATED): Training trên **Kaggle 2× T4 (16GB × 2 = 32GB VRAM total)**. Local dev trên RTX 3050 6GB cho debug + light experiments.
> **Dataset**: 2.47M Reddit posts, 8 columns: `id, subreddit, title, body, upvotes, created_utc, num_comments, label`
> **Context**: Project môn Machine Learning

---

## 0a. DATASET REALITY CHECK (EDA findings, 2026-05-23)

EDA đã chạy đầy đủ trên dataset thật. Trước khi đọc bất kỳ section nào dưới đây, phải hiểu **bản chất thực sự** của dataset, vì nhiều quyết định trong tech_plan đã được điều chỉnh dựa trên những findings này.

### Findings buộc phải biết

| Finding | Số liệu | Ý nghĩa |
|---|---|---|
| **Labels = subreddit name** | MI(subreddit, label) = **0.71 bits** ≈ entropy của label | Task thực ra là phân loại subreddit, không phải detect depression |
| **5 subreddits leaky 100%** | r/teenagers→0, r/depression→1, r/SuicideWatch→1, r/happy→0, r/DeepThoughts→0 | Subreddit feature alone → F1 ≈ 1.0 trivially |
| **r/teenagers dominate** | 1.96M / 2.47M posts (79.2%) | Class imbalance phản ánh cấu trúc subreddit, không phải tỷ lệ trầm cảm thật |
| **Label rate drift 0.73 theo năm** | Khoảng cách max-min giữa các năm 2008-2022 | Random split → over-estimate; cần time-based split |
| **Text signal vẫn có thật** | MH keyword: 84.8% (label=1) vs 19.9% (label=0), lift 4.3x; first-person lift 1.44x | Mô hình text vẫn học được pattern thật, không chỉ memorize subreddit |
| **max_length=256 không đủ** | Token coverage chỉ 89.9% (cần >=95% cho production) | Phải dùng 384 (94.4%) hoặc 512 (96.6%) |
| **Label noise rõ rệt** | Sample r/SuicideWatch có lyrics + bài báo + body rỗng | Cần label smoothing + tolerant loss |
| **Body missing 18.7%** | 462K posts có body rỗng / [removed] / [deleted] | Title-only model phải robust |
| **MentalRoBERTa gated** | HuggingFace yêu cầu auth để access | Cần `huggingface-cli login` trước Phase 2, hoặc fallback roberta-base |
| **Subreddit count rất nhỏ** | Chỉ 24 unique subreddits (19 trong đó rare <50 posts) | "Rare subreddit bucketing" gần như không ý nghĩa |

### Hệ quả về claim của mô hình

> **Cẩn trọng**: Model train trên dataset này **KHÔNG PHẢI** "depression detector" — nó là **"r/depression-style-post detector"**. Sẽ không generalize tốt sang post của người trầm cảm thật ở subreddit khác (r/AskReddit, r/relationship_advice, v.v.). **Phải nói rõ giới hạn này trong report cuối.**

### Quyết định buộc phải điều chỉnh từ findings (đã reflect xuống các section)

1. **Section 4 — Splitting**: random stratified → **time-based** (train ≤ 2020, val 2021, test 2022).
2. **Section 5.5 — max_length**: 256 → **512** (T4 x2 cho phép, coverage 96.6%; ban đầu cân nhắc 384=94.4% nhưng hardware đủ cho 512).
3. **Section 3/6 — Subreddit PURGE (bắt buộc)**: Vì label ≈ subreddit (MI=0.71 bits), subreddit không phải feature mà là **bản sao của target**. **Loại bỏ hoàn toàn subreddit và mọi proxy của nó** khỏi Stage 2: bỏ `subreddit` (categorical), `upvotes_pct_in_subreddit` (cần phân phối toàn subreddit), và `year` (proxy của thành phần subreddit theo năm, đồng thời vỡ dưới time-split). Tiêu chí lọc: *một feature chỉ hợp lệ nếu tính được từ MỘT post đơn lẻ mà không cần biết nó thuộc subreddit nào*. Stage 1 text đã subreddit-blind sẵn (clean thay `r/xxx`→`[SUB]`).
4. **Section 5 — Tokenizer**: thêm fallback `roberta-base` (cùng vocab), document workflow auth cho `mental/mental-roberta-base`.
5. **Section 2 — Cleaning**: bỏ qua rare-subreddit bucketing (chỉ 19 sub rare, 63 rows tổng). Drop hẳn cột `subreddit` sau khi đã có label.
6. **Section 9 — Evaluation**: model duy nhất là subreddit-blind. Vẫn báo cáo **F1 per subreddit** như một *diagnostic* (tách post theo subreddit ở phần đánh giá — KHÔNG đưa subreddit vào model) để phơi bày nếu model overfit theo style của một subreddit cụ thể.

---

## 0. Tổng quan kiến trúc

```
┌──────────────────────────────────────────────────────────────────┐
│                         RAW DATASET                              │
│  id, subreddit, title, body, upvotes, created_utc,               │
│  num_comments, label                                             │
└────────────────────────┬─────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   ┌────────────────┐        ┌────────────────────┐
   │  TEXT STREAM   │        │  METADATA STREAM   │
   │ title + body   │        │ length,    upvotes,│
   │                │        │ num_comments, time │
   └───────┬────────┘        └─────────┬──────────┘
           │                           │
           ▼                           │
   ┌───────────────────┐               │
   │  STAGE 1:         │               │
   │  MentalRoBERTa    │               │
   │  + LoRA           │               │
   │  + Custom Head    │               │
   └───────┬───────────┘               │
           │                           │
           │ P_text (OOF)              │
           │                           │
           └──────────┬────────────────┘
                      ▼
           ┌────────────────────┐
           │  STAGE 2:          │
           │  LightGBM          │
           │  (P_text + meta    │
           │   features)        │
           └─────────┬──────────┘
                     ▼
              P_depression_final
```

**Vì sao kiến trúc này hợp lý:**
- Text model bắt được **ngữ nghĩa**: tự ti, ý định tự hại, tuyệt vọng, mệt mỏi.
- Metadata model bắt được **hành vi**: thời điểm đăng bài (3am vs 2pm), engagement pattern (post nhiều nhưng ít comment), độ dài/cấu trúc bài. **Không dùng subreddit** (nó ≈ label, xem section 0a).
- Hai loại tín hiệu **bổ trợ nhau**, không thay thế.
- Tách rời cho phép **debug độc lập**: nếu ensemble tệ, biết tầng nào yếu.

---

## 1. EDA — Phase bắt buộc trước khi viết bất kỳ dòng code training nào

### 1.1. Sanity checks
- Đếm tổng số rows, số duplicate `id`, số missing per column.
- Phân phối **label**: binary hay multi-class? Tỷ lệ class? (Quan trọng cho loss function)
- Phân phối **subreddit**: bao nhiêu subreddit khác nhau? Top-20 phổ biến? Long-tail?
- Phân phối **độ dài title/body** (số ký tự, số tokens sau khi tokenize):
  - Median, p75, p95, p99
  - % posts có body rỗng / `[removed]` / `[deleted]`
  - % posts vượt 256/512 tokens
- Phân phối **upvotes, num_comments**: skewness, có giá trị âm không, outliers?
- Phân phối **created_utc**: dải thời gian (min, max), gap, mật độ theo tháng/năm.

### 1.2. Quan hệ giữa features và label
- Label rate theo subreddit (subreddit nào predict gần như deterministically?)
- Label rate theo hour-of-day UTC, day-of-week
- Correlation upvotes/num_comments với label
- Phân phối độ dài body theo label

### 1.3. Output cụ thể của EDA
Một notebook `01_eda.ipynb` với:
- Tất cả phân phối ở trên (histogram, boxplot)
- Bảng tóm tắt `summary_stats.csv`
- File `data_issues.md` ghi rõ các vấn đề cần xử lý ở phase cleaning

> **Quyết định phụ thuộc EDA**: max sequence length (256 vs 384 vs 512), có cần subsample subreddit dominant không, có dùng oversampling không.

---

## 2. Data Cleaning & Preparation

### 2.1. Cleaning chung (áp dụng cho cả 2 stream)

| Vấn đề | Xử lý |
|---|---|
| Duplicate `id` | Drop, giữ row đầu tiên |
| `label` null/invalid | Drop row |
| Body = `[removed]` / `[deleted]` / `null` / `""` | Đánh dấu `has_body=False`, set body = `""` |
| Title null | Drop row (title luôn có với Reddit posts) |
| Title = `""` | Đánh dấu `has_title=False` |
| `upvotes` null | Fill 0 |
| `num_comments` null | Fill 0 |
| `created_utc` null | Drop hoặc fill bằng median (báo cáo số lượng) |

### 2.2. Cleaning riêng cho TEXT stream

| Bước | Lý do |
|---|---|
| Lower-case? | **KHÔNG** — RoBERTa case-sensitive, "WHY ME" ≠ "why me" về cường độ cảm xúc |
| Remove URLs | Có — thay bằng token `[URL]` |
| Remove user mentions `u/xxx` | Thay bằng `[USER]` |
| Remove subreddit mentions `r/xxx` | Thay bằng `[SUB]` (vì subreddit đã là feature riêng) |
| Markdown (`**` `***` `[text](url)`) | Strip markdown formatting, giữ plain text |
| Emoji | **GIỮ** — emoji mang signal cảm xúc rất mạnh |
| Excessive whitespace, newlines | Normalize về single space/newline |
| Censored words (f***, s***) | Giữ nguyên |
| HTML entities (`&amp;`, `&gt;`) | Decode |

### 2.3. Cleaning riêng cho METADATA stream

| Feature | Xử lý |
|---|---|
| `upvotes` | Clip outliers ở p99.5. Lưu cả raw và `log1p(upvotes)` |
| `num_comments` | Tương tự upvotes |
| `subreddit` | **Chỉ dùng để derive `label`, rồi DROP cột.** Không đưa vào feature set (label ≈ subreddit, MI=0.71). |
| `created_utc` | Validate trong dải hợp lý (>2005, <now). Convert sang datetime UTC |

### 2.4. Output

Sau cleaning, lưu **2 file song song** chia sẻ chung `id`:
- `posts_text.parquet`: `id, title_clean, body_clean, has_title, has_body, label`
- `posts_meta.parquet`: `id, upvotes, num_comments, created_utc, label` — **không có `subreddit`** (đã drop sau khi derive label). Giữ riêng một cột `subreddit` CHỈ trong file eval/diagnostic (`posts_eval_meta.parquet`) để phân nhóm khi báo cáo F1 per-subreddit, tuyệt đối không nạp vào model.

> Lưu parquet vì nhanh hơn CSV nhiều cho 2M rows.

---

## 3. Feature Engineering

### 3.1. Text features (cho Stage 1)

**Không cần feature engineering thủ công** cho text. Chỉ cần:
- Tokenize `(title, body)` thành cặp với `[SEP]` ở giữa
- Truncation strategy: **head+tail** (giữ title nguyên, cắt giữa body) — vì post depression thường dài + conclusion ở cuối quan trọng
- **`max_length=512`** [UPDATED for T4 x2: cover 96.6% post; chỉ 3.4% cần head+tail truncation]
- Body budget per row: ~500 tokens (sau khi trừ CLS + title + 2 SEP + "..."), chia ~170 head / ~330 tail

### 3.2. Metadata features (cho Stage 2)

#### A. Numerical features
| Feature | Công thức | Lý do |
|---|---|---|
| `upvotes_log` | `log1p(upvotes)` | Distribution lệch nặng |
| `num_comments_log` | `log1p(num_comments)` | Tương tự |
| `comments_per_upvote` | `num_comments / (upvotes + 1)` | Post depression thường nhiều comment relative đến upvote |
| ~~`upvotes_pct_in_subreddit`~~ | ~~percentile rank trong subreddit~~ | **BỎ — leak ẩn**: cần phân phối toàn subreddit để tính ⇒ vi phạm tiêu chí "tính từ 1 post". Nếu cần normalize, dùng percentile **global** (toàn dataset), không theo subreddit. |
| `title_len_chars`, `body_len_chars` | `len(text)` | Post depression thường dài hơn |
| `body_to_title_ratio` | `body_len / (title_len + 1)` | |
| `has_body` | bool | |
| `has_title` | bool | |

#### B. Text-derived features (light NLP, KHÔNG dùng deep model)

| Feature | Lý do |
|---|---|
| `num_exclamations`, `num_questions`, `num_ellipsis` | Cường độ cảm xúc |
| `num_caps_words` (số từ ALL CAPS) | Kêu cứu, nhấn mạnh |
| `num_first_person` (count "i, im, me, my, mine, myself" — first-person **số ít**) | Self-focus là marker depression mạnh nhất trong y văn (Rude 2004, Tackman 2019) |
| `num_negative_words` (đếm từ trong negative lexicon nhỏ) | Quick signal |
| `num_words` | Độ dài token-level (mẫu số cho các rate ở §6.2) |

**[IMPLEMENTED] Tier-2 psycholinguistic markers** (thêm khi rebuild `meta_features.parquet`, subreddit-blind, tính từ 1 post):

| Feature | Lý do |
|---|---|
| `num_absolutist` (always, never, nothing, completely…) | Tuyệt đối hóa: cao bất thường ở forum depression/SI (Al-Mosaiwi & Johnson-Laird 2018) |
| `num_second_person` (you, your, yourself…) | Thấp hơn khi tự-tập-trung |
| `type_token_ratio` (unique/total words) | Đa dạng từ vựng thấp hơn ở người trầm cảm |
| `avg_word_len` | Mean ký tự / token |
| `num_sentences` | Đếm ranh giới câu `[.!?]+` |
| `uppercase_ratio` | Tỉ lệ chữ HOA = cường độ cảm xúc / shouting |

#### C. Temporal features từ `created_utc`

UTC **vẫn dùng được**, dù không biết timezone của user, vì 2 lý do:
1. **r/depression có ~70% user US-based**. UTC hour có correlation đủ mạnh để LGBM khai thác.
2. **Day-of-week** không phụ thuộc timezone nhiều (lệch tối đa 1 ngày).

| Feature | Cách tính | Encoding |
|---|---|---|
| `hour_utc` | `datetime.hour` | Cyclical: `sin(2πh/24)`, `cos(2πh/24)` |
| `dow` (day of week) | `datetime.weekday()` | Cyclical: `sin(2πd/7)`, `cos(2πd/7)` |
| `is_weekend` | `dow >= 5` | Boolean |
| `month` | `datetime.month` | Cyclical |
| ~~`year`~~ | ~~`datetime.year`~~ | **BỎ — proxy subreddit + vỡ time-split**: (1) drift theo năm thực chất là *thành phần subreddit đổi theo năm* ⇒ year lén mang subreddit-composition vào; (2) time-split train ≤2020 / test 2022 ⇒ mọi row test có year=2022 (giá trị model **chưa từng thấy**) → extrapolation harmful, không neutral. |
| `is_night_us_eastern` | `1 if hour_utc in [4..9]` | Heuristic |

> **Cyclical encoding** (sin/cos) quan trọng để LGBM hiểu hour 23 và hour 0 gần nhau.

**Fallback**: Nếu không tin UTC, chạy LGBM 2 lần (có/không feature thời gian), so sánh AUC. Nếu chênh <0.005 → bỏ qua.

#### D. Categorical features

> **PURGE subreddit**: Vì label ≈ subreddit (MI=0.71 bits ≈ entropy của label), đưa `subreddit` vào model = đưa target vào ⇒ leak trivial (F1≈1.0 giả tạo). **Không có feature categorical nào dùng subreddit.** Cột `subreddit` đã bị drop sau khi lấy label (section 2). `has_title` (post-level boolean) vẫn giữ ở nhóm A.

#### E. Feature từ Stage 1 (key feature!)
| Feature | Cách tính |
|---|---|
| `p_text` | OOF prediction từ Stage 1 (xem mục 7) |

---

## 4. Splitting Strategy [REVISED post-EDA]

### Vì sao đổi từ random sang time-based

EDA phát hiện **label rate drift theo năm = 0.73** (max-min giữa các năm 2008-2022). Cause: dataset thu thập post từ các subreddit khác nhau ở các thời điểm khác nhau (r/depression nhiều ở 2010-2014, r/teenagers nhiều ở 2017-2022).

Random split → training data và test data có cùng distribution → metric đẹp giả tạo. Time-based split → realistic generalization estimate.

### 4.1. Time-based split (primary)

```
Train:   created_utc <= 2020-12-31    (~80% data, năm 2008-2020)
Val:     created_utc in 2021           (~10%)
Test:    created_utc in 2022           (~10%)
```

- **Trước khi split**: dedup theo normalized body (lowercased + collapsed whitespace), không chỉ dedup `id`.
- Drop rows có `label` null, `created_utc` null/invalid, hoặc cả title+body đều rỗng.
- Lưu split assignment vào `data/splits/split_assignments.parquet` (id → train/val/test) cho reproducibility.

### 4.2. Stratified random split (secondary, để compare)

Cũng train một version với stratified random split để **báo cáo gap** giữa 2 setting → bằng chứng rõ ràng cho temporal drift.

```
seed=42, stratify=label, train/val/test = 80/10/10
```

### 4.3. K-fold OOF (cho stacking) [REVISED → 3-fold]

Trên **Train** (≤2020, 1,815,216 rows), chia **3 folds** (đã giảm từ 5 do compute budget Kaggle) **stratified by label** với `KFOLD_SEED=2024` (`StratifiedKFold(shuffle=True)`). Mỗi fold ~605,072 rows, label-rate cân bằng (~19.8% positive). Mỗi fold dùng để generate `p_text_oof` cho stage 2.

> **Quan trọng**: KHÔNG dùng time-based fold trong K-fold OOF — sẽ làm fold cuối có quá ít data 2020. Stratified by label trong train set là OK.

> **3 vs 5 fold**: tech_plan ban đầu đề 5-fold; đã chốt **3-fold** vì (1) mỗi fold ~3.5-4h trên Kaggle T4 → 3 fold chạy song song 3 notebook = ~4h tổng, vừa quota; (2) với 1.8M train, mỗi train-subset của 3-fold vẫn ~1.21M rows — đủ lớn, OOF variance không đáng kể so với 5-fold. Đổi `N_FOLDS=3` trong `preprocess/config.py` (đã re-run `02_split.py`).

### 4.4. Stratified per-subreddit evaluation set

Riêng cho final evaluation, tạo thêm **8 subsets** từ test set (2022 posts):
- 1 subset per top-5 subreddit
- 1 subset cho "small subreddits" (gộp các sub n<10000 trong test)
- 1 subset "high-confidence false positives" (label=0 nhưng MH keywords)
- 1 subset "high-confidence false negatives" (label=1 nhưng body rỗng / short)

Cho phép báo cáo F1 per-subreddit → phơi bày overfit subreddit. **Lưu ý**: subreddit ở đây CHỈ dùng để *phân nhóm khi đánh giá*, không bao giờ là input của model (xem section 6.3).

---

## 5. Stage 1 — Text Model (MentalRoBERTa)

### 5.0. Data journey cho Stage 1 — từ raw đến batch train [IMPLEMENTED]

Toàn bộ đường đi của data, đúng theo code (`preprocess/01_clean.py`, `02_split.py`,
`stage1/src/data.py`, `stage1/src/train.py`). Số liệu là thật.

```
RAW (EDA cache, ~2.47M posts: id, subreddit, title, body, upvotes,
     created_utc, num_comments, label)
  │
  ▼  [preprocess/01_clean.py]  → posts_clean.parquet (2,470,017 rows)
  │   • Drop: null label/id, duplicate id, created_utc không hợp lệ
  │          (ngoài 2005..2030), title+body CẢ HAI rỗng sau clean.
  │   • has_body/has_title: body|title ∈ {[removed],[deleted],"",null,nan,
  │          none,deleted,removed} → set "" + cờ False.
  │   • Text clean (title+body): URL→[URL], u/x→[USER], r/x→[SUB] (subreddit-
  │          blind ngay từ text), strip markdown, decode HTML entity.
  │          GIỮ nguyên case + emoji (signal cảm xúc).
  │   • Numeric: upvotes/num_comments fill 0, clip âm→0, clip trên ở p99.5.
  │   • Dedup body: với body ≥ 40 ký tự, hash(normalize(title)||normalize(body)),
  │          giữ bản OLDEST theo created_utc (giữ thứ tự thời gian cho time-split).
  │   • Drop cột phái sinh; giữ subreddit ở cột riêng (eval-only, KHÔNG vào model).
  │
  ▼  [preprocess/02_split.py]  → splits.parquet (id, time_split, random_split, fold)
  │   • time_split (PRIMARY, theo năm): train ≤2020 = 1,815,216 | val 2021 =
  │          442,119 | test 2022 = 212,682.  (lý do: label-rate drift 0.73/năm)
  │   • random_split (SECONDARY): 80/10/10 stratified (chỉ để báo cáo gap drift).
  │   • fold: StratifiedKFold(n_splits=3, shuffle, seed=2024) CHỈ trên time-train
  │          → mỗi fold ~605,072 rows, label-rate cân bằng. (val/test fold = -1)
  │
  ▼  [stage1/src/data.py : build_or_load_tokenized]  → token_cache (HF Dataset, cached)
  │   • merge posts_clean + splits trên id.
  │   • Head+tail tokenize: title KHÔNG cắt; body cắt head 35% / tail 65%, chèn
  │          " ... " giữa; layout RoBERTa [CLS] title [SEP][SEP] body [SEP];
  │          max_length=512 (phủ 96.6%, chỉ 3.4% phải cắt).
  │   • label → labels (float32, cho BCE); drop raw title/body; cache xuống đĩa
  │          (key theo tokenizer+max_len → mọi fold dùng chung 1 cache).
  │
  ▼  [stage1/src/data.py : make_fold_splits(fold=k)]   (chọn theo mode)
  │   • train = time_split=="train" AND fold≠k   (~1,210,144 rows)
  │   • val   = time_split=="train" AND fold==k   (605,072 — OOF slice)
  │   • test  = time_split=="test"                (212,682)
  │   • mode full: train=time-train, val=time-val 2021, test=test 2022.
  │
  ▼  [stage1/src/train.py : train_one]   (xử lý per-fold trước khi vào Trainer)
  │   • Negative undersampling 0.5 trên TRAIN (seed=SEED+fold, val/test GIỮ nguyên):
  │          fold 0 ví dụ 1.21M → 239,760 pos + 485,192 neg = 724,952 rows.
  │   • pos_weight = neg/pos tính lại từ train ĐÃ undersample ≈ 2.02 (cho BCE).
  │   • Carve val_full(605K) → eval subset 60K + quick subset 5K (stratified,
  │          DISJOINT, assert trong code); val_full giữ lại để predict OOF cuối.
  │
  ▼  [Trainer]   • PadCollator: dynamic pad tới câu dài nhất trong batch (pad_id=1).
                 • group_by_length=True khi train (giảm padding ~25-35%), =False
                   khi predict (tránh lệch p_text↔label).
                 • per_device_batch 8 × 2 GPU × grad_accum 4 = effective batch 64.
```

> Tóm tắt 3 cơ chế chống lệch class: (1) `pos_weight` ≈2.02 trong BCE, (2)
> undersample neg 0.5 trên train, (3) label smoothing 0.05. val/test luôn ở
> **phân phối thật** nên metric không bị thổi phồng. Chi tiết: §5.6 / §5.6a / §5.6b.

### 5.1. MentalRoBERTa có classifier head chưa?

**Chưa.** `mental/mental-roberta-base` chỉ là backbone đã pretrain MLM, không có classifier head. Khi load qua `AutoModelForSequenceClassification.from_pretrained(...)`, HuggingFace tự thêm head mặc định (Linear-tanh-Dropout-Linear), random init.

→ Ta sẽ **thay** head mặc định bằng custom head (mục 5.3).

### 5.1a. Access lưu ý: MentalRoBERTa là gated repo [post-EDA]

`mental/mental-roberta-base` trên HuggingFace là **gated repository** — yêu cầu request access + auth. Steps:

1. Tạo HuggingFace account, vào https://huggingface.co/mental/mental-roberta-base và click "Request access".
2. Sau khi được approve (thường vài giờ): `huggingface-cli login` với access token.
3. Verify: `python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mental/mental-roberta-base')"` chạy không lỗi.

**Fallback nếu không có access**: dùng `roberta-base` (tokenizer **giống hệt**, chỉ khác weights). Performance sẽ thấp hơn MentalRoBERTa ~2-4 điểm F1 vì không có domain pretrain. Đã được verify trong EDA — tokenizer interchangeable.

```python
# Fallback chain — tokenizer này identical cho cả 2 model
TOKENIZER_FALLBACKS = ["mental/mental-roberta-base", "roberta-base"]
```

### 5.2. LoRA hay full fine-tune? → LoRA

| Tiêu chí | LoRA | Full fine-tune |
|---|---|---|
| VRAM (3050 6GB) | ~3.5 GB | ~5 GB (rất tight) |
| Train time / epoch (2M samples) | ~3-4h | ~5-6h |
| Số params trainable | ~590K | 110M |
| Risk overfit | Thấp | Cao hơn |
| Khả năng iterate (thử nhiều configs) | Cao | Thấp |
| F1 loss vs full FT | -0.5 đến -1.5 | baseline |

→ **LoRA cho phép chạy 2-3 experiment trong cùng thời gian 1 full fine-tune**. Gap nhỏ với 2M data.

### 5.3. Kiến trúc đầy đủ — Custom Head Design

> **Triết lý**: Giữ lại 80% lợi ích của thiết kế phức tạp với 30% phức tạp. Mỗi component phải **dễ giải thích, dễ debug, gần như free**.

#### Sơ đồ

```
       Input (input_ids, attn_mask, token_type_ids)
                         │
                         ▼
┌─────────────────────────────────────────────┐
│  MentalRoBERTa backbone (LoRA-adapted)       │
│  output_hidden_states=True                   │
│  → 13 tensors [B, L, 768]                    │
└────────────────────┬────────────────────────┘
                     │
            ┌────────▼─────────┐
            │  ① Layer Average │  ← weighted sum 4 layer cuối
            │     (4 params)    │
            └────────┬─────────┘
                     │ [B, L, 768]
            ┌────────▼─────────┐
            │  ② Dual Pool      │  ← CLS + masked mean
            │     concat        │
            └────────┬─────────┘
                     │ [B, 1536]
            ┌────────▼─────────┐
            │  ③ Residual FFN   │  ← 1 block, Transformer-style
            │     1536→768      │
            └────────┬─────────┘
                     │ [B, 768]
            ┌────────▼─────────┐
            │  ④ MSD Head       │  ← Multi-Sample Dropout K=5
            └────────┬─────────┘
                     ▼
                  logits
           (+ label smoothing 0.05)
```

**Trainable params tổng**: ~3.5M (LoRA ~590K + head ~3M). Backbone đóng băng.

#### ① Layer-wise Weighted Average (4 layer cuối)

```
weights = softmax(learnable_4_scalars)         # init = [1,1,1,1]
γ       = learnable_scalar                     # init = 1.0
H_agg = γ · Σ weights[i] · hidden_states[-(i+1)]   # i = 0,1,2,3
```

- **Lý do**: Layer cuối quá MLM-specific. Mix với 3 layer trước (10, 11, 12) cho representation cân bằng giữa syntactic và semantic.
- **Cost**: 5 params. Đúng nghĩa free.
- **Gain**: +0.5–1.0 F1 (theo nhiều paper về BERT fine-tuning).

#### ② Dual Pool (CLS + Mean)

```
cls_vec  = H_agg[:, 0, :]                          # [B, 768]
mean_vec = masked_mean(H_agg, attention_mask)       # [B, 768]
pooled   = concat([cls_vec, mean_vec])              # [B, 1536]
```

`masked_mean` implementation:
```
mask_expanded = attention_mask.unsqueeze(-1).float()
sum_h = (H_agg * mask_expanded).sum(dim=1)
count = mask_expanded.sum(dim=1).clamp(min=1)
mean_vec = sum_h / count
```

- **Lý do**: CLS = global summary do attention. Mean = fallback khi CLS không tốt (RoBERTa không pretrain NSP, CLS không được optimize cho sentence-level).
- **Bỏ max và attention pool**: marginal benefit, tăng phức tạp + VRAM.

#### ③ Residual FFN Block (Transformer-style)

```python
def forward(x):                              # x: [B, 1536]
    # Down-project vào working dim
    x = Linear(1536 → 768)(x)                # entry projection

    # Residual FFN block (kiểu Transformer)
    h = LayerNorm(x)
    h = Linear(768 → 1536)(h)
    h = GELU(h)
    h = Dropout(0.1)(h)
    h = Linear(1536 → 768)(h)
    h = Dropout(0.1)(h)
    return x + h                              # residual
```

- **Lý do**: Residual đảm bảo gradient flow + không mất thông tin. Pre-LayerNorm cho stability. Expand-contract (768→1536→768) giữ capacity.
- **Init trick**: Linear cuối của block init bằng zeros hoặc rất nhỏ (0.001) → ban đầu block là identity (`x + 0 = x`), training stable hơn.
- **1 block là đủ**: nhiều block không cải thiện rõ với head này, chỉ tốn VRAM.

#### ④ Multi-Sample Dropout Output

```python
def head(x):                                  # x: [B, 768]
    x = LayerNorm(x)

    # K=5 dropout masks → 5 logits → average
    logits_list = []
    for _ in range(5):
        x_drop = Dropout(p=0.5)(x)
        logits = output_linear(x_drop)        # shared Linear(768→num_classes)
        logits_list.append(logits)

    return stack(logits_list).mean(dim=0)     # [B, num_classes]
```

- **Lý do**: Như ensemble ngầm 5 model với shared weights → giảm variance, tăng generalization.
- **Cost**: 5× Linear cuối, gần như không đáng kể.
- **Linear dùng chung weights**, chỉ dropout mask khác nhau.

#### Khởi tạo trọng số

| Component | Init |
|---|---|
| Layer aggregation weights | All 1.0 (uniform start) |
| Layer aggregation `γ` | 1.0 |
| Linear layers trong head | Xavier/Glorot uniform |
| **Residual block output Linear** | Init zeros hoặc 0.001 |
| LayerNorm | weight=1, bias=0 |
| Output Linear (MSD) | Xavier |

### 5.4. LoRA config

```
target_modules: ["query", "value"]   # attention only
r: 16
lora_alpha: 32                        # alpha/r = 2.0, standard
lora_dropout: 0.1
bias: "none"
```

**Trainable**:
- LoRA adapters: ~590K
- Custom head (5.3): ~3M
- **Tổng: ~3.5M params trainable** (~3.2% của 110M backbone)

### 5.5. Training config

| Hyperparameter | Giá trị | Lý do |
|---|---|---|
| `max_length` | **512** | T4 x2 cho phép; coverage 96.6% |
| `per_device_batch` | **8** | Rollback từ 16 do OOM với batch 16 trên T4 16GB (gradient_checkpointing + layer-agg 4 layer ăn nhiều activation) |
| `gradient_accumulation` | **4** | Effective batch = 8 × 2 GPU × 4 accum = **64** (giữ nguyên eff batch khi giảm per-device) |
| **n_gpus** | **2 (DataParallel)** | Kaggle 2× T4 16GB |
| `fp16` | True | T4 chỉ hỗ trợ fp16 (không bf16) |
| `gradient_checkpointing` | **True** | Bật làm safety net OOM (~20% chậm hơn, ~40% ít VRAM) — cần thiết với max_len=512 + 4-layer agg |
| `optimizer` | **AdamW thường** (PyTorch, `adamw_torch`) | T4 16GB đủ; bỏ 8-bit cho ổn định. Discriminative LR qua custom optimizer (LoRA vs head) |
| `lr` (LoRA adapters) | 5e-4 | LoRA cần LR cao hơn full FT |
| `lr` (head) | 1e-4 | Head random init, vừa phải |
| `weight_decay` | 0.01 | |
| `warmup_ratio` | 0.06 | |
| `lr_scheduler` | Linear decay sau warmup | |
| `num_epochs` | **1** | MentalRoBERTa đã domain-pretrained + 1.2M train/fold → 1 epoch đủ hội tụ (verified: f1_macro 0.88→0.93 trong nửa epoch đầu). Early-stop hiếm khi kích hoạt trước khi hết epoch. |
| `eval_strategy` | **Every 1500 steps** | Eval trên subset 60K (~7 min/lần) → ~7-8 eval/epoch |
| `save_strategy` | **Every 1500 steps** (= eval) | Checkpoint trùng cadence eval; `load_best_model_at_end=True` |
| `save_total_limit` | **2** | Giữ tối đa 2 checkpoint + best (best không bị xoá) |
| `early_stopping_patience` | **2** (eval cycles) | Dừng nếu f1_macro không tăng ≥0.001 trong 2 lần eval liên tiếp |
| `early_stopping_threshold` | 0.001 | Ngưỡng tối thiểu coi là "cải thiện" |
| `metric_for_best_model` | `f1_macro` | Không phải accuracy/pr_auc (user request) |
| `group_by_length` | True | Gom sample cùng độ dài → giảm padding ~25-35% |
| `dataloader_num_workers` | 2 (kaggle) / 0 (local) | Local=0 cho Windows |
| `dataloader_pin_memory` | True (kaggle) / False (local) | Kaggle dư RAM |

### 5.6. Loss function [FINALIZED post-EDA: binary]

EDA chốt: **binary**, 19.4% positive (không multi-class). Loss đã implement:
- **`BCEWithLogitsLoss(pos_weight)`** — `pos_weight = neg_count / pos_count`, **tính lại runtime từ train fold thực tế** (sau undersampling ≈ 2.02, xem 5.6a).
- **Label smoothing 0.05** (soft target `y* = y·(1−ε) + 0.5·ε`) chống noisy Reddit labels → tạo loss floor ~0.19, nên loss ~0.3 với F1 0.93 là hợp lý.
- Bỏ Focal Loss: BCE+pos_weight đã đủ; thêm focal không cải thiện rõ mà thêm hyperparameter.

### 5.6a. Negative undersampling (Plan B class imbalance) [IMPLEMENTED]

Ngoài `pos_weight`, áp **per-fold negative undersampling** trên TRAIN (val/test giữ NGUYÊN phân phối thật):
- Flag `--undersample_neg 0.5`: giữ 50% negatives trong train.
- Ví dụ fold 0: 1.21M train → giữ 239,760 pos + 485,192 neg = **724,952 rows**; `pos_weight` tự compute = 485K/240K ≈ **2.02**.
- Lý do: tốc độ (train trên ít data hơn) + cân bằng nhẹ. Vì val/test intact → metric vẫn đo trên phân phối thật.

> **Calibration caveat cho Stage 2**: undersampling làm `p_text` bị **đẩy cao** (positive bias). LightGBM tree-based chỉ quan tâm *ranking* nên không ảnh hưởng. Nhưng nếu cần probability calibrated tuyệt đối, phải hiệu chỉnh (Platt/isotonic) — ghi rõ trong report.

### 5.6b. Eval strategy — 3 dataset disjoint [IMPLEMENTED]

Từ val slice của fold (605K rows), chia 3 tập **disjoint** (assert trong code) phục vụ 3 mục đích khác nhau:

```
val_full (605K) ──┬── eval subset (60K stratified)  → Trainer eval / early-stop, mỗi 1500 step (~7 min)
                  ├── quick subset (5K stratified)   → QuickEvalCallback mỗi 100 step (~20s, log F1/recall mid-train)
                  └── full val 605K                  → trainer.predict() CUỐI training → OOF parquet
```

- eval subset & quick subset **disjoint** (seed=42) để quick-eval không leak vào quyết định early-stop.
- OOF cuối cùng predict trên **toàn bộ 605K** (không phải subset) → unbiased.

> **[IMPLEMENTED] Safeguard căn chỉnh p_text↔label**: `group_by_length=True` tăng tốc train
> nhưng khi `trainer.predict()` nó **sắp xếp lại** output theo độ dài → `p_text` lệch khỏi
> `id/label` (AUC sập ~0.5 dù model tốt). Code **tắt `group_by_length` trước mọi predict**
> và kiểm tra cứng `mean p_text(pos) − p_text(neg) > 0.05`, nếu vi phạm thì **raise, không
> ghi file** (áp dụng cả train.py lẫn predict.py/reinfer_fold).

### 5.7. VRAM estimate trên Kaggle T4 16GB (per GPU) [UPDATED]

| Khoản | Ước lượng (batch 8, seq 512, gradient_checkpointing ON) |
|---|---|
| Backbone fp16 frozen | 220 MB |
| LoRA adapters + custom head fp16 + grad + Adam | ~120 MB |
| Activations (batch 8, seq 512, **checkpointing on**) | ~5-6 GB |
| Hidden states 4 layer (for layer-agg) | ~1.2 GB |
| Optimizer buffers + scaler | ~50 MB |
| CUDA context + framework overhead | ~1 GB |
| **Tổng / GPU** | **~8-9 GB / 16 GB** ✓ |
| **Effective batch (2 GPU × 8 batch × 4 accum)** | **64** |

> **Lịch sử OOM**: thử `per_device_batch=16` (effective 16×2×2=64) → **OOM** trên T4 16GB vì activation của 4-layer-agg + max_len 512 vượt budget. Rollback về **8×4** (cùng eff batch 64) + bật gradient_checkpointing. Ổn định, ~8-9 GB/GPU. Không tăng batch thêm để giữ margin an toàn.

**Local dev trên 3050 6GB** (debug only):
- `max_length=256`, batch=4, grad_accum=8, fp16, gradient_checkpointing, 8-bit Adam
- Chạy được trên subset 100K rows để verify pipeline trước khi push lên Kaggle

### 5.8. Output của Stage 1 [IMPLEMENTED — actual filenames]

Mỗi run (`--mode fold --fold k`) ghi vào `stage1/outputs/`:

| Mode | File | Nội dung |
|---|---|---|
| `fold k` | `checkpoints/fold_{k}/checkpoint-*/` | HF Trainer checkpoint (LoRA+head+optimizer state) |
| `fold k` | `predictions/p_text_oof_fold{k}.parquet` | predict trên val slice của fold k (605K rows) — `id, p_text, label` |
| `fold k` | `predictions/p_text_test_fold{k}.parquet` | predict trên test (213K) — để ensemble qua các fold |
| `full`   | `predictions/p_text_val.parquet` | predict trên time-val 2021 |
| `full`   | `predictions/p_text_test.parquet` | predict trên test 2022 |

Sau khi xong cả 3 fold, chạy `predict.py` để ráp:
```bash
python predict.py --assemble_oof    # gộp 3 fold OOF → p_text_oof_train.parquet (1.82M, mỗi id 1 lần)
python predict.py --assemble_test   # avg 3 fold test → p_text_test_ensemble.parquet (213K)
```

> **Lưu ý Kaggle**: checkpoint nằm trong `/kaggle/working` → **mất khi Factory Reset / hết session**. Deliverable thực sự là các file `.parquet` (nhỏ, ~6-8 MB) — download ngay. Model weights chỉ cần nếu muốn re-infer (xem section 10), khi đó zip best checkpoint hoặc chỉ **LoRA+head (~16 MB**, lean `*_trainable.safetensors` — LoRA ~590K + head ~3M params; CELL 8 notebook). File lean này dùng được với `predict.py --reinfer_fold`.

---

## 6. Stage 2 — LightGBM Meta-model

### 6.1. Vì sao LightGBM

| Lý do | |
|---|---|
| Mixed features (numerical + categorical) | ✓ native |
| Handle missing values | ✓ native, không cần impute |
| Handle boolean/low-card categorical (has_body, has_title) | ✓ native |
| Robust với feature scaling | ✓ tree-based |
| Fast train trên CPU (24GB RAM thoải mái cho 2M rows) | ✓ |
| Phổ biến trong ML project, dễ defend trong report | ✓ |

So với XGBoost: nhanh hơn 2-3x, RAM ít hơn, categorical native. So với CatBoost: dễ tune hơn.

### 6.2. Feature set cho LGBM [REVISED post-EDA — SUBREDDIT-BLIND]

EDA findings: MI(subreddit, label)=0.71 bits (≈ max entropy), MI(hour_utc, label)=0.0005.

**Nguyên tắc lọc feature**: một feature chỉ hợp lệ nếu **tính được từ MỘT post đơn lẻ mà không cần biết subreddit của nó**. Mọi thứ vi phạm (subreddit, percentile-trong-subreddit, year-as-composition-proxy) bị loại.

Đây là **danh sách chính xác** các cột trong `meta_features.parquet` (đã build subreddit-blind), cộng `p_text` join từ Stage 1:

```
KEY FEATURE (must):
  p_text                    ← OOF prediction từ Stage 1 (đã subreddit-blind: r/xxx→[SUB])

Length / structure (high-signal post-EDA):
  body_len_chars            ← Spearman với label = +0.45 (cao nhất)
  body_length_bucket        ← MI = 0.17 bits, label_rate đi từ 4% → 67% theo bucket
  title_len_chars
  body_to_title_ratio

Engagement:
  upvotes_log               ← magnitude đọc trực tiếp trên post (hợp lệ)
  num_comments_log          ← Spearman = -0.20
  comments_per_upvote

Lexical / style (text-derived, từ 1 post):
  has_mh_keyword (bool)     ← lift 4.3x; very strong feature
  num_first_person, num_negative_words
  num_exclamations, num_questions, num_caps_words, num_ellipsis
  num_words

Tier-2 psycholinguistic markers [IMPLEMENTED — raw counts/measures]:
  num_absolutist            ← tuyệt đối hóa (Al-Mosaiwi 2018)
  num_second_person
  type_token_ratio          ← đa dạng từ vựng
  avg_word_len
  num_sentences
  uppercase_ratio

Engineered rates [IMPLEMENTED — tính trong stage2/src/data.py, count/num_words]:
  first_person_rate, negative_word_rate, exclamation_rate, question_rate,
  caps_word_rate, ellipsis_rate, absolutist_rate, second_person_rate,
  avg_sentence_len (= num_words / num_sentences)
  → chuẩn hóa theo độ dài: tách tín hiệu style khỏi length (ratio mượt mà
    cây không biểu diễn được trong 1 split)

Temporal (low-signal but cheap, post-EDA):
  hour_sin, hour_cos        ← MI = 0.0005 bits, gần như vô dụng
  dow_sin, dow_cos          ← MI = 0.0011 bits, vô dụng
  is_weekend, is_night_us_eastern
  → giữ cho ablation, nhưng expect ~0 lift

Boolean:
  has_title, has_body       ← has_body MI = 0.033 bits

ĐÃ LOẠI (subreddit hoặc proxy của nó):
  ✗ subreddit                  ← MI=0.71 ≈ entropy label ⇒ là bản sao target
  ✗ upvotes_pct_in_subreddit   ← cần phân phối toàn subreddit để tính
  ✗ year                       ← proxy thành phần subreddit + vỡ time-split

Total model features = 30 base (p_text + 29 cột meta) + 9 engineered rates
= **39**, KHÔNG có subreddit ở bất kỳ dạng nào.
(verify: `meta_features.parquet` có 31 cột = 29 features + id + label;
 9 rate được engineer thêm trong stage2/src/data.py lúc build_dataset)
```

### 6.3. Một model duy nhất, subreddit-blind [REVISED]

Không train "2 versions có/không subreddit". Vì subreddit ≈ label, version-có-subreddit chỉ là **lookup table giả** (F1≈1.0 vô nghĩa), không đáng để build hay maintain. Toàn bộ pipeline dùng **một feature set subreddit-blind duy nhất** (section 6.2).

**Subreddit chỉ xuất hiện ở khâu ĐÁNH GIÁ, không phải training**:

| Dùng subreddit ở đâu | Cho phép? | Mục đích |
|---|---|---|
| Feature đưa vào LGBM | ❌ KHÔNG | (là target trá hình) |
| Phân nhóm để report F1 per-subreddit | ✅ CÓ | diagnostic: model có overfit style 1 subreddit không |
| Cross-subreddit holdout (xem dưới) | ✅ CÓ | đo "content understanding" thực sự |

**Cross-subreddit generalization test (đề xuất nâng lên làm eval chính thực sự đo content):**
- Train: chỉ posts từ r/teenagers, r/depression, r/SuicideWatch
- Test: posts từ r/happy, r/DeepThoughts (subreddit model **chưa từng thấy** lúc train)
- Vì model subreddit-blind, nó **không thể** dùng subreddit lookup ⇒ F1 ở đây phản ánh đúng khả năng hiểu nội dung.

> Trong báo cáo: "We deliberately exclude subreddit (and its proxies) from all model inputs, because MI(subreddit, label)=0.71 bits ≈ H(label) means subreddit is effectively a copy of the target; including it yields a trivial F1≈1.0 lookup. Our reported F1 is the subreddit-blind content-based performance."

### 6.3. Model config baseline

```
objective: binary  (hoặc multiclass)
metric: binary_logloss + auc
boosting_type: gbdt
num_leaves: 63
max_depth: -1
learning_rate: 0.05
n_estimators: 2000 (với early stopping)
min_child_samples: 100  (chống overfit trên feature thưa)
reg_alpha: 0.1
reg_lambda: 0.1
subsample: 0.8
colsample_bytree: 0.8
class_weight: 'balanced'  (nếu imbalanced)
early_stopping_rounds: 100
```

### 6.4. Hyperparameter tuning

**Optuna với 3-fold stratified CV trên Train**, optimize `f1_macro` (dùng lại 3-fold của OOF cho nhất quán).

Search space:
```
num_leaves:        [31, 63, 127, 255]
learning_rate:     log-uniform [0.01, 0.1]
min_child_samples: [50, 100, 200, 500]
reg_alpha:         log-uniform [1e-3, 1.0]
reg_lambda:        log-uniform [1e-3, 1.0]
colsample_bytree:  [0.6, 0.8, 1.0]
subsample:         [0.6, 0.8, 1.0]
```

50 trials với pruning là đủ.

### 6.5. Output của Stage 2

1. Best LGBM model: `models/stage2_lgbm.pkl`
2. Feature importance plot
3. SHAP values trên sample của test set
4. Final predictions: `predictions/p_final_test.parquet`

---

## 7. Stacking & Out-of-Fold (OOF) [REVISED → 3-fold, multi-notebook]

### 7.1. Vấn đề data leakage

**Nếu làm sai**: Train MentalRoBERTa trên toàn bộ time-train (1.82M) → predict trên chính 1.82M đó → `p_text` overfit → LGBM học "p_text ≈ label" → test set tệ.

### 7.2. Giải pháp: OOF predictions (3-fold)

```
Chia time-train (1,815,216 rows, ≤2020) thành 3 folds stratified by label
(StratifiedKFold shuffle=True, KFOLD_SEED=2024). Mỗi fold ~605,072 rows.

For fold k in [0..2]:
    train_subset = time-train \ fold_k     (~1.21M rows; rồi undersample neg 0.5 → ~725K)
    val_subset   = fold_k                   (605K, KHÔNG undersample)

    Train MentalRoBERTa+LoRA trên train_subset
    Predict trên val_subset → p_text_oof_fold{k}.parquet (605K rows)
    Predict trên test (2022, 213K)  → p_text_test_fold{k}.parquet

assemble_oof:  concat 3 fold OOF → p_text_oof_train.parquet (1.82M, mỗi id đúng 1 lần)
assemble_test: avg 3 fold test    → p_text_test_ensemble.parquet (213K)
```

→ `p_text_oof` là unbiased estimator của `P(label | text)`: mỗi row được predict bởi model KHÔNG thấy nó lúc train.

### 7.3. Cho val và test (mode `full`)

Train **một model riêng trên toàn bộ time-train 1.82M** (`--mode full`) → predict time-val 2021 + test 2022 một lần. Đây là model "production" dùng khi inference (section 10).

### 7.4. Trade-off của OOF [REVISED]

**Cost thực tế**: train MentalRoBERTa **4 lần** (3 fold + 1 full). Mỗi fold ~3.5-4h trên Kaggle T4 x2 (train tới early-stop/hết epoch + 2 lần predict 605K & 213K).

**Chiến lược Kaggle (đã chốt)**: chạy **song song trên nhiều notebook/account**:
- Notebook 1: `--mode fold --fold 0`
- Notebook 2: `--mode fold --fold 1`
- Notebook 3: `--mode fold --fold 2`
- Notebook 4: `--mode full`

→ 3 fold song song = ~4h tổng thay vì ~12h tuần tự. Mỗi notebook download `.parquet` về, gộp local bằng `predict.py`.

**Vì sao 3-fold (không 5)**: với 1.82M data, train-subset mỗi fold của 3-fold vẫn ~1.21M — đủ lớn để OOF variance không đáng kể. 5-fold tốn thêm 2 lần train mà gain marginal. Holdout 80/20 thì mất 80% data cho stacking → loại.

---

## 8. End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA PREP (~1h, local) [DONE]                          │
│  ├─ preprocess/01_clean.py    → posts_clean.parquet              │
│  ├─ preprocess/02_split.py    → splits.parquet (time + 3-fold)   │
│  └─ preprocess/03_features.py → meta_features.parquet (subreddit-│
│                                  blind) + eval_subreddit.parquet │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 2: STAGE 1 TRAINING (~4h wall-clock nếu chạy song song)   │
│  Kaggle, mỗi notebook 1 job (xem section 7.4):                   │
│  ├─ train.py --mode fold --fold 0   → p_text_oof_fold0, test_fold0│
│  ├─ train.py --mode fold --fold 1   → p_text_oof_fold1, test_fold1│
│  ├─ train.py --mode fold --fold 2   → p_text_oof_fold2, test_fold2│
│  ├─ train.py --mode full            → p_text_val, p_text_test    │
│  └─ (local) predict.py --assemble_oof / --assemble_test         │
│       → p_text_oof_train.parquet (1.82M), p_text_test_ensemble   │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 3: METADATA FEATURES [DONE ở Phase 1]                     │
│  (meta_features.parquet đã build sẵn; join p_text ở Phase 4)     │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 4: STAGE 2 TRAINING (~30 min, local CPU) [SCAFFOLD DONE]  │
│  ├─ tune.py (Optuna 50 trials, 3-fold CV)      → best_params.json│
│  ├─ train_final.py (full train, best HP)       → stage2_lgbm.pkl │
│  │                                  + feature_importance.{csv,png}│
│  └─ predict.py → p_final_{val,test}.parquet                     │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 5: EVALUATION (~30 min)                                   │
│  ├─ evaluate.py → metrics_report.json, plots/                   │
│  │   (+ per-subreddit F1 join eval_subreddit.parquet, diagnostic)│
│  └─ error_analysis.ipynb                                         │
└─────────────────────────────────────────────────────────────────┘
```

> **Lưu ý**: Stage 2 đã có **scaffold thật** trong `stage2/src/` (`config.py, data.py,
> tune.py, train_final.py, predict.py`) — đã dry-test end-to-end trên OOF fold 1+2.
> Train/tune thật chờ đủ 3 fold OOF (fold 0 đang chạy). Stage 1 (`train.py`,
> `predict.py`) đã hoàn chỉnh; `predict.py` có thêm `--reinfer_fold` để tái tạo
> prediction một fold từ trọng số đã lưu (lean LoRA+head safetensors).

Tổng thời gian active: **~5-6h** (Stage 1 ~4h song song trên Kaggle + Stage 2/eval ~1-2h local).

---

## 9. Evaluation Strategy

### 9.1. Metrics chính

| Metric | Mục đích |
|---|---|
| **F1 macro** | Metric chính cho imbalanced binary |
| **ROC-AUC** | Threshold-independent ranking quality |
| **PR-AUC** | Quan trọng hơn ROC-AUC khi imbalanced |
| **Recall @ Precision=0.9** | Use case y tế: ưu tiên catch positive |
| **Precision @ Recall=0.9** | Use case sàng lọc |
| **Brier Score** | Calibration của probability |

### 9.2. So sánh model variants [REVISED — SUBREDDIT-BLIND]

Tất cả variant đều subreddit-blind. So 3 mức để đo đóng góp của từng tầng:

| Variant | Features | Mục đích |
|---|---|---|
| **A: Stage 1 only (text)** | `p_text > threshold` | text signal đơn thuần |
| **B: Stage 2 only (no p_text)** | meta + text-derived, KHÔNG p_text | metadata-only baseline |
| **C: Full ensemble** | `p_text` + meta (section 6.2) | **kết quả chính của hệ thống** |

→ Diễn giải:
- **C − A**: metadata thực sự thêm bao nhiêu vào text signal → expect nhỏ-vừa
- **A − B**: text model hơn metadata-only baseline bao nhiêu → measure of text understanding
- **C** là con số chính được report (đã subreddit-blind, không trivial).

### 9.3. Diagnostics

- **Calibration curve**: probability có well-calibrated không?
- **Confusion matrix** ở threshold 0.5 và threshold optimal (theo F1)
- **Feature importance** từ LGBM: P_text có rank #1 không?
- **SHAP** trên sample test set
- **Subgroup analysis**: F1 per subreddit
- **Error analysis**: random sample 50 false positives + 50 false negatives, đọc manually, tìm pattern

### 9.4. Ablation study (cho report) [REVISED — SUBREDDIT-BLIND]

**Block 1 — Layer contribution** (số liệu chính, tất cả subreddit-blind):

| Variant | F1 macro | F1 per top-5 sub (diagnostic) |
|---|---|---|
| **C: Full ensemble (p_text + meta)** | **? (số chính)** | ? |
| B: Metadata-only (no p_text) | ? | ? |
| A: Stage 1 text only | ? | ? |
| **Cross-subreddit holdout (train 3 sub → test 2 sub khác)** | **? (content-understanding thực)** | n/a |

**Block 2 — Architecture choices**:

| Variant | F1 (subreddit-blind) |
|---|---|
| Full FT thay vì LoRA | ? |
| Head baseline (CLS + Linear) vs Custom Head (5.3) | ? |
| Custom Head không layer-agg | ? |
| Custom Head không residual FFN | ? |
| Custom Head không MSD | ? |

**Block 3 — Split strategy** (justify time-based split):

| Split | F1 (full ensemble) |
|---|---|
| Time-based (train ≤2020, test 2022) | ? (REAL) |
| Random stratified 80/10/10 | ? (inflated) |
| **Gap = drift effect** | ? |

---

## 10. Inference Pipeline (Production-style)

```python
# Pseudocode flow
def predict(post: dict) -> float:
    # 1. Clean
    title = clean_text(post["title"])
    body = clean_text(post["body"])

    # 2. Text → P_text (head+tail tokenize, max_length=512 — khớp training)
    enc = encode_head_tail(title, body, tokenizer, max_length=512)
    with torch.no_grad():
        p_text = sigmoid(text_model(**enc).logits).item()

    # 3. Build metadata features (SUBREDDIT-BLIND — không có subreddit/year/pct)
    meta = build_meta_features(post)
    meta["p_text"] = p_text

    # 4. LGBM → P_final
    p_final = lgbm_model.predict_proba([meta])[0, 1]

    return p_final
```

Latency budget: ~80ms (50ms text inference + 30ms feature + LGBM).

---

## 11. Reproducibility & Tooling

### 11.1. Project structure

Cấu trúc **thực tế** (đã implement):
```
DeDe/ver2/
├── eda/outputs/                  # data_issues.md, summary stats [DONE]
├── preprocess/
│   ├── config.py                 # N_FOLDS=3, paths, columns, lexicons
│   ├── 01_clean.py               # → posts_clean.parquet
│   ├── 02_split.py               # → splits.parquet (time + 3-fold + random)
│   ├── 03_features.py            # → meta_features.parquet (subreddit-blind)
│   │                             #   + eval_subreddit.parquet (eval-only)
│   ├── utils.py
│   └── outputs/                  # parquet outputs + stats.json
├── stage1/
│   ├── src/
│   │   ├── config.py             # hyperparameters, HW profiles
│   │   ├── data.py               # head+tail tokenize, make_fold_splits, PadCollator
│   │   ├── model.py              # MentalRoBERTaWithCustomHead + CustomHead
│   │   ├── train.py              # train_one(), QuickEvalCallback, --mode fold|full
│   │   ├── predict.py            # assemble_oof / assemble_test / reinfer / reinfer_fold
│   │   └── utils.py              # compute_metrics_binary, load_backbone/tokenizer
│   ├── stage1_code.zip           # bundle để upload Kaggle
│   └── kaggle_notebook_template.py
├── stage2/                       # [SCAFFOLD DONE] LightGBM meta-model
│   └── src/
│       ├── config.py             # 39 feature (30 base + 9 engineered), LGBM/Optuna cfg
│       ├── data.py               # build_dataset: join meta+p_text, engineer rates
│       ├── tune.py               # Optuna 3-fold CV → best_params.json
│       ├── train_final.py        # fit full train → stage2_lgbm.pkl + importance
│       └── predict.py            # p_final_{val,test}.parquet + metrics
└── tech_plan.md
```

### 11.2. Seed control

- `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`
- LightGBM `random_state=42`
- Sklearn `random_state=42` cho train_test_split
- **Lưu seed vào config**, log mọi nơi.

### 11.3. Experiment tracking

- **Weights & Biases** (free tier) hoặc **MLflow** hoặc đơn giản là log file JSON.
- Log: config, metrics per eval step, final test metrics, model size, training time.

### 11.4. Dependencies (pin versions)

```
torch==2.x
transformers==4.x
peft==0.x            # cho LoRA
bitsandbytes==0.x    # 8-bit optimizer
datasets
lightgbm
optuna
scikit-learn
pandas, pyarrow
shap                 # phân tích
matplotlib, seaborn
```

---

## 12. Rủi ro & Mitigation

| Rủi ro | Khả năng | Mitigation |
|---|---|---|
| Subreddit feature dominate (vd `r/depression` → label=1 100%) → data leakage trivial | **Đã xảy ra** (MI=0.71) | **PURGE**: drop subreddit + mọi proxy (pct-in-subreddit, year) khỏi model. Confirm bằng cross-subreddit holdout. (Section 0a, 6.2) |
| OOM trên 3050 dù đã optimize | Trung bình | Giảm max_length 256→192, batch 8→4, aggregate 2 layer thay 4 |
| OOF training quá lâu | Trung bình | **Đã giảm 5→3 fold** (default); chạy song song nhiều notebook Kaggle (~4h wall-clock); undersample neg 0.5 |
| Label noise (Reddit labels thường tự khai báo) | Trung bình | Label smoothing 0.05; discuss trong report |
| Temporal drift (post 2015 vs 2023) | Thấp | **Time-based split để TEST drift** (KHÔNG dùng `year` làm feature — nó là subreddit-proxy + vỡ time-split). |
| LGBM "overfit" leak channel ngoài subreddit | Trung bình | `min_child_samples` cao; audit feature importance — nếu một meta feature rank #1 bất thường, kiểm tra xem có phải subreddit-proxy ẩn |
| Calibration kém của text model | Trung bình | Platt scaling hoặc isotonic regression trên val |
| bitsandbytes lỗi trên Windows | Trung bình | Fallback sang `adamw_torch` (+0.5 GB RAM) |

---

## 13. Timeline đề xuất

| Tuần | Việc |
|---|---|
| **W1** | EDA, cleaning, splitting, build meta features. Setup repo. |
| **W2** | Stage 1: implement custom head, train 1 fold sanity check, debug. |
| **W3** | Stage 1: train full 3-fold OOF + final model (song song nhiều notebook Kaggle). |
| **W4** | Stage 2: feature engineering, train LGBM, Optuna tuning. |
| **W5** | Evaluation, ablation (head components, LoRA vs FT), error analysis. |
| **W6** | Viết report, slide, polish. |

---

## 14. Tổng kết quyết định kỹ thuật [REVISED post-EDA]

| Quyết định | Lựa chọn | Lý do ngắn |
|---|---|---|
| Text model | `mental/mental-roberta-base` (fallback: `roberta-base`) | Domain match; gated repo nên có fallback |
| Classifier head | Custom: Layer-Avg + Dual Pool + Residual FFN + MSD | Sweet spot performance/complexity |
| Fine-tune mode | **LoRA** (r=16, q+v) | Iterate nhanh, đủ tốt |
| Trainable params | ~3.5M (LoRA + head) | ~3% backbone, vẫn expressive |
| **Max length** | **512** (Kaggle T4 x2, coverage 96.6%) | Hardware upgraded |
| **Per-device batch** | **16** × 2 GPU × 2 accum = **64 effective** | T4 16GB cho phép |
| **Gradient checkpointing** | **Off** (Kaggle T4) / On (local 3050) | Tốc độ trên Kaggle |
| **Parallelism** | **DataParallel** (2× T4) | Đủ cho 2 GPU; DDP không cần |
| Truncation | `only_second` (giữ title) | Title luôn quan trọng |
| Loss | Focal/weighted BCE + label smoothing 0.05 | Imbalance (19.4% positive) + noisy labels |
| Stacking | **3-fold OOF** stratified by label | Tránh leakage; 3-fold đủ với 1.82M data |
| Meta model | **LightGBM** | Mixed features, fast, native categorical |
| HP tuning | Optuna 50 trials với 3-fold CV | Standard |
| UTC features | Giữ nhưng expect minimal lift (MI ~0) | Đã đo |
| **Subreddit feature** | **PURGE hoàn toàn** (+ proxy: pct-in-subreddit, year) | EDA: MI=0.71 ≈ H(label) → là bản sao target, không phải feature |
| **Splitting** | **Time-based** (≤2020 / 2021 / 2022) | EDA: drift 0.73 between years |
| Body length features | High priority (Spearman 0.45, MI 0.17) | Đo từ EDA |
| MH keyword feature | Boolean has_mh_keyword (lift 4.3x) | Free signal |
| Rare subreddit bucketing | Bỏ (chỉ 19 sub rare, 63 rows) | EDA: không có rare-tail thực |
| Evaluation | F1 macro + PR-AUC + per-subreddit F1 (diagnostic) + cross-subreddit holdout | Phơi bày overfit style 1 subreddit |
| Test set | Đụng **1 lần ở cuối** | Tránh test leakage |
| **Primary metric** | **F1 (full ensemble, subreddit-blind)** | Số "thực", không trivial |

---

## 14a. Kaggle Workflow [NEW]

Vì training chạy trên Kaggle 2× T4 16GB, có 4 ràng buộc cụ thể cần handle:

### Ràng buộc Kaggle

| Ràng buộc | Workaround |
|---|---|
| **30h GPU/tuần** free tier | Mỗi fold ~3.5-4h × (3 fold + 1 full) = ~14-16h cho Stage 1. Đủ cho 1-2 lần iterate. |
| **12h timeout/notebook run** | 1 fold/notebook (~4h) thừa thời gian. Chạy song song 3-4 notebook (khác account) → ~4h wall-clock. |
| **Internet OFF mặc định** | Upload dataset + model weights làm Kaggle Dataset trước |
| **MentalRoBERTa gated repo** | Download local trước, upload lên Kaggle Dataset, load offline |
| **T4 không hỗ trợ bf16** | Dùng fp16 (đã set) |
| **Multi-GPU strategy** | DataParallel (1 dòng code) — đủ cho 2 GPU; DDP không cần |

### Workflow đề xuất

```
LOCAL (3050):
  1. Run Phase 2 (clean + split + features) [DONE]
  2. Verify Stage 1 code chạy được trên 100K subset
  3. Download MentalRoBERTa weights:
     pip install huggingface-hub
     huggingface-cli login  # cần token + đã được approve access
     huggingface-cli download mental/mental-roberta-base \
         --local-dir ./models/mental-roberta-base

UPLOAD TO KAGGLE:
  4. Upload posts_clean.parquet + splits.parquet làm Kaggle Dataset "dede-preprocessed"
  5. Upload MentalRoBERTa weights folder làm Kaggle Dataset "mentalroberta-weights"
     (hoặc skip nếu dùng roberta-base fallback — có sẵn trên Kaggle internet pre-cache)

KAGGLE NOTEBOOK (1 fold / notebook, chạy song song):
  6. Settings: Accelerator = GPU T4 x2, Internet = Off (sau khi pre-cache)
  7. Notebook chạy: python train.py --mode fold --fold {k} \
                       --model_dir /kaggle/input/.../mental-roberta-base \
                       --undersample_neg 0.5
     → p_text_oof_fold{k}.parquet + p_text_test_fold{k}.parquet → download
  8. Notebook riêng: python train.py --mode full   → p_text_val + p_text_test

LOCAL (3050):
  9. predict.py --assemble_oof / --assemble_test (gộp 3 fold)
  10. Train Stage 2 (LightGBM) — CPU-bound, nhanh
  11. Final evaluation
```

### Code skeleton cho multi-GPU

```python
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MentalRoBERTaWithCustomHead(...).to(device)

# Auto-detect và wrap với DataParallel nếu có >1 GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
    model = nn.DataParallel(model)

# Với HuggingFace Trainer, multi-GPU tự động — không cần wrap
# trainer = Trainer(model=model, args=args, ...)
# trainer.train()  # auto-detects 2 GPUs
```

### Time budget cho Stage 1 (3-fold + full) [REVISED]

| Item | Time / fold |
|---|---|
| Train tới early-stop/hết 1 epoch (~725K rows sau undersample, eff batch 64) | ~2-2.5h |
| Intermediate eval trên 60K subset (4-7 lần) | ~30-40 min |
| Quick eval overhead (mỗi 100 step × ~20s) | ~25 min |
| Final predict val_full (605K) | ~30-50 min |
| Final predict test (213K) | ~10-20 min |
| **Tổng / fold** | **~3.5-4h** |

- **Sequential** (1 account): 3 fold + 1 full ≈ ~14-16h → vẫn fit quota 30h/tuần.
- **Parallel** (3-4 account/notebook): ~4h wall-clock.

→ Đủ chạy full Stage 1 trong 1 tuần Kaggle, còn dư quota cho 1 lần iterate.

---

## 15. Bước tiếp theo

1. ~~Confirm tech plan~~ ✓
2. ~~Phase 1 — EDA~~ ✓ (xem `ver2/eda/outputs/data_issues.md`)
3. ~~Phase 2 — Cleaning + Splitting + Features~~ ✓ (`ver2/preprocess/outputs/`)
4. **Phase 3 — Stage 1 (Kaggle T4 x2)** (đang chuẩn bị):
   - Implement custom head (section 5.3) + head+tail truncation
   - Sanity check trên 100K subset (local 3050)
   - Setup MentalRoBERTa access OR confirm roberta-base fallback
   - Upload dataset + model lên Kaggle
   - Run 3-fold OOF + final full-train model (~4h song song / ~14-16h tuần tự trên Kaggle)
5. **Phase 4 — Stage 2 (Local CPU)**: train **một model subreddit-blind** (feature set section 6.2), Optuna tuning
6. **Phase 5 — Evaluation**: variant A/B/C (text / meta-only / ensemble) + per-subreddit diagnostic + cross-subreddit holdout + temporal split comparison
7. **Phase 6 — Report**: emphasize subreddit-derived label nature (MI=0.71), lý do PURGE subreddit, cross-subreddit holdout = số đo content-understanding thực, time-based realistic estimate
