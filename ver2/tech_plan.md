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
2. **Section 5.5 — max_length**: 256 → **384**.
3. **Section 6/9 — Subreddit ablation**: từ optional → **bắt buộc**, là số liệu chính trong report.
4. **Section 5 — Tokenizer**: thêm fallback `roberta-base` (cùng vocab), document workflow auth cho `mental/mental-roberta-base`.
5. **Section 2 — Cleaning**: bỏ qua rare-subreddit bucketing (chỉ 19 sub rare, 63 rows tổng).
6. **Section 9 — Evaluation**: thêm **stratified F1 per subreddit** trong report, vì nếu r/teenagers F1=1.0 và r/depression F1=1.0 nhưng cross-subreddit prediction tệ → con số tổng đẹp lừa người đọc.

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
   │ title + body   │        │ subreddit, upvotes,│
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
- Metadata model bắt được **hành vi**: subreddit (r/depression vs r/jokes), thời điểm đăng bài (3am vs 2pm), engagement pattern (post nhiều nhưng ít comment).
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
| `subreddit` | Lower-case, strip. Subreddit xuất hiện <50 lần → gộp vào `"_rare_"` |
| `created_utc` | Validate trong dải hợp lý (>2005, <now). Convert sang datetime UTC |

### 2.4. Output

Sau cleaning, lưu **2 file song song** chia sẻ chung `id`:
- `posts_text.parquet`: `id, title_clean, body_clean, has_title, has_body, label`
- `posts_meta.parquet`: `id, subreddit, upvotes, num_comments, created_utc, label`

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
| `upvotes_pct_in_subreddit` | percentile rank trong subreddit | Normalize cross-subreddit |
| `title_len_chars`, `body_len_chars` | `len(text)` | Post depression thường dài hơn |
| `body_to_title_ratio` | `body_len / (title_len + 1)` | |
| `has_body` | bool | |
| `has_title` | bool | |

#### B. Text-derived features (light NLP, KHÔNG dùng deep model)

| Feature | Lý do |
|---|---|
| `num_exclamations`, `num_questions` | Cường độ cảm xúc |
| `num_caps_words` (số từ ALL CAPS) | Kêu cứu, nhấn mạnh |
| `num_first_person` (count "I", "me", "my", "myself") | Self-focus là dấu hiệu depression nổi tiếng trong tâm lý học |
| `num_negative_words` (đếm từ trong negative lexicon nhỏ) | Quick signal |
| `num_sentences` | |
| `avg_word_length` | |

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
| `year` | `datetime.year` | Numerical (bắt drift) |
| `is_night_us_eastern` | `1 if hour_utc in [4..9]` | Heuristic |

> **Cyclical encoding** (sin/cos) quan trọng để LGBM hiểu hour 23 và hour 0 gần nhau.

**Fallback**: Nếu không tin UTC, chạy LGBM 2 lần (có/không feature thời gian), so sánh AUC. Nếu chênh <0.005 → bỏ qua.

#### D. Categorical features
| Feature | Encoding |
|---|---|
| `subreddit` | LightGBM native categorical (set `categorical_feature=['subreddit']`) |

> **KHÔNG dùng one-hot** cho subreddit — quá nhiều unique values.

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

### 4.3. K-fold OOF (cho stacking)

Trên **Train** (≤2020), chia 5 folds **stratified by label** với seed=2024. Mỗi fold dùng để generate `p_text_oof` cho stage 2.

> **Quan trọng**: KHÔNG dùng time-based fold trong K-fold OOF — sẽ làm fold cuối có quá ít data 2020. Stratified by label trong train set là OK.

### 4.4. Stratified per-subreddit evaluation set

Riêng cho final evaluation, tạo thêm **8 subsets** từ test set (2022 posts):
- 1 subset per top-5 subreddit
- 1 subset cho "small subreddits" (gộp các sub n<10000 trong test)
- 1 subset "high-confidence false positives" (label=0 nhưng MH keywords)
- 1 subset "high-confidence false negatives" (label=1 nhưng body rỗng / short)

Cho phép báo cáo F1 per-subreddit → phơi bày overfit subreddit.

---

## 5. Stage 1 — Text Model (MentalRoBERTa)

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
| `per_device_batch` | **16** | Mỗi T4 16GB fit comfortably |
| `gradient_accumulation` | **2** | Effective batch = 16 × 2 GPU × 2 accum = **64** |
| **n_gpus** | **2 (DataParallel)** | Kaggle 2× T4 16GB |
| `fp16` | True | T4 chỉ hỗ trợ fp16 (không bf16) |
| `gradient_checkpointing` | **False** | T4 16GB không cần — tốc độ ưu tiên hơn |
| `optimizer` | **AdamW thường** (PyTorch) | T4 16GB đủ; bỏ 8-bit cho ổn định |
| `lr` (LoRA adapters) | 5e-4 | LoRA cần LR cao hơn full FT |
| `lr` (head) | 1e-4 | Head random init, vừa phải |
| `weight_decay` | 0.01 | |
| `warmup_ratio` | 0.06 | |
| `lr_scheduler` | Linear decay sau warmup | |
| `num_epochs` | 3 | Đủ với 2M samples |
| `eval_strategy` | Every 2000 steps | |
| `early_stopping_patience` | 3 (eval points) | Dừng nếu val F1 không tăng |
| `metric_for_best_model` | `f1_macro` | Không phải accuracy |
| `dataloader_num_workers` | 0 | Windows-specific |
| `dataloader_pin_memory` | False | Tiết kiệm RAM hệ thống |

### 5.6. Loss function

**Nếu binary với class imbalance** (vd 30% positive):
- **Focal Loss** với `γ=2, α=0.7` (cho positive class), HOẶC
- **BCE với pos_weight** = `neg_count / pos_count`
- **Label smoothing 0.05** (chống noisy labels Reddit)

**Nếu multi-class (mức độ trầm cảm 0/1/2/3)**:
- Cross-entropy với class weights + label smoothing
- Cân nhắc ordinal regression (vì các mức độ có thứ tự)

> Quyết định cụ thể sau khi xem distribution label ở EDA.

### 5.7. VRAM estimate trên Kaggle T4 16GB (per GPU) [UPDATED]

| Khoản | Ước lượng |
|---|---|
| Backbone fp16 frozen | 220 MB |
| LoRA adapters + custom head fp16 + grad + Adam | ~120 MB |
| Activations (batch 16, seq 512, no checkpointing) | ~8 GB |
| Hidden states 4 layer (for layer-agg) | ~1.2 GB |
| Optimizer buffers + scaler | ~50 MB |
| CUDA context + framework overhead | ~1 GB |
| **Tổng / GPU** | **~10.6 GB / 16 GB** ✓ |
| **Effective batch (2 GPU × 16 batch × 2 accum)** | **64** |

Còn dư ~5 GB → có thể tăng batch lên 24 nếu cần. Hoặc bật gradient checkpointing và tăng batch lên 32 (effective 128) để train ổn định hơn.

**Local dev trên 3050 6GB** (debug only):
- `max_length=256`, batch=4, grad_accum=8, fp16, gradient_checkpointing, 8-bit Adam
- Chạy được trên subset 100K rows để verify pipeline trước khi push lên Kaggle

### 5.8. Output của Stage 1

1. Best model checkpoint: `models/stage1_mentalroberta_lora_fold{k}/`
2. **OOF predictions** trên train: `predictions/p_text_oof_train.parquet` (id, p_text)
3. Predictions val: `predictions/p_text_val.parquet`
4. Predictions test: `predictions/p_text_test.parquet`

---

## 6. Stage 2 — LightGBM Meta-model

### 6.1. Vì sao LightGBM

| Lý do | |
|---|---|
| Mixed features (numerical + categorical) | ✓ native |
| Handle missing values | ✓ native, không cần impute |
| Categorical native (subreddit) | ✓ không cần encode |
| Robust với feature scaling | ✓ tree-based |
| Fast train trên CPU (24GB RAM thoải mái cho 2M rows) | ✓ |
| Phổ biến trong ML project, dễ defend trong report | ✓ |

So với XGBoost: nhanh hơn 2-3x, RAM ít hơn, categorical native. So với CatBoost: dễ tune hơn.

### 6.2. Feature set cho LGBM [REVISED post-EDA]

EDA findings: MI(subreddit, label)=0.71 bits (≈ max entropy), MI(hour_utc, label)=0.0005. Quyết định:

```
KEY FEATURE (must):
  p_text                    ← OOF prediction từ Stage 1

Numerical (high-signal post-EDA):
  body_len_chars            ← Spearman với label = +0.45 (cao nhất)
  body_length_bucket        ← MI = 0.17 bits, label_rate đi từ 4% → 67% theo bucket
  has_body                  ← MI = 0.033 bits
  num_comments_log          ← Spearman = -0.20
  comments_per_upvote
  upvotes_log, upvotes_pct_in_subreddit
  title_len_chars, body_to_title_ratio
  num_first_person, num_negative_words, num_exclamations, num_questions
  year                      ← MI = 0.078 bits (drift signal)
  has_mh_keyword (boolean)  ← lift 4.3x; very strong feature

Temporal (low-signal but cheap, post-EDA):
  hour_sin, hour_cos        ← MI = 0.0005 bits, gần như vô dụng
  dow_sin, dow_cos          ← MI = 0.0011 bits, vô dụng
  → giữ cho ablation, nhưng expect = 0 lift

Categorical:
  subreddit                 ← MI = 0.71 bits — TRAIN 2 VERSIONS (xem 6.3)
  has_title

Total: ~22 features (bỏ month/is_weekend/is_night_us_eastern do MI ~0)
```

### 6.3. SUBREDDIT ABLATION = PRIMARY EVALUATION [BẮT BUỘC post-EDA]

Vì subreddit MI = 0.71 ≈ entropy max của label, mô hình full sẽ predict label ≈ subreddit lookup → F1 ≈ 1.0 trivially. Đây là **bài toán giả**.

**Phải train 2 versions** và báo cáo cả 2:

| Version | Features | Mục đích | Kỳ vọng |
|---|---|---|---|
| **A: Full** | tất cả 22 features (có subreddit) | Best-case performance | F1 ≈ 0.95-0.99 (gần trivial) |
| **B: Subreddit-blind** | tất cả TRỪ subreddit | "Real" generalization | F1 ≈ 0.80-0.88 |

→ **Version B là kết quả "thực" của mô hình**, version A là upper bound.

> Trong báo cáo, phải nói: "If subreddit is available at inference time, our model achieves F1=A. However, this largely reflects the subreddit-to-label mapping in the dataset. The model's true content-understanding ability is measured by Version B, which achieves F1=B without the subreddit feature."

**Bonus version C** (nếu thời gian cho phép): Cross-subreddit generalization
- Train: chỉ posts từ r/teenagers, r/depression, r/SuicideWatch
- Test: posts từ r/happy, r/DeepThoughts
- Đo F1 khi mô hình thấy subreddit chưa từng gặp → đo thực sự "depression understanding"

### 6.3. Model config baseline

```
objective: binary  (hoặc multiclass)
metric: binary_logloss + auc
boosting_type: gbdt
num_leaves: 63
max_depth: -1
learning_rate: 0.05
n_estimators: 2000 (với early stopping)
min_child_samples: 100  (chống overfit trên subreddit hiếm)
reg_alpha: 0.1
reg_lambda: 0.1
subsample: 0.8
colsample_bytree: 0.8
class_weight: 'balanced'  (nếu imbalanced)
early_stopping_rounds: 100
```

### 6.4. Hyperparameter tuning

**Optuna với 5-fold stratified CV trên Train**, optimize `f1_macro`.

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

## 7. Stacking & Out-of-Fold (OOF)

### 7.1. Vấn đề data leakage

**Nếu làm sai**: Train MentalRoBERTa trên 1.6M → predict trên chính 1.6M đó → `p_text` overfit → LGBM học "p_text ≈ label" → test set tệ.

### 7.2. Giải pháp: OOF predictions

```
Chia Train (1.6M) thành 5 folds stratified by label.

For fold k in [1..5]:
    train_subset = Train \ fold_k
    val_subset = fold_k

    Train MentalRoBERTa+LoRA trên train_subset
    Predict trên val_subset → p_text cho fold k

Concat 5 fold predictions → p_text_oof phủ toàn bộ 1.6M train
                            (mỗi row được predict bởi model KHÔNG thấy nó)
```

→ `p_text_oof` là unbiased estimator của `P(depression | text)` trên dữ liệu unseen.

### 7.3. Cho val và test

Train **một model riêng trên toàn bộ 1.6M train** → predict val và test một lần.

### 7.4. Trade-off của OOF

**Cost**: phải train MentalRoBERTa **6 lần** (5 fold + 1 final) = ~24-30h trên 3050.

**Alternative nếu time-constrained**:
- **3-fold thay vì 5-fold**: compromise, train 4 lần thay vì 6.
- **Holdout-based**: chia train thành 80/20, train trên 80%, predict 20% → dùng 20% đó cho train LGBM. Nhanh hơn 5x nhưng mất 80% data cho stacking.

**Khuyến nghị**: 5-fold nếu deadline cho phép. 3-fold nếu không.

---

## 8. End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA PREP (~1h)                                         │
│  ├─ 01_eda.ipynb                                                 │
│  ├─ 02_clean.py → posts_text.parquet, posts_meta.parquet         │
│  └─ 03_split.py → train/val/test ids, 5-fold ids                 │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 2: STAGE 1 TRAINING (~24-30h)                              │
│  ├─ 04_train_text_fold.py (chạy 5 lần, fold 1..5)                │
│  │   → models/stage1_fold{k}/, p_text_oof_fold{k}.parquet        │
│  ├─ 05_train_text_full.py (train trên toàn bộ train)             │
│  │   → models/stage1_full/                                       │
│  ├─ 06_predict_text.py                                           │
│  │   → p_text_oof_train.parquet (gộp 5 folds)                    │
│  │   → p_text_val.parquet, p_text_test.parquet (từ full model)   │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 3: METADATA FEATURE ENGINEERING (~10 min)                  │
│  └─ 07_build_meta_features.py                                    │
│      → meta_features_{train,val,test}.parquet                    │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 4: STAGE 2 TRAINING (~30 min)                              │
│  ├─ 08_tune_lgbm.py (Optuna 50 trials trên train với 5-fold CV)  │
│  │   → best_params.json                                          │
│  ├─ 09_train_lgbm_final.py (train trên full train với best HP)   │
│  │   → models/stage2_lgbm.pkl                                    │
│  └─ 10_predict_final.py                                          │
│      → p_final_{val,test}.parquet                                │
├─────────────────────────────────────────────────────────────────┤
│ PHASE 5: EVALUATION (~30 min)                                    │
│  ├─ 11_evaluate.py                                               │
│  │   → metrics_report.json, plots/                               │
│  └─ 12_error_analysis.ipynb                                      │
└─────────────────────────────────────────────────────────────────┘
```

Tổng thời gian: **~30-35 giờ chủ động trên máy** (phần lớn là Stage 1 chạy nền).

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

### 9.2. So sánh 6 model variants [REVISED post-EDA: ablation matrix]

EDA cho biết subreddit là confounder cực mạnh → ablation matrix 3×2:

| | With subreddit | Without subreddit |
|---|---|---|
| **Stage 1 only (text)** | n/a (text không dùng subreddit) | A: P_text > 0.5 |
| **Stage 2 only (no P_text)** | B1 | B2 |
| **Full ensemble (P_text + meta)** | C1 (upper bound, trivial) | **C2 (REAL performance)** |

Phải report cả 6 con số. C2 là số "thực" của mô hình, C1 là upper bound (gần như sub→label lookup).

→ Diễn giải:
- **C1 - A**: bao nhiêu performance đến từ subreddit shortcut → expect lớn
- **C2 - A**: bao nhiêu metadata thực sự thêm vào text signal → expect nhỏ-vừa
- **C1 - C2**: chính là leakage do subreddit feature
- **A - B2**: bao nhiêu text model thực sự hơn metadata-only baseline → measure of text understanding

### 9.3. Diagnostics

- **Calibration curve**: probability có well-calibrated không?
- **Confusion matrix** ở threshold 0.5 và threshold optimal (theo F1)
- **Feature importance** từ LGBM: P_text có rank #1 không?
- **SHAP** trên sample test set
- **Subgroup analysis**: F1 per subreddit
- **Error analysis**: random sample 50 false positives + 50 false negatives, đọc manually, tìm pattern

### 9.4. Ablation study (cho report) [REVISED post-EDA]

**Block 1 — Subreddit leakage** (số liệu chính, must-have):

| Variant | F1 macro | F1 per top-5 sub |
|---|---|---|
| C1: Full ensemble WITH subreddit | ? (expect ~0.95+) | ? |
| **C2: Full ensemble WITHOUT subreddit** | **? (REAL number)** | ? |
| B2: Metadata-only WITHOUT subreddit | ? | ? |
| A: Stage 1 text only | ? | ? |
| **Gap C1 - C2** | **= leakage size** | |

**Block 2 — Architecture choices**:

| Variant | F1 (without subreddit) |
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

    # 2. Text → P_text
    inputs = tokenizer(title, body, truncation="only_second",
                       max_length=256, return_tensors="pt")
    with torch.no_grad():
        p_text = sigmoid(text_model(**inputs).logits).item()

    # 3. Build metadata features
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

```
DeDe/
├── data/
│   ├── raw/                  # original dataset
│   ├── processed/            # cleaned parquets
│   └── splits/               # fold assignments
├── models/
│   ├── stage1_fold{1..5}/
│   ├── stage1_full/
│   └── stage2_lgbm.pkl
├── predictions/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 12_error_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── clean.py
│   │   ├── split.py
│   │   └── features.py
│   ├── models/
│   │   ├── text_model.py    # MentalRoBERTa + LoRA + custom head
│   │   └── meta_model.py     # LGBM wrapper
│   ├── training/
│   │   ├── train_text.py
│   │   └── train_lgbm.py
│   └── evaluation/
│       └── metrics.py
├── configs/
│   ├── text_model.yaml
│   └── lgbm.yaml
├── requirements.txt
├── README.md
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
| Subreddit feature dominate (vd `r/depression` → label=1 100%) → data leakage trivial | Cao | EDA kỹ; nếu vậy, cân nhắc loại subreddit khỏi feature hoặc đánh giá cross-subreddit |
| OOM trên 3050 dù đã optimize | Trung bình | Giảm max_length 256→192, batch 8→4, aggregate 2 layer thay 4 |
| OOF training quá lâu (>30h) | Cao | Dùng 3-fold thay 5-fold, hoặc subsample train xuống 1M |
| Label noise (Reddit labels thường tự khai báo) | Trung bình | Label smoothing 0.05; discuss trong report |
| Temporal drift (post 2015 vs 2023) | Thấp | Thêm feature `year`, hoặc time-based split để test |
| LGBM overfit categorical subreddit | Trung bình | `min_child_samples` cao, gộp rare subreddit |
| Calibration kém của text model | Trung bình | Platt scaling hoặc isotonic regression trên val |
| bitsandbytes lỗi trên Windows | Trung bình | Fallback sang `adamw_torch` (+0.5 GB RAM) |

---

## 13. Timeline đề xuất

| Tuần | Việc |
|---|---|
| **W1** | EDA, cleaning, splitting, build meta features. Setup repo. |
| **W2** | Stage 1: implement custom head, train 1 fold sanity check, debug. |
| **W3** | Stage 1: train full 5-fold OOF + final model (chạy nền cả tuần). |
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
| Stacking | **5-fold OOF** stratified by label | Tránh leakage |
| Meta model | **LightGBM** | Mixed features, fast, native categorical |
| HP tuning | Optuna 50 trials với 5-fold CV | Standard |
| UTC features | Giữ nhưng expect minimal lift (MI ~0) | Đã đo |
| **Subreddit feature** | **Train 2 versions** (with/without) | EDA: MI=0.71 → leakage |
| **Splitting** | **Time-based** (≤2020 / 2021 / 2022) | EDA: drift 0.73 between years |
| Body length features | High priority (Spearman 0.45, MI 0.17) | Đo từ EDA |
| MH keyword feature | Boolean has_mh_keyword (lift 4.3x) | Free signal |
| Rare subreddit bucketing | Bỏ (chỉ 19 sub rare, 63 rows) | EDA: không có rare-tail thực |
| Evaluation | F1 macro + PR-AUC + per-subreddit F1 | Phơi bày overfit subreddit |
| Test set | Đụng **1 lần ở cuối** | Tránh test leakage |
| **Primary metric** | **F1 (Version C2, WITHOUT subreddit)** | Số "thực", không trivial |

---

## 14a. Kaggle Workflow [NEW]

Vì training chạy trên Kaggle 2× T4 16GB, có 4 ràng buộc cụ thể cần handle:

### Ràng buộc Kaggle

| Ràng buộc | Workaround |
|---|---|
| **30h GPU/tuần** free tier | Mỗi fold ~1.5-2h × 5 fold = ~10h cho Stage 1 full. Đủ cho 2-3 lần iterate. |
| **12h timeout/notebook run** | Chạy 2-3 fold per notebook. Hoặc 1 fold + extra experiments. |
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

KAGGLE NOTEBOOK (per fold):
  6. Settings: Accelerator = GPU T4 x2, Internet = Off (sau khi pre-cache)
  7. Notebook structure:
     - Cell 1: Load dataset từ /kaggle/input/dede-preprocessed/
     - Cell 2: Load tokenizer + model từ /kaggle/input/mentalroberta-weights/
     - Cell 3: Define model class (custom head)
     - Cell 4: Training loop với DataParallel
     - Cell 5: Generate OOF predictions cho fold k
     - Cell 6: Save p_text_fold{k}.parquet → /kaggle/working/ → download

LOCAL (3050):
  8. Gộp 5 fold OOF predictions
  9. Train Stage 2 (LightGBM) — CPU-bound, nhanh
  10. Final evaluation
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

### Time budget cho Stage 1 full training

| Item | Time / fold | Total (5 fold + 1 full) |
|---|---|---|
| Train (1.45M rows train per fold, 3 epochs) | ~1.5h | ~9h |
| Val prediction (363K rows) | ~5 min | ~30 min |
| Final model on full train (1.81M, 3 epochs) | ~2h | ~2h |
| **Tổng** | | **~12h** (fit dưới quota 30h/tuần) |

→ 1 tuần Kaggle có thể chạy full Stage 1 + 1-2 experiments khác (ablation head components, etc.).

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
   - Run 5-fold OOF + final full-train model (~10-12h trên Kaggle)
5. **Phase 4 — Stage 2 (Local CPU)**: train **2 versions** (with/without subreddit), Optuna tuning
6. **Phase 5 — Evaluation**: ablation matrix 6 variants + per-subreddit + temporal split comparison
7. **Phase 6 — Report**: emphasize subreddit-derived label nature, C1 vs C2 gap, time-based realistic estimate
