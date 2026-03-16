# CAPTCHA Recognition

A CNN-based CAPTCHA solver trained on a custom-generated dataset of 100,000 images, achieving **99.6–99.8% per-character accuracy** and **99.18% sequence accuracy** on a held-out test set.

---

## Results

| Position | Accuracy |
|----------|----------|
| Char 1   | 99.8%    |
| Char 2   | 99.8%    |
| Char 3   | 99.7%    |
| Char 4   | 99.6%    |
| Char 5   | 99.7%    |
| **Sequence (all 5 correct)** | **99.18%** |

---

## Architecture

The model is a multi-output CNN with a **spatial slice** design — each character head only sees the horizontal region of the feature map that corresponds to its position in the image.

```
Input (60 × 200 × 1)
   │
Conv2D(32) → MaxPool
Conv2D(64) → MaxPool
Conv2D(128) → MaxPool
   │
Feature map: ~(7 × 25 × 128)
   │
   ├─ Slice[:,  0: 5,:] → GAP → Dense(64) → Softmax(32) → char_0
   ├─ Slice[:,  5:10,:] → GAP → Dense(64) → Softmax(32) → char_1
   ├─ Slice[:, 10:15,:] → GAP → Dense(64) → Softmax(32) → char_2
   ├─ Slice[:, 15:20,:] → GAP → Dense(64) → Softmax(32) → char_3
   └─ Slice[:, 20:25,:] → GAP → Dense(64) → Softmax(32) → char_4
```

**Total parameters: 144,352 (~564 KB)**

The key insight: standard `GlobalAveragePooling2D` over the full feature map discards all spatial information — every character head receives the same averaged representation with no way to distinguish position. By slicing the feature map horizontally first, each head looks at a different region of the image, which was the fix that pushed accuracy from ~35% to 99%+.

---

## Dataset

Images were generated with a custom Pillow-based generator (`generate_captcha.py`) rather than a public dataset.

- **100,000 images**, 200×60 px, grayscale
- **5 characters** per CAPTCHA, fixed length
- **32-character vocabulary**: digits + lowercase, with ambiguous characters removed (`0`, `o`, `1`, `l`)
- **Train / Val / Test split**: 80,000 / 10,000 / 10,000

Generating from scratch gave full control over image properties and eliminated label noise from existing datasets.

---

## How It Was Built — The Debugging Story

This project went through two major failures before the final architecture.

### Attempt 1: Kaggle dataset + CTC loss

The first approach used a public CAPTCHA dataset (~1,989 images) with a CRNN (CNN + Bidirectional LSTM) and CTC loss. After working through Keras 3 compatibility issues with `ctc_batch_cost` and blank token indexing, the model converged to **4% character accuracy**. The dataset was too small and CTC added more complexity than the problem needed.

### Pivot: Custom dataset + fixed-length classification

Scrapped the Kaggle data entirely. Generated a controlled 10k-image dataset with Pillow, fixed sequence length at 5, dropped CTC, and used a multi-output CNN with one softmax head per character. Early results:

```
Char 1: 99.5%
Char 2: 34.1%
Char 3: 35.6%
Char 4: 36.0%
Char 5: 97.3%
```

Edge characters near perfect, middle characters stuck at ~35%.

### Problem 1: Black triangle artifacts in the generator

Inspecting the generated images revealed black triangles in the corners of each character slot — caused by the rotation function filling empty pixels with black (`0`) instead of the background colour. Middle characters had artifacts on both sides, confusing the model.

**Fix:**
```python
char_img = char_img.rotate(angle, resample=Image.BILINEAR,
                           expand=False, fillcolor=bg_color)
```

Regenerated 100k clean images, retrained — middle characters still stuck at 35%.

### Problem 2: GlobalAveragePooling discards position

The data was clean, so the problem had to be the architecture. `GlobalAveragePooling2D` over the full feature map collapses everything into a single vector — every character head was seeing the same averaged representation. The 35% matched chance-level on a 32-class problem where the model had learned nothing position-specific.

**Fix: spatial slice architecture**

```python
feature_w = IMG_WIDTH // 8
slice_w   = feature_w // CAPTCHA_LEN

for i in range(CAPTCHA_LEN):
    start = i * slice_w
    end   = start + slice_w if i < CAPTCHA_LEN - 1 else feature_w
    char_feat = x[:, :, start:end, :]
    char_feat = layers.GlobalAveragePooling2D()(char_feat)
    char_feat = layers.Dense(64, activation='relu')(char_feat)
    out = layers.Dense(NUM_CLASSES, activation='softmax', name=f'char_{i}')(char_feat)
```

Accuracy jumped to 99%+ immediately.

---

## Project Structure

```
captcha-recognition/
├── data/
│   ├── raw/                  # generated images
│   ├── labels.csv            # filename → label mapping
├── models/
│   └── captcha_model.keras
├── generate_captcha.py       # custom Pillow-based CAPTCHA generator
├── Captcha_solver.ipynb
└── requirements.txt
```

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Generate the dataset**
```bash
python generate_captcha.py
```
Creates 100,000 images under `data/raw/` and writes `data/labels.csv`.

**3. Train**

Open `Captcha_solver.ipynb` and run all cells in order. Training takes roughly 15–30 minutes depending on hardware. The best model is saved automatically to `models/captcha_model.keras` via `ModelCheckpoint`.

**4. Evaluate**

The final cells load the saved model and print per-position accuracy, sequence accuracy, a prediction grid, and a visualisation of the spatial regions each character head attends to.

---

## Tech Stack

- Python 3.12
- TensorFlow / Keras
- Pillow
- NumPy, Pandas, scikit-learn, Matplotlib
