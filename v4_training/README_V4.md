# Listing Lens V4 — AI Generator Adaptation Starter

This is a separate training workspace for preparing and evaluating a **V4 candidate**. It does not replace the live Listing Lens application or overwrite V3 weights.

## What is in this package

- `ai_candidates/GPT/`: 26 newly supplied AI-image candidates
- `ai_candidates/GEMINI/`: 19 newly supplied AI-image candidates
- `ai_candidates/META/`: 9 newly supplied AI-image candidates
- `ai_candidates/v4_candidate_manifest.csv`: SHA-256, image metadata, source folder label, group and split assignment
- `ai_candidates/near_duplicate_review.csv`: five perceptual-similarity pairs from the intake audit; none were automatically deleted
- `ai_candidates/v4_split_plan.json`: split allocation summary
- `prepare_v4_dataset.py`: copies the user's existing V2 split and appends the new AI images to train/validation/test AI folders only
- `train_v4.py`: fine-tunes the existing V3 `.keras` checkpoint into a separate V4 candidate
- `notebooks/08_v4_ai_generator_adaptation.ipynb`: guided workflow

## Important intake findings

The package contains **54 images**, not an additional batch on top of the previously shared 54: GPT 26, Gemini 19, Meta AI 9. All 54 images passed Pillow readability checks. The source package reports zero exact SHA-256 duplicate groups and five perceptually similar pairs. The GPT/GEMINI/META labels are inherited from supplied folder names and have not been independently authenticated.

The split plan keeps perceptually similar candidate groups together and allocates:

| Generator | Train | Validation | New-generator test | Total |
|---|---:|---:|---:|---:|
| GPT | 18 | 4 | 4 | 26 |
| Gemini | 13 | 3 | 3 | 19 |
| Meta AI | 6 | 1 | 2 | 9 |
| **Total** | **37** | **8** | **9** | **54** |

The 17 held-out images (8 validation + 9 test) are a small exploratory sample. They are useful for a first diagnostic but **not enough** to make reliable per-generator generalisation claims. Full prompt/session metadata was not supplied; the split keeps the listed perceptual-similarity components together but cannot guarantee that unseen prompt families are held out.

## Step 0 — Keep the original safe

1. Keep your current GitHub project folder and V3 model unchanged.
2. Extract this starter ZIP separately, or put this folder inside your Listing Lens project as `v4_training/`.
3. Do not commit the raw `ai_candidates/GPT/`, `GEMINI/`, or `META/` image folders, `data_v4/`, or `outputs_v4/` to GitHub. They are local experiment data and checkpoints. The starter `.gitignore` excludes the image folders when this starter is placed at `<project>/v4_training/`.
4. Add these lines to the **Listing Lens root** `.gitignore` if they are not already present:

```gitignore
# Listing Lens V4 local experiment data and checkpoints
/data_v4/
/outputs_v4/
```

5. Keep the original Dataset V2 directory unchanged. The preparation script creates a copy in a new output directory.

## Step 1 — Install the small additional dependency

Activate the same Python 3.11 ARM64 environment that runs Listing Lens, from the root of your project:

```bash
source .venv-arm64/bin/activate
python -m pip install -r v4_training/requirements-v4.txt
```

If this starter is extracted separately, use its requirements file:

```bash
python -m pip install -r /path/to/Listing_Lens_V4_Training_Starter/requirements-v4.txt
```

TensorFlow should already be installed through the main project `requirements.txt` (the recorded V3 runtime was TensorFlow 2.21.0 on Python 3.11 ARM64). Verify:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Step 2 — Prepare Dataset V4

Your existing Dataset V2 root must have this structure:

```text
DATASET_V2_ROOT/
├── train/real/
├── train/ai_generated/
├── validation/real/
├── validation/ai_generated/
├── test/real/
└── test/ai_generated/
```

From your Listing Lens project root, run (replace the V2 path with the actual folder on your Mac):

```bash
python v4_training/prepare_v4_dataset.py \
  --v2-root "/absolute/path/to/car_dataset_split_v1" \
  --output-root "./data_v4"
```

If you extracted the starter separately, point to its script explicitly:

```bash
python /absolute/path/to/Listing_Lens_V4_Training_Starter/prepare_v4_dataset.py \
  --v2-root "/absolute/path/to/car_dataset_split_v1" \
  --output-root "./data_v4"
```

The script:

- Copies all original V2 split folders into `data_v4/` (does not modify the source dataset).
- Keeps all original real photographs in their existing split and class.
- Adds only the 54 new candidates to `ai_generated`, using the split recorded in the manifest.
- Skips new AI images that are exact byte-level duplicates of existing V2 AI images.
- Stops for manual review if a new AI image exactly matches a V2 real-labelled image.
- Writes cross-dataset perceptual near-match reports where ImageHash is available; near matches are not automatically removed.
- Converts WebP/TIFF and other unsupported formats to PNG in the **V4 copy only**, to avoid Keras directory-loader omissions.
- Writes `v4_preparation_summary.json` and provenance CSV files.

Review these files before training:

- `data_v4/v4_preparation_summary.json`
- `data_v4/v4_exact_duplicates_skipped.csv`
- `data_v4/v4_cross_dataset_near_duplicate_review.csv`

## Step 3 — Fine-tune the existing V3 model

From the Listing Lens project root:

```bash
python v4_training/train_v4.py \
  --dataset-root "./data_v4" \
  --v3-model "./models_v3/car_cnn_model_v3.keras" \
  --v3-config "./config/cnn_inference_config.json" \
  --output-dir "./outputs_v4" \
  --batch-size 8
```

If the starter is separate, substitute its full path for `v4_training/train_v4.py`.

The script checks the V3 model SHA-256 against the inference config before training. It loads the saved model and preserves its embedded architecture and augmentation. The input remains 256×256 RGB, float32 in the raw 0–255 range; **do not divide by 255 externally**, because EfficientNetB0 normalization is embedded in the model.

### Proposed V4 fine-tuning schedule

These are conservative V4 adaptation settings, not a claim about the unrecovered original V3 training script:

| Phase | Backbone | Learning rate | Maximum epochs |
|---|---|---:|---:|
| 1 | Frozen; adapt existing classification head | 0.0001 | 5 |
| 2 | Top 40 backbone layers trainable; Batch Normalization frozen | 0.00001 | 12 |

The script uses Adam, binary cross-entropy with 0.05 label smoothing, balanced per-class sample weights, early stopping and learning-rate reduction. Batch size defaults to 8 for the MacBook Air 8 GB RAM environment. Training may be slow; use Colab GPU if needed, but use the same saved V3 model and exact prepared dataset.

## Step 4 — Review the results

The output folder `outputs_v4/` contains:

- `phase1_history.csv`
- `phase2_history.csv`
- `phase1_best.keras`
- `car_cnn_model_v4_candidate.keras`
- `v4_training_report.json`

The validation set is used to choose a candidate threshold by balanced accuracy. The test set is evaluated only after threshold selection. New-generator test images are reported as an AI detection rate by generator; because there are only 4 GPT, 3 Gemini and 2 Meta test images, these percentages are extremely uncertain.

Compare V3 and V4 on the **same original untouched V2 test images** before making any application change. Also inspect false positives on genuine photographs. Do not replace the V3 model merely because the V4 training loss decreases.

## Important limitations

- The original Dataset V2 image folders are not bundled. Supply their existing local split root to the preparation script.
- This intake ZIP repeats the prior 54-image candidate set; it does not contain more than 54 images.
- Cross-dataset exact duplicates are checked locally against V2. Perceptual matches are flags for human review, not automatic deletion.
- The new holdout is too small for a statistically strong GPT/Gemini/Meta comparison. Collect more independently prompted samples per generator before claiming broad generalisation.
- The user's folder names supply generator attribution; no independent authenticity verification of provider origin is performed.
- The active app continues using V3 throughout. `train_v4.py` never updates the app config and never overwrites V3.
