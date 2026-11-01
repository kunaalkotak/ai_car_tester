# V4 training utilities

This folder contains the V4 dataset preparation and fine-tuning scripts. They are included for academic transparency; the Streamlit application does not import them.

## Inputs required to reproduce training

- The Dataset V2 train/validation/test image folders (1,250 original candidates).
- The 54 new AI candidate images and their `v4_candidate_manifest.csv` used for the recorded V4 run.
- The historical V3 EfficientNetB0 checkpoint used to initialise fine-tuning. Its recorded SHA-256 is `ea47dc1e9ded13944a2ac866ae968f3956a003d9faaf39ce4a0dfa8a8cad65b5`. That V3 weight file is intentionally not included in this final application repository because the deployed app needs only V4.
- A Python 3.11 ARM64 environment with the application requirements and `requirements-training.txt` installed.

The raw datasets, candidate images and V3 starting checkpoint are not distributed here. Therefore this repository preserves the training code and recorded outcomes, but does not by itself reproduce the full training run from raw inputs.

## Prepare Dataset V4

```bash
python training/prepare_v4_dataset.py \
  --v2-root /path/to/Dataset_V2_Splits_keras_compatible_v3 \
  --candidate-root /path/to/AI_candidates_with_manifest \
  --output-root ./data_v4
```

The script copies Dataset V2 to a new output folder, adds only the new AI images, checks exact duplicates, flags perceptual near-duplicates for manual review and converts unsupported formats in the copied V4 dataset. It never edits the original dataset.

## Fine-tune V4

After inspecting `data_v4/v4_preparation_summary.json`:

```bash
python training/train_v4.py \
  --dataset-root ./data_v4 \
  --v3-model /path/to/car_cnn_model_v3.keras \
  --v3-sha256 ea47dc1e9ded13944a2ac866ae968f3956a003d9faaf39ce4a0dfa8a8cad65b5 \
  --v3-threshold 0.47 \
  --output-dir ./outputs_v4 \
  --batch-size 8
```

The script saves a candidate and report in the output directory; it does not overwrite `models_v4/` or the app config. The active V4 artifact and exact recorded results are included under `models_v4/` and `documentation/`.
