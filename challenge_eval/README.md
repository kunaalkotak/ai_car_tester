# Listing Lens — External Challenge Evaluation (V3 vs V4)

This kit evaluates the preserved V3 model and the V4 candidate on a new, frozen collection of images that were **not used for training or validation**. It does not train a model, edit either checkpoint, alter the Streamlit app, or tune thresholds from the challenge set.

## 1. Why this test exists

V4 was trained on 1,304 images and compared with V3 on the same V4 test split. Its validation-selected threshold was approximately 0.50; the V3 validation-selected threshold on that development split was approximately 0.40. The earlier deployed V3 threshold was 0.47. The reported V4 test accuracy was 93.91%, AI recall 95.15%, AI F1 94.23% and ROC-AUC 98.89%; those values are development-test results, not a substitute for an independent challenge set.

The purpose of this next evaluation is to see whether the V4 improvement holds on genuinely new GPT, Gemini and Meta AI outputs and on independently sourced genuine car photographs. These are **same-generator-family, new-image** tests, not unseen-generator-family tests, because these generator families were represented in V4 development data.

## 2. Collect a separate challenge set

Recommended starting target:

- 50 new GPT-generated car images.
- 50 new Gemini-generated car images.
- 50 new Meta AI-generated car images.
- 100 genuine car photographs from known camera/original-photo sources.

Aim for diversity: car type, colour, angle, distance, road/parking/indoor setting, lighting, time of day, weather, image composition and phone/camera quality. Keep original generated files; do not screenshot, edit, add watermarks, or re-save at a different quality. Use different prompts and sessions from those used to create training images. Preserve prompts and source/model version in `prompt_log.csv`.

For genuine photographs, use images with known camera/original-photo provenance and permission to use them. Do not label a photo "real" merely because it looks realistic. Avoid collecting personal data, number plates or identifiable people where possible; blur/redact only if needed and document any change, because processing can affect image cues.

Suggested directory layout:

```text
challenge_eval/data/
├── real/
│   ├── real_001.jpg
│   └── ...
└── ai_generated/
    ├── GPT/
    ├── Gemini/
    └── Meta_AI/
```

Put your files in the matching folders. The folder label determines the class: `real` = 0; `ai_generated` = 1. The AI subfolder name determines the generator group. Use `manifest_template.csv` as a starting point for `manifest.csv`; all paths must be relative to `data/`, e.g. `ai_generated/GPT/gpt_001.png`. `prompt_log_template.csv` records the full prompts and tool/model version.

Do not include any of these challenge images in training, fine-tuning, augmentation, threshold selection or model selection. Freeze this set before running evaluation. If you later train on it, it is no longer an independent test set.

## 3. Check project files

From the Listing Lens project root, make sure these paths exist:

```text
models_v3/car_cnn_model_v3.keras
config/cnn_inference_config.json
outputs_v4_full/car_cnn_model_v4_candidate.keras
outputs_v4_full/v4_training_report.json
data_v4_full/
challenge_eval/
```

The script verifies both model SHA-256 values against their config/report before loading. It scans `data_v4_full` for exact-hash overlap and excludes challenge files that are exact duplicates of existing V4 development images. It also removes repeated exact copies within the challenge set. Check `results/excluded_files.csv` after running.

This hash check does not detect visually similar but differently encoded images. Review the contact sheet and provenance manually for near-duplicates and repeated generations.

## 4. Install evaluation dependency

Activate the same Python 3.11 ARM64 environment used by Listing Lens:

```bash
source .venv-arm64/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r challenge_eval/requirements-eval.txt
```

## 5. Run the evaluation

Run from the project root:

```bash
python challenge_eval/evaluate_challenge.py \
  --challenge-root challenge_eval/data \
  --known-data-root data_v4_full \
  --v3-model models_v3/car_cnn_model_v3.keras \
  --v3-config config/cnn_inference_config.json \
  --v4-model outputs_v4_full/car_cnn_model_v4_candidate.keras \
  --v4-report outputs_v4_full/v4_training_report.json \
  --output-dir challenge_eval/results \
  --batch-size 8
```

Thresholds are read from validation-only results in the V4 training report: V3 = 0.40 and V4 = 0.50. The script also reports V3 at its original application threshold of 0.47 as a secondary reference. Do not change thresholds based on this external test output.

## 6. Outputs

- `results/evaluation_summary.json`: overall V3/V4 metrics, confusion matrices, false-positive rate on real photos, model hashes, per-generator AI detection rate and Wilson 95% intervals.
- `results/predictions.csv`: per-image score and prediction from each model, correctness, generator and provenance metadata.
- `results/excluded_files.csv`: unreadable/unlabelled files and exact duplicates excluded from evaluation.

## 7. Interpretation and limitations

- Aim for at least 50 AI images per generator and at least 100 genuine photographs. Smaller counts are exploratory; the script warns when counts are below these targets.
- Wilson intervals are included to show the uncertainty in each generator's detection rate.
- This set tests new images from GPT/Gemini/Meta families, not completely unseen generator families.
- Source folders and user-reported tool labels are provenance metadata, not independent authentication of the generator.
- High AI recall is not enough: inspect false positives among genuine photos because a used-car buyer could be misled by an incorrect warning.
- The model score is an uncalibrated sigmoid score, not a guaranteed probability.
- Do not replace V3 in the app solely because V4 wins this one challenge. Review metrics, individual errors, label provenance and sample size first.
- This tool does not determine if the car exists, seller is genuine, listing is fraudulent, or photo belongs to the advertised vehicle.

## 8. Privacy and GitHub

Challenge photos and evaluation results are intentionally ignored by this folder's `.gitignore`. Keep them local unless you have rights and permission to publish them. Commit the evaluation script and templates, not private or unlicensed images.
