# Listing Lens — Used-Car Photo Authenticity Checker

**Listing Lens** is a Streamlit application for second-hand car buyers. A user uploads a photograph from an online car listing and the application uses one image-classification model, displayed as **CNN**, to assess whether the image resembles a genuine photograph or AI-generated imagery.

> **Responsible-use notice:** Listing Lens is an experimental visual screening tool. It does not prove an image's origin, vehicle existence, seller identity, listing legitimacy or fraud. A genuine photograph can appear in a fraudulent listing, and an AI-looking image does not prove that a car or seller is fake. Buyers should verify the seller, documents and vehicle independently.

## 1. Problem and intended use

Used-car buyers often browse listings on online marketplaces and classifieds. An advertisement may use genuine photographs, heavily edited images, reused images or AI-generated imagery. Listing Lens offers an additional visual signal before a buyer decides what to ask the seller or whether to arrange an inspection.

Typical workflow:

1. Save a photograph from a used-car listing.
2. Upload it to Listing Lens.
3. The app validates and previews the image, preprocesses it and runs the CNN.
4. The result card displays a photograph-looking or AI-generated-looking assessment, an uncalibrated score and buyer-oriented cautions.

The app analyses only the uploaded image. It does not access listing URLs, seller profiles, vehicle registration databases, VIN records, mileage, service history or image provenance.

## 2. How the application works

```text
Upload car photo
      ↓
Image validation and preview
      ↓
RGB decoding and 256 × 256 resize
      ↓
CNN (EfficientNetB0 backbone)
      ↓
Sigmoid model score
      ↓
Apply 0.50 decision threshold
      ↓
Result card, score and buyer guidance
```

- Score **below 0.50** → `Real (0)` / photograph-looking.
- Score **at or above 0.50** → `AI-generated (1)` / AI-generated-looking.
- The displayed sigmoid score is **not calibrated probability** and is not a fraud score.

The model accepts RGB images at 256 × 256. Input values are float32 in the original 0–255 range; EfficientNetB0's normalization is embedded in the saved model, so the application must not divide by 255 again. Supported uploads include JPG/JPEG, PNG, WebP, BMP and GIF (first frame).

## 3. Current application model — CNN V4

The final application ships **one model only**:

| Item | Active setting |
|---|---|
| User-facing name | CNN |
| Technical backbone | ImageNet-pretrained EfficientNetB0, fine-tuned for binary classification |
| Model file | `models_v4/car_cnn_model_v4.keras` |
| Input | 256 × 256 × 3 RGB |
| Output | One sigmoid score; 0 = real, 1 = AI-generated |
| Decision threshold | **0.50** |
| Model file size | 34,358,590 bytes |
| SHA-256 | `36b6e3fe95278cb73923ec01e109671278fe97877e5bb7beae4b947fcd3f6d3e` |

The application has no model selector and no classical-model inference path. The model loader checks the model SHA-256 before loading, then validates the input and output tensor shapes.

### V4 training approach

V4 was fine-tuned from the earlier EfficientNetB0 V3 checkpoint, not trained from random weights. The V3 checkpoint was the starting point for adaptation, but its weight file is **not included** in this final repository because it is not needed for V4 inference. Historical V3 metrics and its source SHA-256 remain documented. To reproduce the fine-tuning run, the matching V3 checkpoint must be supplied separately; see [`training/README.md`](training/README.md).

Recorded V4 settings:

| Setting | Phase 1 | Phase 2 |
|---|---|---|
| Backbone | Frozen | Top 40 layers trainable; Batch Normalization frozen |
| Learning rate | 0.0001 | 0.00001 |
| Maximum epochs | 5 | 12 |
| Epochs recorded | 5 | 8 |

Other settings: Adam optimizer, binary cross-entropy with label smoothing 0.05, batch size 8, class weighting, seed 42, early stopping and learning-rate reduction. The V4 model includes the augmentation/normalization layers used by the saved training architecture.

## 4. Dataset development history

Dataset V2 was assembled from several sources rather than downloaded as one Kaggle dataset. The source archives recorded for assembly were:

1. `real_car_dataset_v2_deduplicated.zip`
2. `Car_AI_Images_FGTS_Additions.zip`
3. `Car_Image_Authenticity_Dataset_v1.zip`

The real-image collection included the existing project collection and the Hugging Face dataset [`Smily6820/car-images`](https://huggingface.co/datasets/Smily6820/car-images). The AI-labelled candidates comprised 474 FGTS additions and 151 images carried over from the original V1 collection.

The recovered FGTS extraction summary records 158 LDM, 158 Stable Diffusion 1.4 and 158 ProGAN candidates at that extraction stage. The separate 151-image V1 contribution does not have a recovered per-generator breakdown. A 100-image AI-generation pilot notebook is preserved in the source archive, but its planned images are not counted as Dataset V2 candidates without per-image provenance.

### Dataset V2 quality audit

The first-pass audit processed 1,250 candidates and identified 183 images for manual review. Flagged images were not automatically deleted. The recorded methods included:

- SHA-256 and decoded-pixel hash checks;
- perceptual-hash similarity checks;
- image readability and resolution checks;
- aspect-ratio, brightness and blur checks;
- car-object-detection screening as a quality aid; and
- manual review/contact sheets.

A quality flag was a heuristic review signal, not proof that an image was unusable. Real-image labels were not independently authenticated one-by-one; AI labels and generator identities were inherited from source manifests/folder names.

### Dataset V2 split

Target ratio was approximately 70:15:15, with random seed 211:

| Split | Real | AI-generated | Total |
|---|---:|---:|---:|
| Training | 437 | 437 | 874 |
| Validation | 94 | 94 | 188 |
| Testing | 94 | 94 | 188 |
| **Total** | **625** | **625** | **1,250** |

The split workflow included hash/group checks. However, source groups overlap across splits in the recorded evaluation, and these results are not a held-out-generator test.

### V4 dataset expansion

Only the AI class was expanded. The 54 new AI candidates were labelled by the supplied source folders:

| Generator label | Added |
|---|---:|
| GPT | 26 |
| Gemini | 19 |
| Meta AI | 9 |
| **Total** | **54** |

The preparation report recorded zero exact duplicates skipped and zero cross-dataset perceptual near-duplicate review pairs. This is the output of the recorded checks, not a guarantee that every semantic near-duplicate or label error was found.

| V4 split | Real | AI-generated | Total |
|---|---:|---:|---:|
| Training | 437 | 474 | 911 |
| Validation | 94 | 102 | 196 |
| Testing | 94 | 103 | 197 |
| **Total** | **625** | **679** | **1,304** |

All 1,304 files were loaded by the V4 training pipeline. The original real-image counts were unchanged. The original Dataset V2 folders were copied; the new AI images were the only additions.

### Historical V3 WebP loader limitation

The earlier V3 Keras directory-loader run skipped WebP files: training loaded 872/874, validation 185/188 and test 186/188. V3 metrics therefore use these smaller effective counts. The V4 preparation converted unsupported image formats in its copied dataset where needed, and the V4 report confirms that all 1,304 files were loaded.

## 5. Model development and results

The project began with five approaches: Logistic Regression, SVM (RBF), Random Forest, Linear Discriminant Analysis (LDA) and a custom CNN. The classical models used flattened 32 × 32 RGB inputs with PCA and StandardScaler. The original custom CNN used 256 × 256 RGB input, three convolution stages (16, 32 and 64 filters), max pooling, Global Average Pooling, dropout, a dense layer and sigmoid output.

### Historical V1 baseline

The original V1 test set contained only 46 images. These are historical baseline results, not directly comparable with later Dataset V2/V4 experiments.

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| Custom CNN | 88.89% | 78.26% |
| Random Forest | 88.89% | 71.74% |
| SVM (RBF) | 84.44% | 76.09% |
| Logistic Regression | 82.22% | 71.74% |
| LDA | 82.22% | 69.57% |

### Dataset V2 model comparison

The custom CNN achieved 82.45% test accuracy, 77.98% AI precision, 90.43% AI recall, 83.74% AI F1 and 89.43% ROC-AUC at threshold 0.32. Classical model results:

| Model | Test accuracy | AI F1 |
|---|---:|---:|
| SVM (RBF) | 75.53% | 74.16% |
| Random Forest | 73.40% | 72.83% |
| Logistic Regression | 65.96% | 64.44% |
| LDA | 64.36% | 62.98% |

### Historical V3 EfficientNetB0 results

V3 used ImageNet-pretrained EfficientNetB0 with a two-phase training process. Its reported validation accuracy was 97.84% and test accuracy was 94.09%; test balanced accuracy 94.10%, AI precision 95.60%, AI recall 92.55%, AI F1 94.05% and ROC-AUC 99.53%. V3 loaded 872 training, 185 validation and 186 test images because of the WebP loader limitation. These are historical V3 results, not the current app model's results.

### V4 final evaluation

V4 results on the full 197-image test split, at the validation-selected threshold of 0.50:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 95.41% | **93.91%** |
| Balanced accuracy | 95.46% | **93.85%** |
| AI precision | 96.97% | **93.33%** |
| AI recall | 94.12% | **95.15%** |
| AI F1 | 95.52% | **94.23%** |
| ROC-AUC | 99.36% | **98.89%** |

Test confusion matrix (rows actual [real, AI], columns predicted [real, AI]): `[[87, 7], [5, 98]]`.

### V3 and V4 on the same V4 test split

The historical V3 baseline and V4 were evaluated on the same 197 test images. Each model used a threshold selected on the same validation split (V3 0.40; V4 0.50). ROC-AUC does not depend on the threshold.

| Metric | V3 historical baseline | V4 active |
|---|---:|---:|
| Accuracy | 93.40% | **93.91%** |
| Balanced accuracy | 93.50% | **93.85%** |
| AI precision | **95.92%** | 93.33% |
| AI recall | 91.26% | **95.15%** |
| AI F1 | 93.53% | **94.23%** |
| ROC-AUC | 97.98% | **98.89%** |

This is a modest trade-off: V4 has higher recall, F1 and AUC, while V3 had higher AI precision and fewer false positives on real images (4 vs 7). V4 was selected as the final application model, but these results do not prove performance on every future generator or marketplace image distribution.

The nine new-generator AI-only test examples were GPT 4/4, Gemini 3/3 and Meta AI 2/2. This is a very small exploratory subset. All three generator families were represented in the V4 development/training pool, so this is **not** a held-out-generator evaluation. Source labels came from user-supplied folder names and were not independently authenticated.

## 6. Repository structure

```text
Listing_Lens/
├── app_cnn.py
├── car_app_ui.py
├── requirements.txt
├── requirements-notebooks.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── config/
│   └── cnn_inference_config.json
├── models_v4/
│   └── car_cnn_model_v4.keras
├── src/
│   └── cnn_inference.py
├── scripts/
│   └── verify_active_cnn.py
├── training/
│   ├── README.md
│   ├── prepare_v4_dataset.py
│   ├── train_v4.py
│   └── requirements-training.txt
├── notebooks/
│   ├── 01_dataset_collection_assembly.ipynb
│   ├── 02_dataset_quality_audit.ipynb
│   ├── 03_train_validation_test_split.ipynb
│   ├── 04_original_five_model_experiments.ipynb
│   ├── 05_dataset_v2_model_comparison.ipynb
│   ├── 06_efficientnetb0_v3_training.ipynb
│   ├── 07_v3_model_evaluation.ipynb
│   ├── 08_v4_ai_generator_adaptation.ipynb
│   ├── 09_v4_final_model_evaluation.ipynb
│   └── source_material/   # recovered historical notebooks/scripts and pilot evidence
└── documentation/
    ├── dataset_sources.md
    ├── dataset_audit_summary.json
    ├── dataset_split_summary.json
    ├── model_comparison.csv
    ├── model_architecture.md
    ├── training_configuration.json
    ├── v3_final_evaluation.json
    ├── v4_training_report.json
    ├── v4_model_comparison.csv
    ├── v4_phase1_history.csv
    ├── v4_phase2_history.csv
    ├── v4_release_notes.md
    └── package_validation.md
```

### Recommended professor walkthrough

1. Notebook 01 — dataset source archives, real/AI source accounting and assembly checks.
2. Notebook 02 — quality audit methods, 183-image review queue and manual-review rationale.
3. Notebook 03 — Dataset V2 split counts, seed 211 and split/leakage limitations.
4. Notebook 04 — five original model approaches and the small V1 baseline.
5. Notebook 05 — Dataset V2 model comparison.
6. Notebook 06 — historical EfficientNetB0 V3 architecture and training procedure.
7. Notebook 07 — historical V3 evaluation and loader limitations.
8. Notebook 08 — V4 dataset expansion and fine-tuning methodology.
9. Notebook 09 — V4 final metrics, confusion matrix, V3 comparison and limitations.

The notebooks in the main sequence are documented walkthroughs/reconstructions where the exact original source notebook was unavailable. The `source_material/` subfolder preserves the original notebooks/scripts that were recovered. It includes an earlier 198-image general-object pilot; that pilot is historical context and is not the 1,250-image car Dataset V2.

## 7. Installation and running on macOS Apple Silicon

The recorded application runtime used Python 3.11 ARM64, TensorFlow 2.21.0, Keras 3.13.2, NumPy 2.1.3 and Streamlit 1.66.1.

```bash
# From the project root
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the key dependencies:

```bash
python -c "import tensorflow as tf, streamlit, numpy; print(tf.__version__, streamlit.__version__, numpy.__version__)"
```

Verify the active CNN model and run a forward-pass smoke test:

```bash
python scripts/verify_active_cnn.py
```

Optionally pass a local car photograph:

```bash
python scripts/verify_active_cnn.py /path/to/car_photo.jpg
```

Launch the application:

```bash
python -m streamlit run app_cnn.py
```

Install notebook dependencies only if you plan to open/run the academic notebooks:

```bash
python -m pip install -r requirements-notebooks.txt
jupyter lab
```

Some notebook cells need external images, manifests or source archives which are intentionally not distributed. The markdown in each notebook identifies those limits; do not treat the reconstructed walkthroughs as proof that absent data has been re-created.

## 8. What is and is not included

Included: the single active V4 model, the Streamlit application, configuration, inference code, verification script, academic notebooks, recovered historical source notebooks/scripts, compact JSON/CSV evidence, training utilities and documented V4 evaluation outputs.

Excluded from GitHub: V3 model weights, raw Dataset V2/V4 images, source ZIP archives, generated AI candidate images, virtual environments, local caches, intermediate `data_v4*` folders, `outputs_v4*` folders, temporary challenge-evaluation folders and duplicate model checkpoints.

The V3 checkpoint was used as V4's fine-tuning starting point. Its historical SHA-256 is retained in `documentation/training_configuration.json` and `documentation/v4_training_report.json`, but the weights are not shipped. This means the repository is fully runnable for V4 inference; reproducing V4 training from scratch additionally requires the external V3 starting checkpoint, Dataset V2 source images and 54 candidate images/manifest.

## 9. Limitations and future scope

- The model estimates visual patterns, not provenance or authenticity proof.
- Real-image labels were not independently authenticated one-by-one; AI and generator labels were inherited from source manifests/folders.
- V3's loader skipped seven WebP files; its historical metrics use 872/185/186 images rather than the complete 874/188/188 split.
- V4 used all 1,304 files in its recorded split, but source groups overlap across splits in the project record.
- The nine new-generator test images are too few for strong GPT/Gemini/Meta AI claims, and these families were represented in the V4 development pool.
- The displayed sigmoid score is uncalibrated.
- Marketplace compression, screenshots, edits, future generators and camera differences may affect performance.

Future work could include a larger independently reviewed dataset, truly held-out generator-family testing, score calibration and robustness testing under marketplace compression. These are recommendations, not completed work.

## 10. Credits and source acknowledgements

- Real-image sources: the project's existing real-car collection and [`Smily6820/car-images`](https://huggingface.co/datasets/Smily6820/car-images) on Hugging Face.
- Dataset assembly source archives: `real_car_dataset_v2_deduplicated.zip`, `Car_AI_Images_FGTS_Additions.zip` and `Car_Image_Authenticity_Dataset_v1.zip`.
- AI generator families explicitly recorded in the FGTS extraction summary include LDM, Stable Diffusion 1.4 and ProGAN; additional generator identities are only reported where source manifests support them.
- Technical backbone: ImageNet-pretrained EfficientNetB0 fine-tuned for binary image classification.
- Application: Python, Streamlit, TensorFlow, Keras, NumPy and Pillow.

The source image archives are not redistributed here. Review upstream licences and terms before redistributing any third-party source images.

## 11. Active model integrity

- File: `models_v4/car_cnn_model_v4.keras`
- Size: 34,358,590 bytes
- SHA-256: `36b6e3fe95278cb73923ec01e109671278fe97877e5bb7beae4b947fcd3f6d3e`
- Threshold: `0.50`
- Input: `256 × 256 × 3`, RGB float32, raw 0–255 range.

The V3 starting-checkpoint SHA-256 is `ea47dc1e9ded13944a2ac866ae968f3956a003d9faaf39ce4a0dfa8a8cad65b5`; the V3 weight file is not part of this final repository. The V4 inference loader validates the active file hash before TensorFlow loads the model.
