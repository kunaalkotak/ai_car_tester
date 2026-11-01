# Listing Lens — Used Car Photo Authenticity Checker

**Listing Lens** is a Streamlit application that helps people buying second-hand cars take a closer look at photographs used in online listings. Upload a car photo and the app uses one image-classification model, displayed in the application as **CNN**, to assess whether the image looks more like an ordinary photograph or AI-generated imagery.

> **Important:** Listing Lens is an experimental image-screening tool. It detects visual patterns learned from a labelled dataset; it does not prove that an image is authentic or fake. It cannot establish whether a vehicle exists, whether a seller is genuine, whether the photograph belongs to the advertised vehicle, or whether a listing is fraudulent. Always verify the car and seller independently.

---

## 1. The problem

Used-car buyers often discover vehicles through online marketplaces and classifieds such as OLX, CarDekho and Cars24. A listing may contain genuine photographs, heavily edited images, reused photographs, or AI-generated imagery. Buyers may not know whether an image deserves extra scrutiny before travelling to view a vehicle or contacting a seller.

Listing Lens provides an additional, limited signal: it classifies the visual appearance of an uploaded car photograph. It is intended to help a buyer decide what questions to ask next—not to replace vehicle checks, seller verification or an in-person inspection.

## 2. Target users and intended use

The primary users are people browsing second-hand car advertisements who want a quick first-pass check of a listing photograph.

A typical workflow is:

1. A buyer saves a photograph from an online used-car listing.
2. The buyer uploads the image to Listing Lens.
3. The application runs the single CNN model and displays a photo assessment and an AI-pattern score.
4. The buyer uses the result as a prompt to request fresh images, a live walk-around video and independent vehicle checks.

The assessment applies only to the uploaded image. It does not check a listing URL, seller profile, registration number, VIN/chassis number, ownership record, mileage, service history, vehicle condition, availability or image provenance.

## 3. How Listing Lens works

```text
User uploads a car photograph
              │
              ▼
Image validation and preview
              │
              ▼
RGB decoding and 256 × 256 resize
              │
              ▼
          CNN model
     (EfficientNetB0 backbone)
              │
              ▼
       Sigmoid model score
              │
              ▼
     Apply 0.50 threshold
          /       \
         /         \
        ▼           ▼
 Photograph-     AI-generated-
 looking         looking
        \           /
         ▼         ▼
       Result, score and buyer guidance
```

The app loads the saved model once and reuses it for later predictions during the Streamlit session. It accepts a single image, creates a preview, preprocesses the image, obtains the model's sigmoid output and compares that output with the fixed decision threshold of **0.50**.

- A score **below 0.50** is classified as `Real (0)` / photograph-looking.
- A score **at or above 0.50** is classified as `AI-generated (1)` / AI-generated-looking.

The displayed score is an **uncalibrated model output**. It is not a calibrated probability that a photograph is AI-generated, and it is not the probability that a listing is fraudulent.

## 4. Main application features

- Single-model inference: the app automatically uses the selected CNN model. There is no model selector or classical-model comparison panel.
- Buyer-oriented interface with the Listing Lens theme and the message: **“Found a used car online? Check its photo before you make the trip.”**
- Image upload and preview, including image dimensions and file size.
- Support for JPG, JPEG, PNG, WebP, BMP and GIF uploads. Animated GIFs use their first frame.
- A plain-language result card explaining the model's assessment.
- A score display showing the AI-pattern model score and the 0.50 decision boundary.
- Practical guidance to request fresh images, a walk-around video, vehicle documents and an independent inspection.
- A clear explanation of what the tool can and cannot establish.

The model is displayed to ordinary application users only as **CNN**. The technical architecture is documented below for project reviewers.

## 5. Dataset collection and sourcing

The project dataset was assembled from multiple sources; it was not simply downloaded as one Kaggle dataset. Dataset V2 was designed as a balanced collection of **1,250 image candidates**: 625 real-image candidates and 625 AI-generated-image candidates.

The project records identify these three source archives as the inputs to the dataset assembly process:

1. `real_car_dataset_v2_deduplicated.zip`
2. `Car_AI_Images_FGTS_Additions.zip`
3. `Car_Image_Authenticity_Dataset_v1.zip`

The real-image collection included material from the project's existing real-car image collection and the Hugging Face dataset [`Smily6820/car-images`](https://huggingface.co/datasets/Smily6820/car-images).

The AI-generated portion was assembled from:

| AI-generated image source | Candidate images |
|---|---:|
| FGTS AI additions archive | 474 |
| AI images from the original V1 dataset | 151 |
| **Total AI-generated candidates** | **625** |

The recovered FGTS extraction summary records **158 LDM, 158 Stable Diffusion 1.4 and 158 ProGAN candidates** (474 in total). This breakdown applies only to the FGTS additions archive at that extraction stage. The separate 151-image contribution carried over from V1 does not have a recovered per-generator breakdown. An earlier 100-image car-generation pilot notebook also exists (34 SD 1.5, 33 SDXL Turbo and 33 Tiny-SD planned), but it is preserved as a distinct pilot and is not counted as part of Dataset V2 without matching per-image provenance.

Combining multiple real and AI image sources was intended to increase variation in vehicle appearance, backgrounds, image composition and image-generation styles. It also reduced dependence on a single source archive. This broader source mix does not, by itself, prove that the final dataset represents every real-world camera, marketplace or AI generator.

### Dataset balance

| Class | Candidate images |
|---|---:|
| Real | 625 |
| AI-generated | 625 |
| **Total** | **1,250** |

The dataset archives and image files are deliberately not included in this GitHub-ready application repository. They are not required for inference because the trained model is packaged separately.

## 6. Dataset cleaning and quality audit

The dataset preparation process included several checks to reduce duplicates, invalid files and unsuitable images before model development. The recorded checks included:

- **Image deduplication:** comparing image files to identify duplicate content.
- **SHA-256 checks:** comparing cryptographic file hashes.
- **Pixel-level hash checks:** finding files with identical decoded pixel content, including files saved with different encodings.
- **Perceptual-hash checks:** identifying visually similar images that may differ slightly in encoding, size or appearance.
- **Image readability verification:** checking that image files could be opened and decoded.
- **Resolution and aspect-ratio checks:** identifying unusually small or unusually shaped images.
- **Brightness and blur checks:** flagging images with potentially problematic visual quality.
- **Car object-detection screening:** using object detection as an additional check that a candidate image appeared to contain a car.
- **Manual review:** examining images flagged by the automated checks.

The first-pass quality audit processed all **1,250 candidates** and identified **183 images for review**. These 183 images were not automatically deleted. The review approach was to retain images that were readable and usable even when they had imperfect composition, unusual aspect ratios or object-detection failures.

Object detection was used only as a quality-screening aid. A failed detection was not treated as proof that an image did not contain a car or was unusable. Likewise, a successful detection was not treated as proof of image authenticity.

### Dataset-label and audit limitations

- The authenticity of every real-image label was not independently verified.
- AI labels and generator identities were inherited from source manifests or source-folder labels.
- Brightness, blur, resolution, perceptual-hash and object-detection flags were heuristic indicators, not proof that an image was unusable.
- Combining sources and running quality checks improves dataset organisation but does not eliminate label noise or source bias.

The source archives, raw images and detailed audit manifests are not part of this cleaned code ZIP. The dataset history above records the project dataset notes used for this README; it should not be interpreted as a fresh re-audit of the absent source images.

## 7. Train, validation and test split

Dataset V2 was divided into training, validation and test sets using a target ratio of approximately **70:15:15** and random seed **211**.

| Split | Real | AI-generated | Total |
|---|---:|---:|---:|
| Training | 437 | 437 | 874 |
| Validation | 94 | 94 | 188 |
| Testing | 94 | 94 | 188 |
| **Total** | **625** | **625** | **1,250** |

The split process included hash and group checks. These checks were part of dataset preparation; they should not be interpreted as proof that the final model evaluation was isolated by generator family or by every source group. In particular, the recorded V3 evaluation is **not a held-out-generator test**, and the V3 metadata notes source-group overlap across splits.

### Important V3 training-loader limitation

The original V3 training script used Keras' directory-loading pipeline, which skipped some WebP files. As a result, the V3 training run did not load every image in the expected split:

| Split | Expected images | Actually loaded by V3 run | Not loaded |
|---|---:|---:|---:|
| Training | 874 | 872 | 2 |
| Validation | 188 | 185 | 3 |
| Testing | 188 | 186 | 2 |

The V3 performance metrics in this README were obtained using the smaller loaded counts shown above. In particular, the V3 test accuracy is based on **186 loaded test images**, not the full 188-image test split. No corrected V3 result is claimed here.

The inference application accepts WebP uploads through a separate Pillow RGB-decoding and bilinear-resize fallback for usability. This does not change the fact that WebP files were skipped by the recorded V3 training loader and were not represented in that training run in the same way as JPEG, PNG, BMP and GIF files.

## 8. Model development history

The project began with five model approaches. The purpose of the initial experiments was to compare conventional machine-learning approaches with a CNN-based image classifier before selecting one model for the final application.

### 8.1 The original five models

1. Logistic Regression
2. Support Vector Machine (SVM) with an RBF kernel
3. Random Forest
4. Linear Discriminant Analysis (LDA)
5. A custom CNN trained from scratch

The four classical models used flattened **32 × 32 RGB** images together with PCA and StandardScaler preprocessing. The original custom CNN used 256 × 256 RGB input and a small convolutional architecture:

- Input: 256 × 256 × 3.
- Conv2D with 16 filters.
- Conv2D with 32 filters.
- Conv2D with 64 filters.
- Max pooling.
- Global Average Pooling.
- Dropout.
- Dense layer.
- Sigmoid output for binary classification.

### 8.2 Original V1 experiment

The earlier V1 experiment used a small test set containing only **46 images**. Its historical validation and test accuracies were:

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| Custom CNN | 88.89% | 78.26% |
| Random Forest | 88.89% | 71.74% |
| SVM (RBF) | 84.44% | 76.09% |
| Logistic Regression | 82.22% | 71.74% |
| LDA | 82.22% | 69.57% |

These are historical baseline results from the smaller V1 experiment. They should not be compared as if they came from the same test set or protocol as the later Dataset V2 and V3 experiments.

### 8.3 Dataset V2 comparison

After assembling Dataset V2, the five approaches were retrained or evaluated using the expanded dataset. The custom CNN V2 results were:

| Metric | Custom CNN V2 result |
|---|---:|
| Test accuracy | 82.45% |
| AI precision | 77.98% |
| AI recall | 90.43% |
| AI F1-score | 83.74% |
| ROC-AUC | 89.43% |
| Selected threshold | 0.32 |

The Dataset V2 classical model test results were:

| Model | Test accuracy | AI F1-score |
|---|---:|---:|
| SVM (RBF) | 75.53% | 74.16% |
| Random Forest | 73.40% | 72.83% |
| Logistic Regression | 65.96% | 64.44% |
| LDA | 64.36% | 62.98% |

The results suggested that the custom CNN was stronger than the four classical approaches in the recorded V2 evaluation. The project then moved beyond the classical feature-based models and experimented with transfer learning using EfficientNetB0.

## 9. EfficientNetB0 model development history

The current application model is **EfficientNetB0 V4**, fine-tuned from the project's EfficientNetB0 V3 checkpoint. In the Streamlit application, it is displayed simply as **CNN**. For technical transparency, its architecture is an **ImageNet-pretrained EfficientNetB0 backbone** fine-tuned for binary image classification.

### 9.1 Model architecture

- Input image: 256 × 256 × 3 RGB.
- EfficientNetB0 backbone with the original classification top removed.
- GlobalAveragePooling2D.
- Dropout(0.40).
- Dense(64, ReLU) with L2 regularisation of 1e-4.
- Dropout(0.30).
- Dense(1, sigmoid) output.

The model accepts raw RGB values in the 0–255 range. EfficientNetB0's built-in preprocessing performs the model's input normalisation; the inference code must not divide the input by 255 a second time.

### 9.2 Two-phase training procedure

**Phase 1 — Train the classification head**

- Keep the pretrained EfficientNetB0 backbone frozen.
- Train the new classification head.
- Learning rate: **0.001**.
- Maximum: **15 epochs**.

**Phase 2 — Fine-tune the backbone**

- Unfreeze the top 40 backbone layers.
- Keep Batch Normalization layers frozen.
- Fine-tune with learning rate **0.00001**.
- Maximum: **25 epochs**.

Other recorded techniques included data augmentation, dropout, L2 regularisation, label smoothing (0.05), early stopping and learning-rate reduction. Training augmentation included horizontal flipping, small rotations, zoom, contrast changes and brightness changes. Augmentation is a training-time procedure; the inference application does not randomly augment the buyer's uploaded photograph.

### 9.3 Previous V3 experiment results (preserved baseline)

The preliminary V3 metrics were recorded as follows:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 97.84% | 94.09% |
| Balanced accuracy | 97.87% | 94.10% |
| AI precision | 100.00% | 95.60% |
| AI recall | 95.74% | 92.55% |
| AI F1-score | 97.83% | 94.05% |
| ROC-AUC | 99.64% | 99.53% |
| Images loaded | 185 | 186 |

**Selected decision threshold: 0.47.** The threshold was selected using validation balanced accuracy (the middle of a near-optimal plateau) and was frozen before test evaluation.

The V3 results were produced from **872 training images, 185 validation images and 186 test images**. They are not results from the full 874/188/188 expected split. The 94.09% test accuracy must therefore be read as an observed result on 186 loaded test images, not as a fully evaluated 188-image test result.

The V3 test set was not used for training or threshold tuning. However, this evaluation was not a held-out-generator test. The project has not established that the model generalises to completely unseen image generators, image-generation services or future model families.

### 9.4 Historical V1–V3 experiment comparison

The following table brings together the three stages of model development. The results are reported in their original evaluation contexts; different dataset versions and test-set sizes mean they are not a perfectly controlled head-to-head comparison.

| Experiment | Model | Validation accuracy | Test accuracy | Test AI F1 | Evaluation note |
|---|---|---:|---:|---:|---|
| V1 | Custom CNN | 88.89% | 78.26% | Not supplied | 46-image test set |
| V1 | Random Forest | 88.89% | 71.74% | Not supplied | 46-image test set |
| V1 | SVM (RBF) | 84.44% | 76.09% | Not supplied | 46-image test set |
| V1 | Logistic Regression | 82.22% | 71.74% | Not supplied | 46-image test set |
| V1 | LDA | 82.22% | 69.57% | Not supplied | 46-image test set |
| V2 | Custom CNN | Not repeated here | 82.45% | 83.74% | Dataset V2 experiment; effective loaded count not recovered |
| V2 | SVM (RBF) | Not repeated here | 75.53% | 74.16% | Dataset V2 experiment; effective loaded count not recovered |
| V2 | Random Forest | Not repeated here | 73.40% | 72.83% | Dataset V2 experiment; effective loaded count not recovered |
| V2 | Logistic Regression | Not repeated here | 65.96% | 64.44% | Dataset V2 experiment; effective loaded count not recovered |
| V2 | LDA | Not repeated here | 64.36% | 62.98% | Dataset V2 experiment; effective loaded count not recovered |
| V3 | EfficientNetB0 experiment (historical app model) | 97.84% | 94.09% | 94.05% | 185 validation / 186 test images loaded |
| V4 | Fine-tuned EfficientNetB0 V3 (current app name: CNN) | 95.41% | 93.91% | 94.23% | 196 validation / 197 test images loaded; full 1,304-image split |

The V1 AI F1 values and the V2 validation accuracy values are not reproduced in this summary table because they were not provided as part of the historical figures used for this README. The individual V1/V2 result tables above preserve the reported metrics without filling in missing values.

## 10. Current application model: EfficientNetB0 V4

The active Streamlit application now loads **V4**, a fine-tuned continuation of the V3 EfficientNetB0 checkpoint. The user-facing model name remains **CNN**. V3 has been retained unchanged in `models_v3/` as a rollback backup. V4 is stored separately in `models_v4/`.

V4 was fine-tuned using the complete Dataset V2 split plus 54 new AI candidates from GPT (26), Gemini (19) and Meta AI (9). Only AI images were added; no real images were added or removed. The final split contained 1,304 images: 911 training (437 real, 474 AI), 196 validation (94 real, 102 AI) and 197 test (94 real, 103 AI). The training report confirms that all 1,304 files were loaded.

**V4 training configuration**

| Setting | V4 value |
|---|---|
| Starting checkpoint | Preserved EfficientNetB0 V3 |
| Input | 256 × 256 × 3 RGB; float32, raw 0–255 |
| Phase 1 | Frozen backbone; classification head; learning rate 0.0001; up to 5 epochs |
| Phase 2 | Top 40 backbone layers fine-tuned; Batch Normalization frozen; learning rate 0.00001; up to 12 epochs |
| Optimizer | Adam |
| Loss | Binary cross-entropy with label smoothing 0.05 |
| Batch size | 8 |
| Class weighting | 1.0423 (real), 0.9610 (AI-generated) |
| Random seed | 42 |

### V4 results

The decision threshold of **0.50** was selected using V4 validation balanced accuracy and frozen before test evaluation.

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 95.41% | 93.91% |
| Balanced accuracy | 95.46% | 93.85% |
| AI precision | 96.97% | 93.33% |
| AI recall | 94.12% | 95.15% |
| AI F1-score | 95.52% | 94.23% |
| ROC-AUC | 99.36% | 98.89% |
| Images loaded | 196 | 197 |

Test confusion matrix (rows = actual [real, AI]; columns = predicted [real, AI]): `[[87, 7], [5, 98]]`. V4 detects more AI examples than V3 at the validation-selected comparison thresholds, but has lower AI precision and more real-image false positives.

### V3 versus V4 on the same V4 test set

Each model's operating threshold was selected independently on the same V4 validation split (V3 = 0.40; V4 = 0.50). This is the fairest threshold-based comparison available in the training report.

| Metric | V3 backup | V4 active |
|---|---:|---:|
| Accuracy | 93.40% | **93.91%** |
| Balanced accuracy | 93.50% | **93.85%** |
| AI precision | **95.92%** | 93.33% |
| AI recall | 91.26% | **95.15%** |
| AI F1-score | 93.53% | **94.23%** |
| ROC-AUC | 97.98% | **98.89%** |

The V4 improvement is real but modest. It favours AI recall and AUC, while V3 makes fewer false-positive AI predictions on genuine test photographs (4 versus 7). The final V4 threshold of 0.50 reflects the validation balanced-accuracy selection; it is not a calibrated probability boundary.

### New-generator subset

The V4 test split included nine new AI candidates: GPT 4/4 detected, Gemini 3/3 detected and Meta AI 2/2 detected. These counts are too small for reliable generator-specific claims. All three generator families were represented in the V4 development/training pool, so this is not a held-out-generator evaluation. The labels are based on user-provided source folders and were not independently authenticated.

V4 was selected as the active application model for this project release, while V3 remains available for rollback. This is a practical project decision, not evidence that V4 will generalise to all future AI generators or marketplace conditions. See [V4 training report](documentation/v4_training_report.json), [V4 release notes](documentation/v4_release_notes.md), [V4 model comparison](documentation/v4_model_comparison.csv) and the retained [V3 evaluation](documentation/v3_final_evaluation.json).

## 11. Model threshold and score interpretation

The final application uses a fixed decision threshold of **0.50**:

| CNN output | Application label |
|---|---|
| Score < 0.50 | Real / photograph-looking (class 0) |
| Score ≥ 0.50 | AI-generated-looking (class 1) |

The threshold was selected on the validation set using balanced accuracy and frozen before test evaluation. It is stored in `config/cnn_inference_config.json` and imported by the inference module, so the application and configuration share one value.

The displayed score is the model's sigmoid output expressed on a 0–100 display scale. It is **not calibrated**. For example, a score of 80/100 should not be interpreted as an 80% probability that the image is AI-generated, or an 80% probability that a seller is dishonest. It only describes the raw model signal relative to the project's chosen decision threshold.

## 12. Inference preprocessing

Preprocessing is part of the model pipeline. The inference code follows the recorded training input route: RGB decoding, direct bilinear resize to 256 × 256 and raw 0–255 float32 values. EfficientNetB0 normalization remains inside the saved model.

| Step | Inference behaviour |
|---|---|
| Colour | Decode as RGB with 3 channels |
| Target size | 256 × 256 pixels |
| Resize | Bilinear resize directly to a square |
| Cropping | No centre crop; the original aspect ratio is stretched to the square target |
| Tensor type | `float32` |
| Pixel range | Raw RGB values in the 0–255 range |
| Normalisation | Performed inside EfficientNetB0; inference does not divide by 255 |
| GIF | Uses the first frame |
| WebP | Pillow RGB decoding and bilinear resize as an inference-only fallback |

JPEG, PNG, BMP and GIF inputs are decoded with `tf.io.decode_image(..., channels=3, expand_animations=False)` and resized with `tf.image.resize(..., method="bilinear")`. WebP bytes are decoded with Pillow, converted to RGB and bilinearly resized directly to a float32 array. This is an inference-time compatibility path. The V4 training report records that all 1,304 files in its V4 split were loaded; the earlier V3 WebP shortfall is retained as historical context in the V3 evaluation record.

The user-facing preview applies EXIF orientation so that the image appears correctly in the interface. The model preprocessing follows the documented training decode-and-resize route for formats represented in training, with the separate WebP fallback described above.

## 13. Project structure and academic development trail

The repository deliberately separates the **active single-model application** from the **academic notebooks and historical evidence**. The application imports only the active V4 CNN inference path; the preserved V3 file is a rollback backup and is not loaded by the UI. The notebooks preserve the ML development story without restoring the older models to the live UI.

```text
Listing-Lens/
├── .gitignore
├── .streamlit/
│   └── config.toml
├── app_cnn.py
├── car_app_ui.py
├── requirements.txt
├── requirements-notebooks.txt       # Optional research/notebook dependencies
├── README.md
├── config/
│   └── cnn_inference_config.json
├── models_v3/
│   └── car_cnn_model_v3.keras        # Preserved V3 rollback backup
├── models_v4/
│   └── car_cnn_model_v4.keras        # Active application model (CNN)
├── src/
│   ├── __init__.py
│   └── cnn_inference.py              # Only active inference implementation
├── scripts/
│   └── verify_active_cnn.py          # Active model load/prediction smoke test
├── notebooks/
│   ├── 01_dataset_collection_assembly.ipynb
│   ├── 02_dataset_quality_audit.ipynb
│   ├── 03_train_validation_test_split.ipynb
│   ├── 04_original_five_model_experiments.ipynb
│   ├── 05_dataset_v2_model_comparison.ipynb
│   ├── 06_efficientnetb0_v3_training.ipynb
│   ├── 07_final_model_evaluation.ipynb
│   └── source_material/
│       ├── early_198_image_pilot/    # Recovered original notebooks and pilot artifacts
│       ├── ai_image_generation/      # Earlier 100-image car generation notebook
│       └── dataset_collection/      # Recovered HF selection/threshold scripts
└── documentation/
    ├── dataset_sources.md
    ├── dataset_audit_summary.json
    ├── dataset_split_summary.json
    ├── model_architecture.md
    ├── model_comparison.csv
    ├── training_configuration.json  # V3 historical training config
    ├── v3_final_evaluation.json
    ├── final_evaluation.json        # Current V4 evaluation
    ├── v4_training_report.json
    ├── v4_phase1_history.csv
    ├── v4_phase2_history.csv
    ├── v4_model_comparison.csv
    ├── v4_release_notes.md
    ├── source_availability.md
    ├── source_evidence/
    │   └── fgts_collection_summary.json
    └── figures/
        └── v3_test_confusion_matrix_reconstructed.png
```

### Seven main notebooks — recommended professor walkthrough

| Stage | Notebook | What it explains |
|---|---|---|
| 1 | [Dataset collection and assembly](notebooks/01_dataset_collection_assembly.ipynb) | Three source archives, real/AI sources, 625/625 balancing, source manifests, SHA-256 and pixel hashes |
| 2 | [Dataset quality audit](notebooks/02_dataset_quality_audit.ipynb) | Readability, dimensions, aspect ratio, brightness, blur, hashes, car-detection screening, review flags and contact sheets |
| 3 | [Train/validation/test split](notebooks/03_train_validation_test_split.ipynb) | 70:15:15 target, seed 211, class counts, grouping and hash/leakage checks; V3 WebP loader shortfall |
| 4 | [Original five-model experiments](notebooks/04_original_five_model_experiments.ipynb) | Logistic Regression, SVM, Random Forest, LDA and custom CNN; historical V1 baseline results |
| 5 | [Dataset V2 model comparison](notebooks/05_dataset_v2_model_comparison.ipynb) | V2 CNN and classical model results, metrics and threshold interpretation |
| 6 | [EfficientNetB0 V3 training](notebooks/06_efficientnetb0_v3_training.ipynb) | ImageNet backbone, model head, augmentation, frozen-head training and fine-tuning |
| 7 | [Final model evaluation](notebooks/07_final_model_evaluation.ipynb) | Historical V3 threshold 0.47, validation/test metrics, reconstructed test confusion matrix, ROC/error-analysis requirements and limitations |

The seven top-level notebooks are **documented reconstructions/guide notebooks**, not falsely represented as the original executed notebooks. This is particularly important for Stages 1–3 and 6–7: the original V2 per-image source/audit/split manifests and the V3 `train_improved.py` (or equivalent training source) were not recovered. Each notebook states what is recorded, what is reconstructed, and what cannot be reproduced without the source images or missing artifacts.

### Recovered original notebooks and evidence

The `notebooks/source_material/` folder preserves the original notebooks that were available. The six notebooks in `early_198_image_pilot/` cover an earlier **198-image general-object pilot**, using a seed-42 split and 30-image test evaluation. That work is valuable historical evidence but is **not** the later 1,250-image car Dataset V2. Its original validation/test CSVs, training history and figures are retained alongside the notebooks. The old pilot model weights were intentionally not copied.

The archived car-image generation notebook documents a separate 100-image generation pilot (34 SD 1.5, 33 SDXL Turbo, 33 Tiny-SD planned). It is not used to alter the documented V2 counts. The recovered Hugging Face script documents one 51-image selection workflow, not all 625 real images. Read [source availability and reconstruction notes](documentation/source_availability.md) before treating any archived artifact as evidence for a particular experiment.

### Supporting documentation and evidence

- [Dataset sources and provenance](documentation/dataset_sources.md)
- [Dataset audit summary](documentation/dataset_audit_summary.json)
- [Dataset split summary](documentation/dataset_split_summary.json)
- [Model architecture](documentation/model_architecture.md)
- [Model comparison CSV](documentation/model_comparison.csv)
- [V3 historical training configuration](documentation/training_configuration.json)
- [V4 training report](documentation/v4_training_report.json)
- [V4 model comparison](documentation/v4_model_comparison.csv)
- [Current V4 evaluation summary](documentation/final_evaluation.json)
- [V4 release notes](documentation/v4_release_notes.md)
- [Source availability and reconstruction notes](documentation/source_availability.md)
- [Package validation and reproducibility boundary](documentation/package_validation.md)

The source image archives and all 1,304 V4 images are deliberately excluded. The academic notebooks include reusable checks and clearly labelled reconstructions, but the full historical dataset assembly/audit/split and V3 retraining cannot be reproduced from this ZIP alone without the source images, original per-image manifests and original training script. V4 training outputs and histories are preserved as evidence; the exact training script used for the completed V4 run was not included in the supplied artifacts. The application remains an inference-only package with one active CNN model and one preserved rollback checkpoint.

## 14. Technology stack

| Technology | Use |
|---|---|
| Python 3.11 | Application runtime used for the recorded V3 environment |
| Streamlit | Web application and interactive user interface |
| TensorFlow 2.21.0 | Load and run the saved neural-network model; decode and resize images |
| Keras 3.13.2 | Saved `.keras` model format and model API |
| NumPy 2.1.3 | Image tensor and prediction array handling |
| Pillow | Image preview, EXIF orientation and WebP fallback decoding |

The final application has one active inference model (V4); V3 is retained separately as a rollback backup and is not loaded by the UI. Scikit-learn, joblib, PCA and StandardScaler are not runtime dependencies because the classical models have been removed from the application.

## 15. Installation on macOS Apple Silicon

The recorded V3 and V4 runs used the Python 3.11 ARM64 environment with TensorFlow 2.21.0, Keras 3.13.2 and NumPy 2.1.3. TensorFlow 2.21.0 provides a macOS ARM64 wheel for Python 3.11; this is the appropriate package family for an Apple Silicon Mac. See the [official TensorFlow pip installation guide](https://www.tensorflow.org/install/pip) and [TensorFlow package files](https://pypi.org/project/tensorflow/).

### Step 1 — Check Python and architecture

Open Terminal in the project folder and check that Python is running natively as ARM64:

```bash
python3.11 --version
python3.11 -c "import platform; print(platform.machine())"
```

On an Apple Silicon Python installation, the architecture check should print `arm64`. Use a native Apple Silicon Python installation rather than an x86_64 Python running through Rosetta.

### Step 2 — Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Upgrade pip and install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow is a large package. Installation may take some time and requires an internet connection. The project does not require a separate dataset download to run the application.

### Step 4 — Verify the runtime

```bash
python -c "import tensorflow as tf, streamlit, numpy; print('TensorFlow:', tf.__version__); print('Streamlit:', streamlit.__version__); print('NumPy:', numpy.__version__)"
```

The expected pinned versions are TensorFlow `2.21.0`, Streamlit `1.66.1` and NumPy `2.1.3`.

### Step 5 — Run Listing Lens

```bash
streamlit run app_cnn.py
```

Streamlit prints a local address in the terminal. Open the displayed local URL (usually `http://localhost:8501`) in your browser.

To stop the app, press `Control + C` in the terminal. To leave the virtual environment, run:

```bash
deactivate
```

### Optional — install and open the academic notebooks

The app-only dependencies are kept minimal. To work through the notebooks, reactivate the same virtual environment and install the optional research dependencies:

```bash
source .venv/bin/activate
python -m pip install -r requirements-notebooks.txt
jupyter lab
```

Open the seven notebooks in `notebooks/` in numerical order. The recovered originals are under `notebooks/source_material/`. Some dataset cells need the external source archives and image folders, which are not distributed in this ZIP; the notebook Markdown explains those reproducibility boundaries.

## 16. GitHub setup and upload

The repository retains the active V4 model and the V3 rollback backup:

```text
models_v4/car_cnn_model_v4.keras  # active
models_v3/car_cnn_model_v3.keras  # backup
```

Each model is approximately 34 MB (about 32.8 MiB). This is larger than GitHub's 25 MiB per-file browser-upload limit, but below the 100 MiB per-file limit for command-line Git uploads. Therefore, use Git from Terminal rather than dragging the model into GitHub's browser uploader. GitHub requires Git LFS for individual files larger than 100 MiB. See [GitHub's file upload guidance](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository) and [repository limits](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits).

### Step 1 — Create an empty GitHub repository

Create a new repository on GitHub. Do not initialise it with a README or other files if you plan to push this project as the first commit.

### Step 2 — Initialise Git locally

Run these commands from the root of the extracted `Listing-Lens` folder:

```bash
git init
git add .
git status
git commit -m "Prepare Listing Lens single-model application"
git branch -M main
```

Review `git status` before committing. The `.gitignore` excludes local virtual environments, caches, raw dataset folders, known source dataset ZIP archives and Streamlit secrets. It keeps the small CSV evidence files in `documentation/` and the preserved historical pilot folder under version control. It does not ignore the selected `.keras` model.

### Step 3 — Connect the GitHub repository and push

Replace the URL below with the HTTPS URL of your own empty GitHub repository:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Both model files will be included in the push. Each is below GitHub's 100 MiB individual-file limit, so Git LFS is not required for this release.

## 17. Limitations and responsible use

1. **Not forensic verification:** the model estimates whether an image resembles patterns learned from the training data. It cannot prove an image's origin or authenticity.
2. **Not fraud detection:** a real photograph can be used in a fraudulent listing; an AI-looking image does not prove that the vehicle or seller is fake.
3. **Inherited labels:** real-image labels were not independently verified image by image, and AI labels and generator names were inherited from source labels/manifests.
4. **Small evaluation sets:** V3 validation used 185 images and V3 testing used 186 images. Metrics from samples of this size have uncertainty.
5. **Historical V3 split loading:** the V3 loader skipped two training, three validation and two test WebP files. V4 loaded all 1,304 files in its recorded split.
6. **No held-out-generator test:** generator families GPT, Gemini and Meta AI appear in the V4 development/training pool and in the nine-image new-generator subset. Generalisation to completely unseen generator families has not been demonstrated.
7. **Score calibration:** the displayed sigmoid score is not a calibrated probability.
8. **Image distribution changes:** marketplace compression, screenshots, filters, image edits, new cameras and new generative models may affect performance.
9. **Image-format handling:** WebP uploads are accepted through the Pillow preprocessing fallback. V4 training loaded all 1,304 files in its reported split; V3 had the historical WebP shortfall described above.
10. **No buyer due diligence replacement:** always inspect the vehicle, verify its documents and seller, and avoid paying a deposit based only on an online photograph or model output.

### Future scope

Potential next steps include obtaining a larger independently reviewed and genuinely held-out-generator evaluation set, calibrating model scores, examining false positives and false negatives by source, and testing robustness to marketplace compression and image editing. These are future recommendations, not work claimed to have been completed in this project version.

## 18. Credits and dataset acknowledgements

- **Real-image source acknowledgement:** project real-car image collection and the Hugging Face dataset [`Smily6820/car-images`](https://huggingface.co/datasets/Smily6820/car-images).
- **Dataset assembly archives:** `real_car_dataset_v2_deduplicated.zip`, `Car_AI_Images_FGTS_Additions.zip` and `Car_Image_Authenticity_Dataset_v1.zip`.
- **AI-generated image labels:** derived from the source archive labels/manifests, including generator-family labels such as Stable Diffusion variants and SDXL where recorded.
- **Model architecture:** EfficientNetB0 with ImageNet-pretrained weights, fine-tuned for the project's binary image-classification task.
- **Application and inference stack:** Python, Streamlit, TensorFlow, Keras, NumPy and Pillow.

The source image archives are not redistributed in this repository. Before redistributing any original source images or datasets, review the licences and terms attached to each upstream source. This project does not claim ownership of third-party source images.

---

## 19. Model artifact integrity

The active application model is V4. The V3 checkpoint is retained as an unchanged rollback backup.

| Item | Active V4 value |
|---|---|
| Application display name | CNN |
| Technical architecture | ImageNet-pretrained EfficientNetB0 backbone, fine-tuned from V3 |
| Active model path | `models_v4/car_cnn_model_v4.keras` |
| Input shape | 256 × 256 × 3 |
| Output | One sigmoid score; class 0 = real, class 1 = AI-generated |
| Decision threshold | 0.50 |
| Model file size | 34,358,590 bytes |
| SHA-256 | `36b6e3fe95278cb73923ec01e109671278fe97877e5bb7beae4b947fcd3f6d3e` |

V3 rollback backup: `models_v3/car_cnn_model_v3.keras`; SHA-256 `ea47dc1e9ded13944a2ac866ae968f3956a003d9faaf39ce4a0dfa8a8cad65b5`. The inference loader checks the active model SHA-256 before loading. If the weights are replaced or corrupted, the application stops with a checksum error rather than silently loading a different model.

To verify the active model locally (with the project environment activated), run:

```bash
python scripts/verify_active_cnn.py
```

To perform a prediction smoke test on a real photograph:

```bash
python scripts/verify_active_cnn.py /path/to/car_photo.jpg
```
