# V4 application promotion notes

## Active model

The Streamlit application now loads **CNN V4**, a fine-tuned EfficientNetB0 model initialized from the preserved V3 checkpoint. The UI continues to display the model simply as **CNN**.

- Active model: `models_v4/car_cnn_model_v4.keras`
- SHA-256: `36b6e3fe95278cb73923ec01e109671278fe97877e5bb7beae4b947fcd3f6d3e`
- Decision threshold: `0.50` (selected on the V4 validation set by balanced accuracy)
- Input: RGB 256 × 256, float32, raw 0–255 range; normalization remains inside the model.
- V3 backup: `models_v3/car_cnn_model_v3.keras` (unchanged; original SHA-256 `ea47dc1e9ded13944a2ac866ae968f3956a003d9faaf39ce4a0dfa8a8cad65b5`)

## V4 dataset and metrics

V4 training used all 1,304 available images: 911 train, 196 validation and 197 test. The class counts were 437/474 for train, 94/102 for validation and 94/103 for test (real/AI). All file counts matched the images loaded by the training pipeline.

At threshold 0.50, V4 test metrics were: accuracy 93.91%, balanced accuracy 93.85%, AI precision 93.33%, AI recall 95.15%, AI F1 94.23% and ROC-AUC 98.89%. The actual test confusion matrix (rows actual [real, AI], columns predicted [real, AI]) was `[[87, 7], [5, 98]]`.

On the same test images, V3 was evaluated at its validation-selected threshold 0.40: accuracy 93.40%, balanced accuracy 93.50%, AI precision 95.92%, AI recall 91.26%, AI F1 93.53% and ROC-AUC 97.98%. V4 has a modest advantage on accuracy, balanced accuracy, recall, F1 and AUC; V3 has higher AI precision and fewer real-image false positives (4 versus 7).

## New-generator subset and limitations

Nine new AI images in the V4 test split were detected: GPT 4/4, Gemini 3/3 and Meta AI 2/2. This is exploratory only. The sample sizes are too small for reliable per-generator claims, source labels were supplied through folder names rather than independently authenticated, and all three generator families were represented in the V4 training/development pool. This is **not** a held-out-generator test.

The V4 model is now configured as the active application model at the user's request. Keep the V3 checkpoint as a rollback copy. The model output remains an uncalibrated visual-pattern score and does not establish photo provenance, vehicle authenticity or seller legitimacy.
