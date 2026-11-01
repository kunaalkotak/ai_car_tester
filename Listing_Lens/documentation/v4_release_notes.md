# Listing Lens V4 release notes

## Active application model

The Streamlit app now ships exactly one inference model: **CNN V4**, using an ImageNet-pretrained EfficientNetB0 backbone fine-tuned for the binary real/AI-generated task.

- Active artifact: `models_v4/car_cnn_model_v4.keras`
- SHA-256: `36b6e3fe95278cb73923ec01e109671278fe97877e5bb7beae4b947fcd3f6d3e`
- User-facing name: `CNN`
- Decision threshold: `0.50` (selected on V4 validation balanced accuracy)
- Input: RGB 256 × 256; float32, raw 0–255 range. Normalisation remains inside the saved model.

The V3 checkpoint was used as the starting point for fine-tuning V4, but V3 weights are intentionally not distributed in this final application repository. The historical V3 SHA-256 and comparative metrics are retained in the documentation. V3 weights are not required to run the application.

## V4 dataset and test results

V4 used all 1,304 available images: 911 train, 196 validation and 197 test. Counts were 437 real / 474 AI for training, 94 / 102 for validation, and 94 / 103 for test. All file counts matched the images loaded by the training pipeline.

At threshold 0.50, V4 test metrics were: accuracy 93.91%, balanced accuracy 93.85%, AI precision 93.33%, AI recall 95.15%, AI F1 94.23% and ROC-AUC 98.89%. The test confusion matrix (rows actual [real, AI], columns predicted [real, AI]) was `[[87, 7], [5, 98]]`.

On the same test images, the historical V3 baseline was evaluated at its validation-selected threshold 0.40: accuracy 93.40%, balanced accuracy 93.50%, AI precision 95.92%, AI recall 91.26%, AI F1 93.53% and ROC-AUC 97.98%. The comparison is modest: V4 improves recall and AUC, while V3 had higher AI precision and fewer real-image false positives. The V3 weights are not included in this release.

## New-generator subset and limitations

Nine new AI images in the V4 test split were detected: GPT 4/4, Gemini 3/3 and Meta AI 2/2. This is exploratory only. The sample sizes are too small for reliable per-generator claims, source labels were supplied through folder names rather than independently authenticated, and all three generator families were represented in the V4 development/training pool. This is not a held-out-generator test.

The sigmoid output is an uncalibrated visual-pattern score. It does not establish photo provenance, vehicle authenticity or seller legitimacy.
