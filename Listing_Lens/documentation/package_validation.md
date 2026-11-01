# Package validation record

Updated for the single-model V4 GitHub release.

## Static and artifact checks

- Only one model weight file is distributed: `models_v4/car_cnn_model_v4.keras`.
- Active V4 model size: 34,358,590 bytes.
- Active V4 SHA-256: `36b6e3fe95278cb73923ec01e109671278fe97877e5bb7beae4b947fcd3f6d3e`.
- Application display name: `CNN`; no model selector or classical-model inference path is present.
- Active decision threshold: `0.50`, selected on V4 validation balanced accuracy.
- Keras archive integrity check passed; saved configuration identifies a Functional model with input `(None, 256, 256, 3)`, EfficientNetB0 backbone and one sigmoid output.
- V4 report confirms 1,304 expected files and 1,304 files loaded: 911 training, 196 validation and 197 test.
- V3's historical starting checkpoint SHA-256 is recorded as `ea47dc1e9ded13944a2ac866ae968f3956a003d9faaf39ce4a0dfa8a8cad65b5`; its weights are not included because the application does not need them.
- Application Python modules and the active-model smoke-test script pass Python syntax compilation.
- The README, notebooks and documentation distinguish historical V3 development from the active V4 release.
- No raw image datasets, source ZIP archives, virtual environments, training outputs or challenge photographs are bundled.

## Runtime validation boundary

The packaging environment did not have TensorFlow installed. Consequently, a live TensorFlow model deserialization and Streamlit prediction were not run during packaging. The Keras archive, model configuration metadata, dimensions, SHA-256 and static Python checks were verified. Run `python scripts/verify_active_cnn.py` in the target Apple Silicon environment to perform the TensorFlow load and forward-pass smoke test, then launch Streamlit with `python -m streamlit run app_cnn.py`.

## Evaluation caveats

- V4 test results use 197 images (94 real, 103 AI-generated) and threshold 0.50 selected from validation.
- V3 was compared on the same test split using a threshold of 0.40 selected from the same validation split; ROC-AUC is threshold-independent. The V3 weights are not distributed in this release.
- V4's nine new-generator examples (GPT 4, Gemini 3, Meta AI 2) are exploratory and too small for reliable per-generator claims.
- The three named generator families are represented in the V4 development/training pool; this is not a held-out-generator evaluation.
- Generator source labels come from user-provided folder names and were not independently authenticated.
- The displayed sigmoid score remains uncalibrated and does not prove authenticity, vehicle existence or seller legitimacy.
