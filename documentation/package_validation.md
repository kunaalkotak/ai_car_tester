# Package validation record

Updated for the V4-active application package on 2026-11-01.

## Static and artifact checks

- Active application model: `models_v4/car_cnn_model_v4.keras`.
- Active V4 model size: 34,358,590 bytes.
- Active V4 SHA-256: `36b6e3fe95278cb73923ec01e109671278fe97877e5bb7beae4b947fcd3f6d3e` (matches both the uploaded training report and `config/cnn_inference_config.json`).
- Preserved V3 backup: `models_v3/car_cnn_model_v3.keras`; SHA-256: `ea47dc1e9ded13944a2ac866ae968f3956a003d9faaf39ce4a0dfa8a8cad65b5`.
- V3 backup hash remains unchanged from the V4 training report.
- Active application model display name: `CNN`; no model selector or classical-model inference path is present.
- Active decision threshold: `0.50`, selected on V4 validation balanced accuracy.
- Keras archive integrity check passed; saved configuration identifies a Functional model with input shape `(None, 256, 256, 3)`, an EfficientNetB0 backbone and one sigmoid output.
- V4 report confirms 1,304 expected files and 1,304 files loaded: 911 training, 196 validation and 197 test.
- Application Python modules and the active-model smoke-test script pass Python syntax compilation.
- README and documentation were updated to distinguish current V4 from historical V3 results.
- Two `.keras` artifacts are retained intentionally: one active V4 model and one V3 rollback backup. No classical model weights are included.
- No image datasets or original source ZIP archives are bundled.

## Runtime validation boundary

The packaging environment did not have TensorFlow installed. Consequently, the V4 model was not deserialized through TensorFlow and a live Streamlit prediction was not run here. The Keras ZIP archive, model configuration metadata, dimensions, SHA-256 and static Python checks were verified. Run `python scripts/verify_active_cnn.py` in the target Apple Silicon environment to perform the actual TensorFlow load and forward-pass smoke test, then launch Streamlit with `python -m streamlit run app_cnn.py`.

## Evaluation caveats

- V4 test results use 197 images (94 real, 103 AI-generated) and a threshold of 0.50 selected from validation.
- V3 was compared on the same test split using a threshold of 0.40 selected from the same validation split; V3's original threshold 0.47 is separately recorded in the report.
- V4's nine new-generator examples (GPT 4, Gemini 3, Meta AI 2) are exploratory and too small for reliable per-generator claims.
- The three named generator families are represented in the V4 development/training pool; this is not a held-out-generator evaluation.
- Generator source labels come from user-provided folder names and were not independently authenticated.
- The displayed sigmoid score remains uncalibrated and does not prove authenticity, vehicle existence or seller legitimacy.
