# Package validation record

Generated for the academic GitHub-ready revision on 2026-11-01.

## Static checks

- Original selected model retained without retraining or conversion.
- Model SHA-256: `ea47dc1e9ded13944a2ac866ae968f3956a003d9faaf39ce4a0dfa8a8cad65b5` (matches `config/cnn_inference_config.json`).
- Application model display name: `CNN`.
- Decision threshold: `0.47`.
- Active application code contains no model selector and no classical-model loading/inference path.
- Seven main guide notebooks and seven recovered original notebook/source files are present (14 `.ipynb` files total).
- All notebooks pass notebook-format validation.
- Python code cells in the seven main guide notebooks pass Python syntax parsing.
- Application Python modules pass `compileall`.
- README local notebook/document links resolve.
- Exactly one `.keras` model artifact is present; no source dataset ZIPs or image datasets are bundled.
- Final project archive passes ZIP CRC verification.

## Runtime validation boundary

This packaging environment did not have TensorFlow and Streamlit available for a live app launch or model prediction. Consequently, this record does **not** claim that the Keras model was deserialized in this environment or that a live Streamlit prediction was completed. After extracting the project, install `requirements.txt` and follow the README's runtime verification steps on the target machine.

The academic notebooks are documented guides/reconstructions where the original V2/V3 scripts, source images or per-image manifests were not recovered. They are structurally validated, but data-dependent cells cannot reproduce the historical dataset run without those external inputs.
