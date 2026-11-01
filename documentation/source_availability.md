# Source Availability and Reconstruction Notes

This file separates recovered originals from newly written academic guide notebooks.

## Recovered original project materials

- `notebooks/source_material/early_198_image_pilot/Prepare_FOML_Dataset.ipynb`
- `notebooks/source_material/early_198_image_pilot/Verify_FOML_Dataset.ipynb`
- `notebooks/source_material/early_198_image_pilot/Split_FOML_Dataset.ipynb`
- `notebooks/source_material/early_198_image_pilot/Classical_ML_PCA_LDA_Baseline.ipynb`
- `notebooks/source_material/early_198_image_pilot/CNN_Baseline_FOML.ipynb`
- `notebooks/source_material/early_198_image_pilot/Evaluate_FOML_Test_Set.ipynb`
- `notebooks/source_material/ai_image_generation/Generate_100_AI_Car_Images_MultiModel_FIXED.ipynb`
- `notebooks/source_material/dataset_collection/Finalize_Selected_51_HF_Car_Photos.py`
- `documentation/source_evidence/fgts_collection_summary.json`

The six early-pilot notebooks document a 198-image general-object experiment. The split notebook uses random seed 42 and expects 138/30/30 train/validation/test images. The recovered `foml_final_test_report` is a separate 30-image test report. Additional preserved pilot artifacts include the CNN training history, validation metrics, training curves, CNN validation confusion matrix, and classical-model validation metrics. These are retained as historical source materials and are explicitly not labelled as the later 1,250-image car Dataset V2 notebooks. The old pilot CNN weights and classical model bundle were not included.

The separate car-image generation notebook describes an earlier 100-image generation plan (SD 1.5 / SDXL Turbo / Tiny-SD, 34/33/33). It is retained as background evidence, not counted in the FGTS 474 additions without the matching per-image source manifest.

## Reconstructed guide notebooks

The seven notebooks at the top of `notebooks/` are documented guide/reconstruction notebooks. They were created because the exact V2 assembly/audit/split notebooks and exact V3 training script were not present in the uploaded `Listing_Lens_Clean_Source.zip` or in the relevant recovered project files available for this revision.

- Notebook 01 uses user-provided Dataset V2 counts and the recovered FGTS collection summary.
- Notebook 02 documents the reported audit methods and 183-image manual-review queue. Exact historic thresholds, car-detector model and full audit manifest were not recovered.
- Notebook 03 records the V2 70:15:15 split and seed 211. The original per-image split manifest was not recovered.
- Notebook 04 records the 46-image V1 model results supplied in project notes and contrasts them with the recovered 30-image early-pilot report.
- Notebook 05 records the V2 model metrics supplied in project notes. Per-image predictions and complete classical metrics were not recovered.
- Notebook 06 reconstructs the V3 architecture/training record from the serialized `.keras` configuration and `config/cnn_inference_config.json`. The original `train_improved.py` (or equivalent source) was not found.
- Notebook 07 records V3 metrics. The test confusion matrix is reconstructed from the saved metric fractions. The original per-image V3 test scores were not recovered, so a ROC curve is not fabricated; code is included to create it when scores are available.

## Reproducibility boundary

The notebooks explain the workflow and contain reusable validation code, but dataset assembly, the full audit, V2 split assignment, and V3 retraining cannot be fully reproduced from this ZIP alone because the source archives, 1,250 images, per-image manifests, original V3 training script and prediction exports are deliberately absent or were not recovered. The final app remains runnable with the packaged model and application dependencies only.
