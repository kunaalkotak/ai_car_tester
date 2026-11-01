# Preserved source notebooks, scripts and reports

This folder preserves files recovered from earlier project work. Not every file belongs to the final 1,250-image car Dataset V2. The subfolder names and the README descriptions are intentional: they prevent distinct experiments from being presented as one continuous dataset run.

## `early_198_image_pilot/` — original source notebooks and pilot artifacts

The six original notebooks here are an earlier FOML general-object pilot, not the later car-image Dataset V2. They work with a 198-image dataset, a `random_state=42` split and a 30-image test report. They are retained verbatim. The folder also preserves the pilot's validation CSVs, training history, validation/training plots, test predictions and confusion-matrix figure. These are useful academic evidence, but their numbers must not be confused with the 46-image V1 baseline table or Dataset V2/V3 results in the main notebooks.

The old `cnn_model.keras` and classical model bundle were deliberately not copied: historical code and evaluation evidence are preserved, but duplicate/obsolete model weights are not.

## `ai_image_generation/`

`Generate_100_AI_Car_Images_MultiModel_FIXED.ipynb` documents a separate earlier car AI-image-generation pilot. It planned 100 outputs across Stable Diffusion v1.5, SDXL Turbo and Tiny-SD (34/33/33). Do not treat this plan as proof that those 100 outputs were part of the 474 FGTS additions or the V1 carry-over.

## `dataset_collection/`

`Finalize_Selected_51_HF_Car_Photos.py` documents a fixed 51-image selection workflow from the Hugging Face `Smily6820/car-images` repository. It is supporting evidence, not a manifest for all 625 real-image candidates. `cnn_threshold_analysis.py` is a legacy threshold-analysis utility tied to an older `car_model_utils` module that is not present in the final app; it is retained as historical source code, not as an active inference dependency.

The source archives, all 1,250 images, virtual environments and obsolete model weight files are intentionally excluded.
