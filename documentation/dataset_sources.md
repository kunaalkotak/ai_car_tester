# Dataset sources and provenance

Dataset V2 is recorded as a balanced set of 1,250 candidates: 625 real and 625 AI-generated. The three source archives named in the project records were:

1. `real_car_dataset_v2_deduplicated.zip` — 625 real-image candidates from the existing project real-car collection, including a contribution from the Hugging Face dataset `Smily6820/car-images`.
2. `Car_AI_Images_FGTS_Additions.zip` — 474 additional AI-generated candidates. The recovered FGTS `collection_summary.json` records 158 LDM, 158 Stable Diffusion 1.4 and 158 ProGAN candidates (474 in total; seed 42). It records zero skipped invalid/exact-duplicate candidates at that extraction stage; it does not replace the later merged-dataset audit.
3. `Car_Image_Authenticity_Dataset_v1.zip` — 151 AI-generated images retained from the earlier V1 dataset. The exact per-generator breakdown of these 151 images was not recovered.

The real-image contribution from Hugging Face was part of the existing project data. A recovered script `notebooks/source_material/dataset_collection/Finalize_Selected_51_HF_Car_Photos.py` records a 51-image selection from `Smily6820/car-images`, the repository's declared CC BY 4.0 licence, and an upstream note that the repository describes the dataset as a duplicate of `KAI-KratosAI/car-images`. The script notes that upstream per-photo rights were not independently established. The 51 files should not be interpreted as the entire Hugging Face contribution.

A separate archived car-image generation notebook documents an earlier 100-image pilot plan: 34 Stable Diffusion v1.5, 33 SDXL Turbo, and 33 Segmind Tiny-SD. That is retained as historical source material, not counted as the 474 FGTS additions or the 151 V1 images without matching per-image evidence.

Combining sources was intended to improve variety in vehicles, image composition, backgrounds and generator styles. This does not guarantee generalisation to unseen data. Labels/generator families were inherited from source folders/manifests; real-image authenticity was not independently verified image by image.

The source archives and all 1,250 images are intentionally excluded from this code-and-documentation ZIP. They must be provided separately to rerun dataset-level notebooks.
