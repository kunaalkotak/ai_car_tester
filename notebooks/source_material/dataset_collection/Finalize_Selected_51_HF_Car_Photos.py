# Paste this entire script into the FINALIZATION CELL of the existing
# Select_51_HuggingFace_Real_Car_Photos_AUTHFIX.ipynb notebook.
# It uses the existing Colab runtime variables created by the earlier cells.

SELECTED_HF_IDS = [
    # Contact sheet 1
    "HF00116", "HF00386", "HF00514", "HF00388", "HF00172",
    "HF00233", "HF00454", "HF00166", "HF00179", "HF00100",
    "HF00224", "HF00471", "HF00184",
    # Contact sheet 2
    "HF00477", "HF00252", "HF00375", "HF00099", "HF00352",
    "HF00475", "HF00181", "HF00214", "HF00345", "HF00035",
    "HF00505", "HF00333", "HF00193",
    # Contact sheet 3
    "HF00276", "HF00199", "HF00244", "HF00283", "HF00157",
    "HF00216", "HF00101", "HF00041", "HF00083", "HF00106",
    "HF00128", "HF00362", "HF00204",
    # Contact sheet 4
    "HF00396", "HF00334", "HF00404", "HF00536", "HF00370",
    "HF00376", "HF00201", "HF00026", "HF00018", "HF00351",
    "HF00131", "HF00438",
]

# Confirm that the earlier notebook cells completed.
required_names = ["pool", "source_path_by_id", "WORK", "REPO_URL", "REPO_ID", "LICENSE_NAME", "make_contact_sheet"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise RuntimeError(
        "The earlier notebook cells are not loaded: " + ", ".join(missing_names) +
        ". Run the existing notebook through the contact-sheet cell first, then run this cell."
    )

if len(SELECTED_HF_IDS) != 51 or len(set(SELECTED_HF_IDS)) != 51:
    raise RuntimeError("Selection must contain exactly 51 unique candidate IDs.")

pool_ids = set(pool["candidate_id"].astype(str))
missing_ids = sorted(set(SELECTED_HF_IDS) - pool_ids)
if missing_ids:
    raise RuntimeError("These selected IDs are not in the current review pool: " + ", ".join(missing_ids))

selected = pool.set_index("candidate_id").loc[SELECTED_HF_IDS].reset_index().copy()

# These 51 were selected from the contact sheets for visible car presence,
# reasonable framing and diversity of vehicles/views/locations. Please inspect
# the output contact sheet at full size before considering the set final.
SELECT_DIR = WORK / "selected_real_car_images"
SELECT_DIR.mkdir(parents=True, exist_ok=True)
for old_file in SELECT_DIR.iterdir():
    if old_file.is_file():
        old_file.unlink()
    elif old_file.is_dir():
        shutil.rmtree(old_file)

selected_rows = []
for i, (_, row) in enumerate(tqdm(selected.iterrows(), total=len(selected), desc="Preparing selected real-car images"), start=1):
    candidate_id = str(row["candidate_id"])
    src = source_path_by_id.get(candidate_id)
    if src is None or not Path(src).exists():
        raise FileNotFoundError(f"Original source file is missing for {candidate_id}: {src}")

    out_name = f"REAL_HF_{i:03d}.jpg"
    out_path = SELECT_DIR / out_name
    with Image.open(src) as raw:
        im = ImageOps.exif_transpose(raw).convert("RGB")
        final_size = im.size
        im.save(out_path, "JPEG", quality=95, optimize=True)

    record = row.to_dict()
    record.update({
        "final_image_name": out_name,
        "label": "real",
        "source_uploader": "Smily6820",
        "license_declared_by_repository": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_dataset_url": REPO_URL,
        "upstream_note": "Dataset card states this collection is a duplicate of KAI-KratosAI/car-images.",
        "transformations": "EXIF orientation applied; converted to RGB JPEG quality 95; no resize or crop.",
        "selection_note": "Selected after contact-sheet review; inspect full-resolution image before final training use.",
        "final_width": final_size[0],
        "final_height": final_size[1],
        "selected_on_utc": datetime.now(timezone.utc).isoformat(),
    })
    selected_rows.append(record)

metadata = pd.DataFrame(selected_rows)
metadata.to_csv(SELECT_DIR / "metadata.csv", index=False)

# A record of the fixed selection makes it reproducible.
pd.DataFrame({"candidate_id": SELECTED_HF_IDS}).to_csv(SELECT_DIR / "selected_candidate_ids.csv", index=False)

attribution = f"""# Attribution and use notes

This supplementary real-car image subset was selected from **{REPO_ID}**.

- Dataset page: {REPO_URL}
- Repository uploader: Smily6820
- Declared repository licence: CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/
- Upstream note: the repository dataset card states it is a duplicate of `KAI-KratosAI/car-images`.
- Access date (UTC): {datetime.now(timezone.utc).date()}

The repository declares CC BY 4.0; however, the dataset card does not independently establish the original rights holder or source URL for each underlying photograph. Preserve this attribution and verify any upstream rights as required. The selected files were orientation-corrected and converted to RGB JPEG quality 95 without resizing or cropping.

These images are candidate real-car photographs selected by inspecting contact-sheet thumbnails. The source metadata alone does not prove camera authenticity. Inspect the final contact sheet/full-resolution images and remove any unsuitable or questionable examples before training. For public redistribution, confirm the applicable licence and attribution obligations.
"""
(SELECT_DIR / "ATTRIBUTION_AND_NOTES.md").write_text(attribution, encoding="utf-8")

selected_sheet = WORK / "selected_real_car_contact_sheet.jpg"
make_contact_sheet(selected, 0, len(selected), selected_sheet, thumb_size=(180, 125), cols=5)

final_zip = Path("/content/real_car_supplement_51.zip")
with zipfile.ZipFile(final_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for file_path in sorted(SELECT_DIR.iterdir()):
        if file_path.is_file():
            zf.write(file_path, arcname=f"real_car_supplement_51/{file_path.name}")
    zf.write(selected_sheet, arcname="real_car_supplement_51/review/selected_contact_sheet.jpg")

print("Selected image count:", len(selected))
print("All selected IDs were found in the current review pool.")
print("ZIP created:", final_zip)
print("IMPORTANT: inspect selected_contact_sheet.jpg and spot-check full-resolution images before final training use.")
try:
    from google.colab import files
    files.download(str(final_zip))
except Exception as exc:
    print("Automatic download skipped; file remains at", final_zip, "Reason:", exc)

display(Image.open(selected_sheet))
