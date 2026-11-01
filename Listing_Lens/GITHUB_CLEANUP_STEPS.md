# Applying the single-model V4 GitHub cleanup

This release keeps only the V4 model weight in the current GitHub tree. V3's metrics and source hash remain documented; the V3 weight file is removed because the app does not use it.

## 1. Overlay the updated files

Extract the supplied patch into the root of your existing `Listing_Lens` Git working folder, allowing files to be replaced. Do not delete the `.git` directory.

## 2. Remove old tracked model and local-work folders from the Git index

Run from the project root:

```bash
git rm -r --ignore-unmatch models_v3
# Remove obsolete challenge collection tooling if it was staged/tracked:
git rm -r --cached --ignore-unmatch challenge_eval data_v4 data_v4_full outputs_v4 outputs_v4_full v4_training
```

The `--cached` command removes tracked versions from Git but leaves your local data/working folders on disk. `models_v3/` is removed from the working tree too. If you want to keep the V3 starting weights for possible future retraining, copy that file outside the repository before the first command.

## 3. Check the active model is present

```bash
shasum -a 256 models_v4/car_cnn_model_v4.keras
```

Expected SHA-256:

```text
36b6e3fe95278cb73923ec01e109671278fe97877e5bb7beae4b947fcd3f6d3e
```

## 4. Review exactly what will be committed

```bash
git add -A
git status --short
git diff --cached --stat
```

Confirm `models_v4/car_cnn_model_v4.keras` is included, `models_v3/` is deleted, notebooks 01–09 are present, and no `.venv`, dataset images or training output folders are staged.

## 5. Commit and push

```bash
git commit -m "Clean repository and document final Listing Lens V4"
git push origin main
```

The deletion commit removes V3 from the current branch tree. The previous Git commit still contains that historical blob. A history rewrite is only needed if the goal is to remove it from all prior commits and reduce historical clone size; that requires coordinated `git filter-repo` and force-push and is not necessary for normal cleanup.
