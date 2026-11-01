#!/usr/bin/env python3
"""Evaluate Listing Lens V3 and V4 on a frozen, external challenge set.

Expected folders:
  data/real/*
  data/ai_generated/GPT/*
  data/ai_generated/Gemini/*
  data/ai_generated/Meta_AI/*

Uses the same inference decoding/resize path as Listing Lens V3:
- RGB, 256x256, float32, pixel range 0..255
- TensorFlow decode_image + bilinear resize for JPEG/PNG/BMP/GIF
- Pillow first-frame decode + bilinear resize for WebP
- No external /255 normalization (normalization lives inside the model)

Thresholds are loaded from the V4 training report (validation-selected) and
V3's validation-selected comparison threshold. The external challenge set is
never used to choose or tune thresholds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
IMAGE_SIZE = (256, 256)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def list_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    )


def load_optional_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel = (row.get("relative_path") or "").strip().replace("\\", "/")
            if not rel:
                continue
            if rel in rows:
                raise ValueError(f"Duplicate relative_path in manifest.csv: {rel}")
            rows[rel] = {k: (v or "").strip() for k, v in row.items() if k}
    return rows


def infer_metadata(path: Path, root: Path, manifest: dict[str, dict[str, str]]) -> dict[str, str | int]:
    rel = path.relative_to(root).as_posix()
    parts = Path(rel).parts
    if len(parts) < 2:
        raise ValueError(f"Image is not inside a labelled folder: {rel}")
    top = parts[0].lower()
    if top == "real":
        label = "real"
        generator = "real_photograph"
    elif top == "ai_generated":
        label = "ai_generated"
        generator = parts[1] if len(parts) >= 3 else "unspecified_ai_source"
    else:
        raise ValueError(
            f"Unexpected folder for {rel}. Put genuine photos in data/real/ and AI images in data/ai_generated/<generator>/"
        )
    extra = manifest.get(rel, {})
    supplied_label = extra.get("label", "").lower()
    if supplied_label and supplied_label != label:
        raise ValueError(f"Manifest label mismatch for {rel}: folder says {label}, manifest says {supplied_label}")
    if extra.get("generator"):
        generator = extra["generator"]
    return {
        "relative_path": rel,
        "label": label,
        "y_true": 0 if label == "real" else 1,
        "generator": generator,
        "source_tool": extra.get("source_tool", generator),
        "prompt_or_session_id": extra.get("prompt_or_session_id", ""),
        "group_id": extra.get("group_id", ""),
        "collected_on": extra.get("collected_on", ""),
        "notes": extra.get("notes", ""),
    }


def prepare_image(path: Path, tf) -> np.ndarray:
    """Match the production inference preprocessing as closely as possible."""
    encoded = path.read_bytes()
    if not encoded:
        raise ValueError("empty file")
    ext = path.suffix.lower()
    if ext == ".webp":
        with Image.open(path) as image:
            image.seek(0)
            rgb = image.convert("RGB")
            resized = rgb.resize((256, 256), resample=Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32)
    else:
        decoded = tf.io.decode_image(
            tf.convert_to_tensor(encoded, dtype=tf.string),
            channels=3,
            expand_animations=False,
        )
        resized = tf.image.resize(decoded, IMAGE_SIZE, method="bilinear")
        arr = np.asarray(resized, dtype=np.float32)
    if arr.shape != (256, 256, 3) or not np.isfinite(arr).all():
        raise ValueError(f"unexpected decoded array shape/values: {arr.shape}")
    return arr


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )
    y_pred = (scores >= float(threshold)).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "ai_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "ai_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "ai_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "real_false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "real_specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix_actual_rows_predicted_columns": cm.tolist(),
        "counts": {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp},
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt((p * (1 - p) / total) + (z * z / (4 * total * total))) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def load_verified_model(tf, model_path: Path, expected_hash: str, label: str):
    if not model_path.is_file():
        raise FileNotFoundError(f"{label} model not found: {model_path}")
    actual = sha256_file(model_path)
    if expected_hash and actual != expected_hash.lower():
        raise ValueError(f"{label} SHA-256 mismatch. Expected {expected_hash}, found {actual}. Refusing to load.")
    model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(model.input_shape[-3:]) != (256, 256, 3) or model.output_shape[-1] != 1:
        raise ValueError(f"Unexpected {label} model input/output shape: {model.input_shape} -> {model.output_shape}")
    return model, actual


def scan_known_hashes(root: Path | None) -> dict[str, str]:
    known: dict[str, str] = {}
    if root is None or not root.is_dir():
        return known
    for path in list_images(root):
        try:
            known.setdefault(sha256_file(path), str(path.relative_to(root)))
        except OSError:
            continue
    return known


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["relative_path", "reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Listing Lens V3 and V4 on an independent challenge set.")
    parser.add_argument("--challenge-root", type=Path, default=Path("challenge_eval/data"))
    parser.add_argument("--known-data-root", type=Path, default=Path("data_v4_full"), help="Dataset V4 root used for exact-hash overlap exclusion; set to '' to disable.")
    parser.add_argument("--v3-model", type=Path, default=Path("models_v3/car_cnn_model_v3.keras"))
    parser.add_argument("--v3-config", type=Path, default=Path("config/cnn_inference_config.json"))
    parser.add_argument("--v4-model", type=Path, default=Path("outputs_v4_full/car_cnn_model_v4_candidate.keras"))
    parser.add_argument("--v4-report", type=Path, default=Path("outputs_v4_full/v4_training_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("challenge_eval/results"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--v3-threshold", type=float, default=None, help="Defaults to V3 validation-selected threshold in V4 report.")
    parser.add_argument("--v4-threshold", type=float, default=None, help="Defaults to V4 validation-selected threshold in V4 report.")
    args = parser.parse_args()

    challenge_root = args.challenge_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not challenge_root.is_dir():
        raise SystemExit(f"Challenge folder not found: {challenge_root}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    v3_config_path = args.v3_config.expanduser().resolve()
    v4_report_path = args.v4_report.expanduser().resolve()
    if not v3_config_path.is_file():
        raise SystemExit(f"V3 config not found: {v3_config_path}")
    if not v4_report_path.is_file():
        raise SystemExit(f"V4 training report not found: {v4_report_path}")
    v3_config = json.loads(v3_config_path.read_text(encoding="utf-8"))
    v4_report = json.loads(v4_report_path.read_text(encoding="utf-8"))

    v3_expected_hash = str(v3_config.get("model_sha256", "")).lower()
    v4_expected_hash = str(v4_report.get("v4_candidate_sha256", "")).lower()
    if not v3_expected_hash or not v4_expected_hash:
        raise SystemExit("Could not find expected model SHA-256 values in V3 config / V4 report.")

    v3_threshold = args.v3_threshold
    if v3_threshold is None:
        v3_threshold = float(v4_report["v3_baseline_on_same_v4_validation_and_test_splits"]["validation_selected_threshold"])
    v4_threshold = args.v4_threshold
    if v4_threshold is None:
        v4_threshold = float(v4_report["selected_threshold_from_validation_balanced_accuracy"])
    if not 0 < v3_threshold < 1 or not 0 < v4_threshold < 1:
        raise SystemExit("Thresholds must be strictly between 0 and 1.")

    # Verify that all challenge files are valid, labelled, unique, and distinct
    # from the known V4 dataset before importing TensorFlow or running inference.
    all_paths = list_images(challenge_root)
    if not all_paths:
        raise SystemExit(f"No images found under {challenge_root}. Add real/ and ai_generated/<generator>/ images first.")
    manifest_path = challenge_root / "manifest.csv"
    manifest = load_optional_manifest(manifest_path)
    known_root = args.known_data_root.expanduser().resolve() if str(args.known_data_root) else None
    known_hashes = scan_known_hashes(known_root)
    seen_challenge_hashes: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for path in all_paths:
        try:
            meta = infer_metadata(path, challenge_root, manifest)
            digest = sha256_file(path)
            if digest in known_hashes:
                excluded.append({"relative_path": meta["relative_path"], "reason": "exact duplicate of known V4 dataset image", "matching_known_path": known_hashes[digest]})
                continue
            if digest in seen_challenge_hashes:
                excluded.append({"relative_path": meta["relative_path"], "reason": "exact duplicate inside challenge set", "matching_challenge_path": seen_challenge_hashes[digest]})
                continue
            with Image.open(path) as image:
                image.verify()
            # Reopen to capture dimensions after verification.
            with Image.open(path) as image:
                image.seek(0)
                width, height = image.size
            meta["sha256"] = digest
            meta["width"] = int(width)
            meta["height"] = int(height)
            meta["absolute_path"] = str(path)
            records.append(meta)
            seen_challenge_hashes[digest] = str(path)
        except Exception as exc:
            excluded.append({"relative_path": str(path.relative_to(challenge_root)), "reason": f"invalid_or_unlabelled: {exc}"})

    write_csv(output_dir / "excluded_files.csv", excluded)
    if len(records) < 2 or len({int(r["y_true"]) for r in records}) < 2:
        raise SystemExit(f"Need at least one readable image of each class. Accepted={len(records)}; exclusions={len(excluded)}. See {output_dir / 'excluded_files.csv'}")
    # A manifest is a convenience for provenance. Warn if it includes paths that
    # aren't in the folders, so typos don't silently disappear.
    actual_rel = {str(r["relative_path"]) for r in records}
    manifest_orphans = sorted(set(manifest) - {p.relative_to(challenge_root).as_posix() for p in all_paths})

    try:
        import tensorflow as tf
        from sklearn.metrics import roc_auc_score  # noqa: F401
    except Exception as exc:
        raise SystemExit(f"Evaluation requires TensorFlow and scikit-learn in the active project environment: {exc}") from exc

    v3_path = args.v3_model.expanduser().resolve()
    v4_path = args.v4_model.expanduser().resolve()
    v3_model, v3_hash = load_verified_model(tf, v3_path, v3_expected_hash, "V3")
    v4_model, v4_hash = load_verified_model(tf, v4_path, v4_expected_hash, "V4")

    y_true = np.asarray([int(r["y_true"]) for r in records], dtype=int)
    v3_scores: list[float] = []
    v4_scores: list[float] = []
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start:start + args.batch_size]
        batches = np.stack([prepare_image(Path(r["absolute_path"]), tf) for r in batch_records]).astype(np.float32)
        s3 = np.asarray(v3_model.predict(batches, verbose=0)).reshape(-1)
        s4 = np.asarray(v4_model.predict(batches, verbose=0)).reshape(-1)
        if len(s3) != len(batch_records) or len(s4) != len(batch_records):
            raise RuntimeError("Prediction count mismatch for one of the models.")
        if not np.isfinite(s3).all() or not np.isfinite(s4).all():
            raise RuntimeError("A model produced non-finite prediction scores.")
        v3_scores.extend(np.clip(s3, 0, 1).astype(float).tolist())
        v4_scores.extend(np.clip(s4, 0, 1).astype(float).tolist())

    s3 = np.asarray(v3_scores, dtype=float)
    s4 = np.asarray(v4_scores, dtype=float)
    v3_result = metrics(y_true, s3, v3_threshold)
    v4_result = metrics(y_true, s4, v4_threshold)
    # Preserve the previously deployed V3 threshold as a secondary reference.
    v3_original_threshold = float(v3_config.get("decision_threshold", 0.47))
    v3_original_result = metrics(y_true, s3, v3_original_threshold)

    predictions: list[dict[str, Any]] = []
    by_generator: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        p3 = int(s3[i] >= v3_threshold)
        p4 = int(s4[i] >= v4_threshold)
        predictions.append({
            "relative_path": record["relative_path"],
            "label": record["label"],
            "generator": record["generator"],
            "source_tool": record["source_tool"],
            "prompt_or_session_id": record["prompt_or_session_id"],
            "group_id": record["group_id"],
            "collected_on": record["collected_on"],
            "sha256": record["sha256"],
            "width": record["width"], "height": record["height"],
            "v3_ai_score": float(s3[i]), "v3_prediction": "ai_generated" if p3 else "real",
            "v4_ai_score": float(s4[i]), "v4_prediction": "ai_generated" if p4 else "real",
            "v4_correct": int(p4 == int(y_true[i])),
            "v3_correct": int(p3 == int(y_true[i])),
        })
        if int(y_true[i]) == 1:
            by_generator[str(record["generator"])].append(i)

    per_generator: dict[str, Any] = {}
    for generator, indices in sorted(by_generator.items()):
        detected3 = int(np.sum(s3[indices] >= v3_threshold))
        detected4 = int(np.sum(s4[indices] >= v4_threshold))
        n = len(indices)
        per_generator[generator] = {
            "n_ai_images": n,
            "v3_detected": detected3,
            "v3_detection_rate": detected3 / n,
            "v3_detection_rate_wilson_95_ci": wilson_interval(detected3, n),
            "v4_detected": detected4,
            "v4_detection_rate": detected4 / n,
            "v4_detection_rate_wilson_95_ci": wilson_interval(detected4, n),
            "v3_mean_ai_score": float(np.mean(s3[indices])),
            "v4_mean_ai_score": float(np.mean(s4[indices])),
        }

    real_n = int(np.sum(y_true == 0))
    warnings: list[str] = []
    if real_n < 100:
        warnings.append(f"Only {real_n} genuine-photo challenge images; target at least 100.")
    for generator, info in per_generator.items():
        if info["n_ai_images"] < 50:
            warnings.append(f"Only {info['n_ai_images']} AI images for {generator}; target at least 50 per generator.")
    if not known_hashes:
        warnings.append("Known Dataset V4 root was not scanned (path missing or disabled); external-set overlap was not checked against training/validation/test images.")
    if manifest_orphans:
        warnings.append(f"Manifest has {len(manifest_orphans)} paths that do not exist in the challenge folders.")
    if excluded:
        warnings.append(f"{len(excluded)} files were excluded as invalid, unlabelled, or exact duplicates. Review excluded_files.csv.")
    warnings.append("This is an external same-generator-family challenge set, not a fully unseen-generator test, because GPT/Gemini/Meta families were represented in the V4 development data.")
    warnings.append("Do not tune thresholds using this challenge result. Thresholds are frozen from the V4 development validation set.")

    summary = {
        "project": "Listing Lens external challenge evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "challenge_root": str(challenge_root),
        "known_data_root": str(known_root) if known_root else None,
        "n_accepted": len(records),
        "n_real": real_n,
        "n_ai_generated": int(np.sum(y_true == 1)),
        "excluded_files": len(excluded),
        "preprocessing": {
            "input_shape": [256, 256, 3], "color": "RGB", "dtype": "float32",
            "range": "0..255", "resize": "bilinear direct resize", "external_divide_by_255": False,
            "note": "Matches the Listing Lens inference pipeline; EfficientNetB0 normalization is embedded in each model."
        },
        "models": {
            "V3": {"path": str(v3_path), "sha256": v3_hash, "threshold_validation_selected": v3_threshold,
                   "metrics_at_validation_selected_threshold": v3_result,
                   "metrics_at_original_app_threshold": {"threshold": v3_original_threshold, **v3_original_result}},
            "V4": {"path": str(v4_path), "sha256": v4_hash, "threshold_validation_selected": v4_threshold,
                   "metrics_at_validation_selected_threshold": v4_result},
        },
        "per_ai_generator": per_generator,
        "warnings_and_interpretation": warnings,
        "manifest_orphan_paths": manifest_orphans,
    }
    write_csv(output_dir / "predictions.csv", predictions)
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "accepted_images": len(records), "real": real_n, "ai_generated": int(np.sum(y_true == 1)),
        "V3_metrics": v3_result, "V4_metrics": v4_result,
        "per_ai_generator": per_generator, "warnings": warnings,
        "outputs": str(output_dir),
    }, indent=2))
    print(f"\nSaved: {output_dir / 'evaluation_summary.json'}")
    print(f"Saved: {output_dir / 'predictions.csv'}")
    print(f"Saved: {output_dir / 'excluded_files.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
