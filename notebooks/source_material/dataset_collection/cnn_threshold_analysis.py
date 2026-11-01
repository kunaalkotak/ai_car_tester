#!/usr/bin/env python3
"""Evaluate CNN decision thresholds on the validation split only.

This script does not retrain or modify the model. It saves per-image sigmoid
scores and a threshold sweep so the team can examine the precision/recall/FPR
trade-off without looking at the held-out test set.

Example:
  python scripts/cnn_threshold_analysis.py \
    --dataset-root ~/Desktop/car-dataset-check/car_dataset_split_v1

The dataset root must contain validation/real and validation/ai_generated,
(or it can point directly at the validation folder containing those classes).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from car_model_utils import load_cnn_model, prepare_image  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def locate_validation_dir(dataset_root: Path) -> Path:
    root = dataset_root.expanduser().resolve()
    candidates = [root, root / "validation"]
    for candidate in candidates:
        if (candidate / "real").is_dir() and (candidate / "ai_generated").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find validation class folders. Expected either "
        f"'{root / 'validation' / 'real'}' and '{root / 'validation' / 'ai_generated'}', "
        "or a dataset root that directly contains real/ and ai_generated/."
    )


def read_validation_images(validation_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    arrays: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    errors: list[str] = []

    for class_name, label in (("real", 0), ("ai_generated", 1)):
        class_dir = validation_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        files = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise FileNotFoundError(f"No supported images found in: {class_dir}")

        for path in files:
            try:
                with Image.open(path) as image:
                    image.load()
                    _, cnn_input, _ = prepare_image(image)
                arrays.append(cnn_input[0])
                labels.append(label)
                paths.append(str(path))
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")

    if not arrays:
        raise RuntimeError("No valid validation images could be read.")
    X = np.asarray(arrays, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    return X, y, paths, errors


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metric_row(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (scores >= threshold).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))

    precision_ai = safe_div(tp, tp + fp)
    recall_ai = safe_div(tp, tp + fn)
    f1_ai = safe_div(2 * precision_ai * recall_ai, precision_ai + recall_ai)
    specificity_real = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, len(y_true))
    balanced_accuracy = (recall_ai + specificity_real) / 2
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_ai": precision_ai,
        "recall_ai": recall_ai,
        "f1_ai": f1_ai,
        "specificity_real": specificity_real,
        "false_positive_rate_real": safe_div(fp, fp + tn),
        "true_positive_ai": tp,
        "false_negative_ai": fn,
        "true_negative_real": tn,
        "false_positive_real": fp,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", required=True, type=Path,
        help="Folder containing validation/real and validation/ai_generated, or the validation folder itself.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "reports" / "experiments" / "cnn_threshold_analysis",
        help="Directory for CSV outputs (default: reports/experiments/cnn_threshold_analysis).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="CNN inference batch size (default: 16).",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    validation_dir = locate_validation_dir(args.dataset_root)
    print(f"Validation directory: {validation_dir}")
    X, y_true, image_paths, read_errors = read_validation_images(validation_dir)
    print(f"Valid images: {len(y_true)} (real={int(np.sum(y_true == 0))}, AI-generated={int(np.sum(y_true == 1))})")

    print("Loading saved CNN model (no training will occur)...")
    model = load_cnn_model()
    try:
        raw = model.predict(X, batch_size=args.batch_size, verbose=0)
    except TypeError:
        raw = model.predict(X, verbose=0)
    scores = np.asarray(raw, dtype=np.float64).reshape(-1)
    if len(scores) != len(y_true) or not np.all(np.isfinite(scores)):
        raise RuntimeError(
            f"Unexpected CNN output shape or non-finite scores: {np.asarray(raw).shape}"
        )
    scores = np.clip(scores, 0.0, 1.0)

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_image = []
    for path, y, score in zip(image_paths, y_true, scores):
        per_image.append({
            "image_path": path,
            "true_label": "ai_generated" if int(y) == 1 else "real",
            "true_label_id": int(y),
            "cnn_ai_score_uncalibrated": float(score),
            "prediction_at_0_50": "ai_generated" if score >= 0.50 else "real",
            "correct_at_0_50": bool((score >= 0.50) == (int(y) == 1)),
        })
    write_csv(args.output_dir / "cnn_validation_scores.csv", per_image)

    thresholds = [round(float(t), 2) for t in np.arange(0.20, 0.801, 0.05)]
    sweep = [metric_row(y_true, scores, t) for t in thresholds]
    # This flag highlights a descriptive validation-set maximum only. It is not
    # a promise of better test/generalization performance.
    best = max(sweep, key=lambda r: (float(r["f1_ai"]), float(r["recall_ai"]), -float(r["false_positive_rate_real"])))
    for row in sweep:
        row["max_validation_f1_ai_tie_break_recall_then_lower_fpr"] = bool(
            math.isclose(float(row["threshold"]), float(best["threshold"]))
        )
    write_csv(args.output_dir / "cnn_threshold_sweep.csv", sweep)

    if read_errors:
        (args.output_dir / "image_read_errors.txt").write_text("\n".join(read_errors) + "\n", encoding="utf-8")

    baseline = metric_row(y_true, scores, 0.50)
    print("\nCNN validation metrics at the current threshold 0.50:")
    for key in ("accuracy", "balanced_accuracy", "precision_ai", "recall_ai", "f1_ai", "false_positive_rate_real"):
        print(f"  {key}: {float(baseline[key]):.4f}")
    print("\nBest threshold by validation AI F1 (ties: higher AI recall, then lower real-image FPR):")
    print(f"  threshold={float(best['threshold']):.2f}, AI precision={float(best['precision_ai']):.3f}, "
          f"AI recall={float(best['recall_ai']):.3f}, AI F1={float(best['f1_ai']):.3f}, "
          f"real-image FPR={float(best['false_positive_rate_real']):.3f}")
    print(f"\nWrote: {args.output_dir / 'cnn_validation_scores.csv'}")
    print(f"Wrote: {args.output_dir / 'cnn_threshold_sweep.csv'}")
    print("IMPORTANT: This is validation-only exploratory analysis. Do not use these results as test metrics or proof of generalization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
