#!/usr/bin/env python3
"""Verify the currently configured Listing Lens CNN and optionally predict one photo."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.cnn_inference import CONFIG, MODEL_PATH, AI_THRESHOLD, load_model, prepare_input, predict


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="Optional image path for a real-photo prediction")
    args = parser.parse_args()

    print(f"Active model: {MODEL_PATH}")
    print(f"SHA-256: {sha256(MODEL_PATH)}")
    print(f"Threshold: {AI_THRESHOLD:.2f}")
    print(f"Configured hash match: {sha256(MODEL_PATH) == CONFIG.get('model_sha256')}")
    model = load_model()
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    if args.image:
        image_path = Path(args.image).expanduser().resolve()
        image_bytes = image_path.read_bytes()
        batch = prepare_input(image_bytes, image_path.suffix)
    else:
        batch = np.zeros((1, 256, 256, 3), dtype=np.float32)
        print("No photo supplied; running a synthetic tensor forward-pass smoke test.")

    result = predict(model, batch)
    print("Prediction smoke test: PASS")
    print(f"Label: {result['prediction']}")
    print(f"AI-pattern score: {result['ai_score']:.4f}")
    print(f"Threshold: {result['threshold']:.2f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
