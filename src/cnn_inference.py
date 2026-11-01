"""Single-model CNN inference for Listing Lens.

The packaged CNN is an ImageNet-pretrained EfficientNetB0 model. The application
uses the short display name "CNN"; architecture details are documented in the
README and inference configuration.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "cnn_inference_config.json"

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG: dict[str, Any] = json.load(config_file)

MODEL_PATH = PROJECT_ROOT / CONFIG["model_file"]
IMAGE_SIZE = tuple(int(value) for value in CONFIG["input_preprocessing"]["target_size"])
AI_THRESHOLD = float(CONFIG["decision_threshold"])
EXPECTED_HASH = str(CONFIG.get("model_sha256", "")).strip().lower()
ALLOWED_EXTENSIONS = {
    str(extension).lower()
    for extension in CONFIG["input_preprocessing"]["supported_extensions"]
}
LABELS = {0: "real", 1: "ai_generated"}


def _tensorflow():
    """Import TensorFlow lazily so syntax/config checks do not need it installed."""
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - depends on the local runtime
        raise RuntimeError(
            "CNN inference requires TensorFlow. Activate your project environment "
            "and install the packages in requirements.txt."
        ) from exc
    return tf


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def load_model(model_path: str | Path = MODEL_PATH):
    """Load the saved model after validating its hash and tensor dimensions."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"CNN model not found: {path}. Check the model_file entry in "
            "config/cnn_inference_config.json."
        )

    if EXPECTED_HASH:
        actual_hash = _sha256(path)
        if actual_hash != EXPECTED_HASH:
            raise ValueError(
                "CNN model SHA-256 does not match the inference configuration. "
                f"Expected {EXPECTED_HASH}, received {actual_hash}."
            )

    tf = _tensorflow()
    model = tf.keras.models.load_model(path, compile=False)

    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)
    expected_input = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    if input_shape[-3:] != expected_input:
        raise ValueError(
            f"Unexpected CNN input shape {input_shape}; expected (*, {expected_input})."
        )
    if output_shape[-1] != 1:
        raise ValueError(
            f"Expected one sigmoid output from CNN, received shape {output_shape}."
        )
    return model


def _prepare_webp_fallback(image_bytes: bytes) -> np.ndarray:
    """Decode WebP with Pillow and bilinearly resize it as an inference fallback.

    WebP files were skipped by the recorded Keras training directory loader, so
    this path is for upload usability and is not represented in the V3 training
    evaluation. It preserves the previous V3 inference route for WebP inputs.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.seek(0)
            rgb = image.convert("RGB")
            resized = rgb.resize(
                (int(IMAGE_SIZE[1]), int(IMAGE_SIZE[0])),
                resample=Image.Resampling.BILINEAR,
            )
            return np.asarray(resized, dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Could not decode the uploaded WebP image: {exc}") from exc


def prepare_input(image_bytes: bytes, extension: str) -> np.ndarray:
    """Return a (1, 256, 256, 3) float32 tensor in the raw 0..255 range.

    JPEG, PNG, BMP and GIF bytes use TensorFlow's image decoder and bilinear
    resize, matching the recorded Keras directory-loader pipeline. WebP uses a
    separate Pillow RGB/resize fallback because it was skipped by that loader.
    EfficientNetB0 includes its own input normalisation inside the saved model;
    do not divide the input by 255 here.
    """
    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        raise ValueError("The uploaded image is empty or is not valid image data.")

    ext = str(extension).strip().lower()
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    if ext not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported image format {ext!r}. Supported formats: {supported}.")

    encoded = bytes(image_bytes)
    if ext == ".webp":
        image_array = _prepare_webp_fallback(encoded)
    else:
        tf = _tensorflow()
        try:
            decoded = tf.io.decode_image(
                tf.convert_to_tensor(encoded, dtype=tf.string),
                channels=3,
                expand_animations=False,
            )
            resized = tf.image.resize(decoded, IMAGE_SIZE, method="bilinear")
            image_array = np.asarray(resized, dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"Could not decode the uploaded image: {exc}") from exc

    expected_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    if image_array.shape != expected_shape:
        raise ValueError(
            f"Invalid preprocessed image shape {image_array.shape}; expected {expected_shape}."
        )
    if not np.isfinite(image_array).all():
        raise ValueError("The preprocessed image contains non-finite values.")

    return image_array[np.newaxis, ...]


def predict(model, image_batch: np.ndarray, threshold: float = AI_THRESHOLD) -> dict[str, Any]:
    """Return the CNN classification and its uncalibrated sigmoid score."""
    threshold = float(threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("The decision threshold must be between 0 and 1.")

    batch = np.asarray(image_batch, dtype=np.float32)
    expected_batch_shape = (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    if batch.shape != expected_batch_shape:
        raise ValueError(
            f"Expected input shape {expected_batch_shape}, received {batch.shape}."
        )
    if not np.isfinite(batch).all():
        raise ValueError("The input batch contains non-finite values.")

    raw_output = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
    if raw_output.size != 1 or not np.isfinite(raw_output[0]):
        raise ValueError(f"Unexpected CNN output: {raw_output!r}")

    ai_score = float(np.clip(raw_output[0], 0.0, 1.0))
    # Model predictions are float32. Compare the threshold at the same precision
    # so an output represented at the configured threshold is treated consistently at the decision boundary.
    predicted_id = int(np.float32(ai_score) >= np.float32(threshold))
    return {
        "model": "CNN",
        "predicted_id": predicted_id,
        "prediction": LABELS[predicted_id],
        "ai_score": ai_score,
        "real_score_complement": 1.0 - ai_score,
        "threshold": threshold,
        "score_type": "uncalibrated sigmoid score; not a calibrated probability",
    }


def predict_image_bytes(model, image_bytes: bytes, extension: str) -> dict[str, Any]:
    """Prepare and classify one uploaded image."""
    batch = prepare_input(image_bytes, extension)
    return predict(model, batch)
