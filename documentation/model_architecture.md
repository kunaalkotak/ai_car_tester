# Active CNN V4 Architecture — Technical Description

The Streamlit application calls the selected model **CNN**. Its saved technical architecture uses an ImageNet-pretrained EfficientNetB0 backbone.

```text
Input: 256 × 256 × 3 RGB (float32, raw values 0–255)
    │
    ▼
Training-only augmentation layers
    ├─ Random horizontal flip
    ├─ Random rotation (factor 0.04)
    ├─ Random zoom (factor 0.10)
    ├─ Random contrast (factor 0.10)
    └─ Random brightness (factor 0.10)
    │
    ▼
EfficientNetB0 backbone (ImageNet pretrained, include_top=False)
    │  Built-in rescaling / normalization
    ▼
GlobalAveragePooling2D
    ▼
Dropout(0.40)
    ▼
Dense(64, ReLU, L2 regularization 1e-4)
    ▼
Dropout(0.30)
    ▼
Dense(1, Sigmoid)
    │
    ▼
AI-generated score; compare with threshold 0.50
```

- Class 0: `real`
- Class 1: `ai_generated`
- Model output: one sigmoid score (uncalibrated).
- Inference uses raw RGB 0–255 values; the model contains normalization layers, so the application must not pre-divide by 255.
- Training augmentation is disabled during inference.

The active V4 model is a fine-tuned continuation of the preserved V3 EfficientNetB0 checkpoint. The active model SHA-256 is checked by the inference loader and recorded in `config/cnn_inference_config.json`. V3 remains in `models_v3/` as an unchanged rollback backup.
