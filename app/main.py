from io import BytesIO

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from model_loader import load_model
from monitoring import (
    get_prediction_metrics,
    log_prediction,
    measure_time,
)
from drift import check_drift
from system_monitoring import get_system_metrics
from preprocessing import preprocess_image


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="EuroSAT ViT LoRA API",
    description="Image classification API using a quantized ViT + LoRA model.",
    version="1.0.0",
)


# ============================================================
# LOAD MODEL ON STARTUP
# ============================================================

model, class_names = load_model()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "EuroSAT ViT LoRA API is running",
        "model": "ViT-Base + LoRA + INT8",
        "classes": len(class_names),
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": len(class_names),
    }


# ============================================================
# MODEL OBSERVABILITY METRICS
# ============================================================

@app.get("/metrics")
def metrics():

    return get_prediction_metrics()


# ============================================================
# INFRASTRUCTURE MONITORING
# ============================================================

@app.get("/system")
def system_metrics():

    return get_system_metrics()


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    true_label: str | None = Form(default=None),
):

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="File type could not be determined.",
        )

    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        contents = await file.read()

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        )

    # --------------------------------------------------------
    # Data drift check
    # --------------------------------------------------------

    try:

        drift_result = check_drift(
            image
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Data drift check failed: {exc}",
        )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    try:

        pixel_values = preprocess_image(
            image
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Image preprocessing failed: {exc}",
        )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    start_time = measure_time()

    try:

        with torch.no_grad():

            outputs = model(
                pixel_values=pixel_values
            )

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        predicted_id = torch.argmax(
            probabilities,
            dim=-1,
        ).item()

        confidence = probabilities[
            0,
            predicted_id,
        ].item()

        predicted_class = class_names[
            predicted_id
        ]

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {exc}",
        )

    # --------------------------------------------------------
    # Calculate inference latency
    # --------------------------------------------------------

    latency_ms = (
        measure_time() - start_time
    ) * 1000

    # --------------------------------------------------------
    # Log prediction
    # --------------------------------------------------------

    log_prediction(
        filename=file.filename,
        predicted_class=predicted_class,
        confidence=confidence,
        latency_ms=latency_ms,
        true_label=true_label,
        drift_result=drift_result,
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": round(
            confidence,
            4,
        ),
        "latency_ms": round(
            latency_ms,
            2,
        ),
        "true_label": true_label,
        "drift": drift_result,
    }