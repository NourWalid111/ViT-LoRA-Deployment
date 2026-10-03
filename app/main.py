from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from model_loader import load_model
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
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "message": "EuroSAT ViT LoRA API is running",
        "model": "ViT-Base + LoRA + INT8",
        "classes": len(class_names),
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": len(class_names),
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
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
    # Response
    # --------------------------------------------------------

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": round(
            confidence,
            4,
        ),
    }