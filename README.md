# Assignment 14 — Optimized ViT Deployment with LoRA, Quantization, FastAPI, and Docker

## Overview

This project fine-tunes a Vision Transformer (ViT) model on the EuroSAT RGB dataset using Low-Rank Adaptation (LoRA), applies dynamic INT8 quantization for deployment, and exposes the model through a FastAPI inference service running inside Docker.

The project demonstrates an end-to-end machine learning deployment pipeline:

Dataset → ViT + LoRA → Best Checkpoint → INT8 Quantization → FastAPI → Docker

---

## Objectives

- Fine-tune a pretrained Vision Transformer using LoRA.
- Reduce the number of trainable parameters.
- Save the best validation checkpoint.
- Apply dynamic INT8 quantization.
- Build a FastAPI image-classification API.
- Containerize the API using Docker.
- Test inference locally and inside the Docker container.

---

## Dataset

The project uses the **EuroSAT RGB** dataset.

### Classes

The model predicts 10 land-use classes:

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

### Dataset split

- Training images: 21,600
- Validation images: 5,400
- Total images: 27,000

---

## Model

The base model is:

```text
google/vit-base-patch16-224

The model uses:
- ViT-Base
- 224 × 224 input resolution
- 10 output classes
- LoRA applied to Query and Value projections
- Last 3 transformer blocks targeted
LoRA configuration
Rank: 16
Alpha: 16
Dropout: 0.05
Target blocks: 9, 10, 11
Target projections: Q and V

Parameter efficiency
Total parameters:
85,953,802

Trainable parameters:
147,456

Trainable percentage:
0.1716%

Training
Training was performed using a GPU environment.
Training configuration
Epochs: 3
Learning rate: 1e-4
Weight decay: 0.01
Optimizer: AdamW
Batch size: 16

Results
| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
|------:|-----------:|---------------:|----------------:|---------------------:|
| 1 | 0.8850 | 73.22% | 0.6598 | 80.65% |
| 2 | 0.2285 | 93.50% | 0.3952 | 89.11% |
| 3 | 0.1558 | 95.22% | 0.3054 | 91.35% |

Best validation accuracy:
91.35%

Best validation loss:
0.3054

The best checkpoint is saved as:
models/best_lora/best_model.pt

Quantization
Dynamic INT8 quantization was applied to the trained model.
The quantized model is saved as:
models/quantized/vit_lora_int8.pt

Model size reduction
Original checkpoint:
327.96 MB

Quantized model:
84.99 MB

Reduction:
242.97 MB

Percentage reduction:
74.08%

The quantized model was tested successfully using CPU inference.
Local Inference Test
The quantized model was tested with:
AnnualCrop_1.jpg

Prediction:
Predicted class: AnnualCrop
Confidence: 60.61%

This confidence represents the prediction for this individual image and is not the overall validation accuracy.
FastAPI
The project provides three endpoints.
Root endpoint
GET /

Example response:
{
  "message": "EuroSAT ViT LoRA API is running",
  "model": "ViT-Base + LoRA + INT8",
  "classes": 10
}

Health endpoint
GET /health

Example response:
{
  "status": "healthy",
  "model_loaded": true,
  "num_classes": 10
}

Prediction endpoint
POST /predict

The endpoint accepts an image file and returns the predicted EuroSAT class and confidence.
Example:
{
  "filename": "AnnualCrop_1.jpg",
  "predicted_class": "AnnualCrop",
  "confidence": 0.6061
}

Running the API Locally
Activate the virtual environment and move into the app directory:
cd app
python -m uvicorn main:app --host 127.0.0.1 --port 8000

Test the health endpoint:
curl.exe http://127.0.0.1:8000/health

Test prediction:
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@data/EuroSAT/EuroSAT_RGB/AnnualCrop/AnnualCrop_1.jpg"

Docker
The application is containerized using Docker.
Build the image
From the project root:
docker build -t eurosat-vit-api .

Run the container
docker run --name eurosat-vit-container -p 8000:8000 eurosat-vit-api

The API is then available at:
http://127.0.0.1:8000

Test the Docker container
Health check:
curl.exe http://127.0.0.1:8000/health

Prediction:
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@data/EuroSAT/EuroSAT_RGB/AnnualCrop/AnnualCrop_1.jpg"

Docker inference test result:
{
  "filename": "AnnualCrop_1.jpg",
  "predicted_class": "AnnualCrop",
  "confidence": 0.6223
}

This confirms that the quantized model can be loaded and used for inference inside the Docker container.
Project Structure
Assignment_14_ViT_Deployment/
│
├── app/
│   ├── main.py
│   ├── model_loader.py
│   └── preprocessing.py
│
├── training/
│   ├── dataset.py
│   ├── lora_model.py
│   ├── train_lora.py
│   └── quantize_model.py
│
├── models/
│   ├── best_lora/
│   │   └── best_model.pt
│   └── quantized/
│       └── vit_lora_int8.pt
│
├── tests/
│   └── test_quantized_model.py
│
├── data/
│   └── EuroSAT/
│
├── requirements.txt
├── requirements-docker.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md

Technologies
- Python
- PyTorch
- Torchvision
- Hugging Face Transformers
- Vision Transformer (ViT)
- LoRA
- Dynamic INT8 Quantization
- FastAPI
- Uvicorn
- Docker
- EuroSAT
Deployment Pipeline
EuroSAT Dataset
       ↓
Pretrained ViT
       ↓
LoRA Fine-Tuning
       ↓
Best Model Checkpoint
       ↓
Dynamic INT8 Quantization
       ↓
Quantized Model
       ↓
FastAPI
       ↓
Docker Container
       ↓
Image Classification API

Final Results
Metric	Result
Classes	10
Training images	21,600
Validation images	5,400
Total parameters	85,953,802
Trainable parameters	147,456
Trainable percentage	0.1716%
Best validation accuracy	91.35%
Best validation loss	0.3054
Original model size	327.96 MB
Quantized model size	84.99 MB
Size reduction	74.08%
FastAPI	Working
Docker deployment	Working
Docker inference	Passed


Conclusion
This project demonstrates a complete optimized computer vision deployment workflow.
LoRA enables parameter-efficient fine-tuning of the Vision Transformer, while dynamic INT8 quantization significantly reduces the model's storage footprint. FastAPI provides a lightweight inference interface, and Docker packages the complete inference environment into a reproducible deployment.
The final Dockerized API successfully loads the quantized ViT + LoRA model and performs EuroSAT image classification.

### One small note

Your README's numbers are based on the results we actually obtained during the project, including **91.35% validation accuracy**, **74.08% size reduction**, and the successful Docker prediction.

Save the file.

Then **don't rebuild Docker yet**. The next thing we'll do is a quick `git status`/project cleanup check so we don't accidentally commit the 27,000-image dataset or your `.venv`.

## Assignment 15 — Git, Branching and CI/CD

This assignment extends the ViT deployment project with a Git-based development workflow and continuous integration.

### Git Workflow

The project uses two branches:

- `main` — stable version
- `develop` — development version

The workflow demonstrates:

- `git status`
- `git add`
- `git commit`
- `git push`
- `git pull`
- `git diff`
- `git merge`
- Merge conflict resolution

### CI/CD

GitHub Actions is used to automate project validation.

The CI workflow performs automated checks when changes are pushed to the repository.

Planned CI jobs include:

- Install dependencies
- Run Python validation/tests
- Validate the Docker build