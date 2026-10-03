from PIL import Image
from transformers import AutoImageProcessor
from torchvision import transforms


MODEL_NAME = "google/vit-base-patch16-224"


def create_preprocessor():

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std,
        ),
    ])

    return transform


def preprocess_image(image):

    if not isinstance(image, Image.Image):
        raise TypeError(
            "Expected a PIL Image."
        )

    image = image.convert("RGB")

    transform = create_preprocessor()

    pixel_values = transform(image)

    # Add batch dimension
    pixel_values = pixel_values.unsqueeze(0)

    return pixel_values