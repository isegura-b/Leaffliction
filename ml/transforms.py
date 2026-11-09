"""Prepara las imágenes para la CNN."""
from torchvision import transforms


def to_rgb(image):
    """Pasa la imagen a RGB."""
    return image.convert("RGB")


def get_transforms():
    """Ajusta el tamaño y pasa los píxeles a valores entre -1 y 1."""
    return transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])
