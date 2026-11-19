"""Carga los conjuntos de entrenamiento y validación preparados."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .transforms import get_transforms


def load_dataset(dataset_path, transform=None):
    """Carga las imágenes; cada carpeta es una clase."""
    return ImageFolder(root=dataset_path, transform=transform)


def get_dataloaders(dataset_path="prepared", training_transform=None,
                    validation_transform=None):
    """Carga train/ y validation/ en lotes de 32, sin dividirlos."""
    if training_transform is None:
        training_transform = get_transforms()
    if validation_transform is None:
        validation_transform = get_transforms()

    dataset_path = Path(dataset_path)
    training = load_dataset(dataset_path / "train", training_transform)
    validation = load_dataset(dataset_path / "validation", validation_transform)
    if training.class_to_idx != validation.class_to_idx:
        raise ValueError("Train y validation deben tener las mismas clases.")
    training_loader = DataLoader(training, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=32, shuffle=False)
    return training_loader, validation_loader
