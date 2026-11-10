"""Carga y divide el dataset."""

from random import Random

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from .transforms import get_transforms


def load_dataset(dataset_path, transform=None):
    """Carga las imágenes; cada carpeta es una clase."""
    return ImageFolder(root=dataset_path, transform=transform)


def split_dataset(dataset_path, training_transform=None,
                  validation_transform=None):
    """Divide cada clase: 80% para entrenar y 20% para validar."""
    if training_transform is None:
        training_transform = get_transforms()
    if validation_transform is None:
        validation_transform = get_transforms()

    dataset = load_dataset(dataset_path, training_transform)
    validation_dataset = load_dataset(dataset_path, validation_transform)
    random = Random(42)  # Para que la división sea siempre la misma.
    training_indices = []
    validation_indices = []

    for label in range(len(dataset.classes)):
        indices = [
            index for index, target in enumerate(dataset.targets)
            if target == label
        ]
        random.shuffle(indices)
        split = round(len(indices) * 0.8)
        training_indices.extend(indices[:split])
        validation_indices.extend(indices[split:])

    return (
        Subset(dataset, training_indices),
        Subset(validation_dataset, validation_indices),
    )


def get_dataloaders(dataset_path, training_transform=None,
                    validation_transform=None):
    """Carga lotes de 32 imágenes para entrenar y validar."""
    training, validation = split_dataset(
        dataset_path, training_transform, validation_transform
    )
    training_loader = DataLoader(
        training,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    return training_loader, validation_loader
