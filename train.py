
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import torch

from ml.dataset import get_dataloaders
from ml.model import create_model
from ml.trainer import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    # Comprobar que tenemos los datos preparados.
    if args.epochs < 1:
        parser.error("El número de épocas debe ser mayor que cero.")

    if (
        not (args.directory / "train").is_dir()
        or not (args.directory / "validation").is_dir()
    ):
        parser.error("Faltan train/ o validation/. Ejecuta prepare.py.")

    effects = ("Flip", "Rotate", "Skew", "Shear", "Crop", "Distortion")
    augmented_images = [
        image for image in (args.directory / "train").rglob("*")
        if image.is_file() and image.suffix.lower() == ".jpg"
        and any(f"_{effect}" in image.stem for effect in effects)
    ]
    if not augmented_images:
        parser.error("Faltan las imágenes aumentadas. Ejecuta Augmentation.py.")

    # Cargar imágenes.
    try:
        train_loader, validation_loader = get_dataloaders(args.directory)
    except (FileNotFoundError, ValueError) as error:
        parser.error(f"No se pueden cargar las imágenes: {error}")

    classes = train_loader.dataset.classes
    if len(classes) < 2:
        parser.error("Se necesitan mas de dos clases de enfermedades.")

    # Crear y entrenar la CNN.
    model = create_model()
    model = train_model(model, train_loader, validation_loader, epochs=args.epochs)

    # Guardar los pesos aprendidos y los nombres de las enfermedades.
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes
    }, "model.pth")

    print("Modelo guardado en model.pth")

    with ZipFile("Leaffliction.zip", "w", ZIP_DEFLATED) as archive:
        archive.write("model.pth")
        for image in augmented_images:
            archive.write(image, image.relative_to(args.directory))

    print("Modelo e imágenes aumentadas guardados en Leaffliction.zip")


if __name__ == "__main__":
    main()
