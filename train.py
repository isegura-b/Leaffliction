
import argparse
from pathlib import Path
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

    # Cargar imágenes.
    try:
        train_loader, validation_loader = get_dataloaders(args.directory)
    except (FileNotFoundError, ValueError) as error:
        parser.error(f"No se pueden cargar las imágenes: {error}")

    classes = train_loader.dataset.classes
    if len(classes) != 8:
        parser.error("Se necesitan ocho clases de enfermedades.")

    # Crear y entrenar la CNN.
    model = create_model()
    model = train_model(model, train_loader, validation_loader, epochs=args.epochs)

    # Guardar los pesos aprendidos y los nombres de las enfermedades.
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes
    }, "model.pth")

    print("Modelo guardado en model.pth")


if __name__ == "__main__":
    main()
