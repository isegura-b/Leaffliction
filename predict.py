import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from ml.model import create_model
from ml.transforms import get_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error("No se encuentra la fotografía.")
    if not Path("model.pth").is_file():
        parser.error("Falta model.pth. Ejecuta train.py para generarlo.")

    try:
        # Recupera los pesos y las clases en el orden del entrenamiento.
        checkpoint = torch.load("model.pth", map_location="cpu",
                                weights_only=True)
        model = create_model()
        model.load_state_dict(checkpoint["model_state_dict"])
        classes = checkpoint["classes"]

        with Image.open(args.image) as image:
            original = image.convert("RGB")
        transformed = get_transforms()(original)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(f"No se puede cargar la imagen o el modelo: {error}")

    # Añade la dimensión del lote y predice sin calcular gradientes.
    model.eval()
    with torch.no_grad():
        prediction = model(transformed.unsqueeze(0)).argmax(dim=1).item()
    print(f"Predicción: {classes[prediction]}")

    # Deshace la normalización: de [-1, 1] a [0, 1].
    visible = (transformed * 0.5 + 0.5).permute(1, 2, 0)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(visible)
    axes[1].set_title(f"Transformada: {classes[prediction]}")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
