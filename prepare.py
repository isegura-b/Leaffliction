# Copia los originales en train (80%) y validation (20%), con semilla 42.

import argparse
from pathlib import Path
from random import Random
from shutil import copy2

from torchvision.datasets.folder import IMG_EXTENSIONS


def prepare_dataset(source="leaves/images", destination="prepared"):
    # Crea una división reproducible sin modificar los originales.
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if (source == destination or source in destination.parents
            or destination in source.parents):
        raise ValueError("Origen y destino deben estar separados.")
    if destination.exists():
        raise ValueError("El destino ya existe; usa un directorio nuevo.")

    if not source.is_dir():
        raise ValueError("El directorio de origen no existe.")
    random = Random(42)
    classes = {}
    for folder in sorted(source.rglob("*")):
        if not folder.is_dir():
            continue
        images = sorted(p for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)
        if not images:
            continue
        name = folder.name
        if name in classes:
            raise ValueError(f"Nombre de clase repetido: {name}")
        random.shuffle(images)
        split = round(len(images) * 0.8)
        if split == 0 or split == len(images):
            raise ValueError(f"{name}: faltan imágenes para ambos conjuntos.")
        classes[name] = (images[:split], images[split:])

    if not classes:
        raise ValueError("No se encontraron imágenes por clase.")

    for name, (training, validation) in classes.items():
        for subset, images in (("train", training),
                               ("validation", validation)):
            folder = destination / subset / name
            folder.mkdir(parents=True)
            for image in images:
                copy2(image, folder / image.name)
        print(f"{name}: train={len(training)}, validation={len(validation)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copia los originales en train (80%) y validation (20%), "
                    "con semilla 42."
    )
    parser.add_argument("source", nargs="?", default="leaves/images")
    parser.add_argument("destination", nargs="?", default="prepared")
    args = parser.parse_args()
    try:
        prepare_dataset(args.source, args.destination)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
