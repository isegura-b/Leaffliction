import torch.nn as pt


def create_model():
    # Sequential aplica las capas en orden, de arriba abajo.
    # Conv2d indica canales de entrada, de salida y tamaño del filtro.
    # padding=1 conserva el tamaño con filtros de 3 x 3.
    # ReLU pone los negativos a cero; MaxPooling reduce ancho y alto a la mitad.
    return pt.Sequential(
        # Bloque 1: de 3 canales RGB a 16 mapas de 112 x 112.
        pt.Conv2d(3, 16, 3, padding=1),
        pt.ReLU(),
        pt.MaxPool2d(2),

        # Bloque 2: de 16 a 32 mapas de 56 x 56.
        pt.Conv2d(16, 32, 3, padding=1),
        pt.ReLU(),
        pt.MaxPool2d(2),

        # Bloque 3: de 32 a 64 mapas de 28 x 28.
        pt.Conv2d(32, 64, 3, padding=1),
        pt.ReLU(),
        pt.MaxPool2d(2),

        # Flatten crea un vector por imagen; Linear calcula 8 puntuaciones.
        pt.Flatten(),
        pt.Linear(64 * 28 * 28, 8),
    )
