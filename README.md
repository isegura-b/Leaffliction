# Leaffliction

Proyecto de visión artificial para analizar hojas y clasificar enfermedades de
manzanos y vides.

Este README explica el funcionamiento actual del proyecto en Linux. Las diferencias
con `Leaffliction.pdf` aparecen marcadas como **Pendiente**.

## 1. Instalación

Se recomienda Python 3.10 o 3.11.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

`requirements.txt` instala OpenCV, NumPy, Matplotlib, PyTorch, Torchvision y
Pillow.

Para salir del entorno virtual:

```bash
deactivate
```

## 2. Estructura

Las fotografías originales deben tener esta estructura:

```text
leaves/
└── images/
    ├── Apple/
    │   ├── Apple_Black_rot/
    │   ├── Apple_healthy/
    │   ├── Apple_rust/
    │   └── Apple_scab/
    └── Grape/
        ├── Grape_Black_rot/
        ├── Grape_Esca/
        ├── Grape_healthy/
        └── Grape_spot/
```

Archivos principales:

- `Distribution.py`: muestra la distribución de las clases.
- `Augmentation.py`: equilibra las clases de Training.
- `Transformation.py`: muestra o guarda transformaciones visuales.
- `prepare.py`: divide las imágenes en Training y Validation.
- `train.py`: entrena la CNN y genera el modelo y el ZIP.
- `predict.py`: clasifica una fotografía usando el modelo entrenado.
- `ml/`: carga de datos, transformaciones, CNN y entrenamiento.

## 3. Preparar Training y Validation

Desde la raíz del proyecto:

```bash
python3 prepare.py
```

El programa:

- Lee `leaves/images/` de forma recursiva.
- Copia el 80 % de cada clase a `prepared/train/`.
- Copia el 20 % restante a `prepared/validation/`.
- Usa la semilla 42 para repetir siempre la misma división.
- No modifica las fotografías originales.

También acepta un origen y un destino:

```bash
python3 prepare.py leaves/images prepared
```

El destino no debe existir. Al terminar debe aparecer:

```text
prepared/
├── train/
│   └── ocho carpetas de enfermedades
└── validation/
    └── las mismas ocho carpetas
```

Validation debe contener solamente fotografías originales. Nunca se debe aumentar
Validation.

## 4. Procesamiento de imágenes

### Distribution.py

Después de ejecutar `prepare.py`:

```bash
python3 Distribution.py prepared/train
python3 Distribution.py prepared/validation
```

El programa imprime el número de entradas por clase y muestra gráficos de Apple y
Grape.

**Pendiente:** solo revisa las subcarpetas inmediatas y no filtra las entradas por
extensión. No funciona correctamente con `leaves/images`, ni con Apple o Grape por
separado, porque siempre intenta dibujar ambos tipos.

### Augmentation.py

Primero debe existir `prepared/train/`. Después ejecuta:

```bash
python3 Augmentation.py
```

El programa equilibra las clases directamente dentro de `prepared/train/`. Genera
imágenes Flip, Rotate, Skew, Shear, Crop y Distortion con nombres como:

```text
image_Flip_1.jpg
image_Rotate_2.jpg
```

Al terminar muestra las seis aumentaciones de una fotografía. No modifica
`prepared/validation/`.

**Pendiente:** utiliza la ruta fija `prepared/train`, no acepta argumentos y no
crea el `augmented_directory` solicitado en el PDF.

### Transformation.py

Mostrar las transformaciones de una fotografía:

```bash
python3 Transformation.py "leaves/images/Apple/Apple_healthy/image (1).JPG"
```

Procesar un directorio y guardar los resultados:

```bash
python3 Transformation.py -src prepared/train -dst transformations
```

Guardar únicamente las máscaras:

```bash
python3 Transformation.py -src prepared/train -dst masks -mask
```

Consultar la ayuda:

```bash
python3 Transformation.py -h
```

El destino debe estar fuera del origen. Estas transformaciones son resultados
visuales y no son las imágenes aumentadas que utiliza `train.py`.

## 5. Entrenamiento

Antes de entrenar:

1. Ejecuta `prepare.py`.
2. Ejecuta `Augmentation.py`.
3. Comprueba que Training y Validation contienen las mismas ocho clases.

Entrenar durante cinco épocas:

```bash
python3 train.py prepared
```

Cambiar el número de épocas:

```bash
python3 train.py prepared --epochs 10
```

En cada época se muestran:

- `Pérdida` o Loss: error medio en Training.
- `Accuracy Validation`: porcentaje de aciertos en Validation.

El PDF exige más del 90 % de Accuracy Validation con un mínimo de 100 imágenes.

Si el entrenamiento termina correctamente se crean, en el directorio actual:

- `model.pth`: pesos entrenados y nombres de las clases.
- `Leaffliction.zip`: contiene `model.pth` y las imágenes JPG aumentadas,
  guardadas como `train/<clase>/<imagen>.jpg`.

El ZIP no incluye los originales ni Validation. Si faltan imágenes aumentadas,
`train.py` termina antes de entrenar y muestra un error.

**Pendiente:** el PDF muestra `train.py` recibiendo directamente un directorio de
enfermedades. El código actual necesita el directorio `prepared/`. También falta
realizar el entrenamiento definitivo y demostrar una accuracy superior al 90 %.

## 6. Predicción

`predict.py` necesita `model.pth` en el directorio actual:

```bash
python3 predict.py "leaves/images/Apple/Apple_healthy/image (1).JPG"
```

El programa carga los pesos en CPU, transforma la fotografía, imprime la enfermedad
predicha y muestra la imagen original junto a la transformada.

`predict.py` no carga el modelo directamente desde `Leaffliction.zip`.

## 7. Firma y entrega

Genera la firma solo cuando el ZIP sea definitivo:

```bash
sha1sum Leaffliction.zip
sha1sum Leaffliction.zip > signature.txt
```

Comprueba que coincide:

```bash
sha1sum -c signature.txt
```

El resultado correcto es:

```text
Leaffliction.zip: OK
```

El PDF exige entregar los programas y `signature.txt`. No se deben subir:

- `leaves/` ni ningún dataset.
- `prepared/`.
- `transformations/` o `masks/`.
- `model.pth`.
- `Leaffliction.zip`.
- El entorno virtual y archivos temporales.

**Pendiente crítico:** actualmente las fotografías de `leaves/` están seguidas por
Git, aunque el PDF prohíbe entregar el dataset. Deben retirarse de la entrega antes
de la evaluación.

## 8. Problemas habituales

### Falta una dependencia

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### Faltan las fotografías

Colócalas en las ocho carpetas situadas bajo `leaves/images/Apple/` y
`leaves/images/Grape/`.

### Falta prepared/

```bash
python3 prepare.py
```

`prepare.py` no sobrescribe un destino existente.

### Faltan imágenes aumentadas

```bash
python3 Augmentation.py
```

Comprueba que `prepared/train/` contiene nombres con `_Flip_`, `_Rotate_`,
`_Skew_`, `_Shear_`, `_Crop_` o `_Distortion_`.

### Falta model.pth

```bash
python3 train.py prepared
```

## Orden de ejecución

```bash
source .venv/bin/activate
python3 prepare.py
python3 Distribution.py prepared/train
python3 Augmentation.py
python3 train.py prepared
python3 predict.py "leaves/images/Apple/Apple_healthy/image (1).JPG"
```

`Transformation.py` es independiente del entrenamiento. `signature.txt` se crea
solamente después del entrenamiento definitivo.
