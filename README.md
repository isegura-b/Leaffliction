# Leaffliction

Computer Vision project for leaf disease recognition.

## Requirements

- Python 3
- OpenCV
- NumPy
- Matplotlib

## Setup

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
opencv-python
numpy
matplotlib
```

## Dataset structure

The original dataset contains eight classes, divided into Apple and Grape diseases.

```text
leaves/
└── images/
    ├── Apple_Black_rot/
    ├── Apple_healthy/
    ├── Apple_rust/
    ├── Apple_scab/
    ├── Grape_Black_rot/
    ├── Grape_Esca/
    ├── Grape_healthy/
    └── Grape_spot/
```

## Part 1: Distribution

Analyze the dataset by counting the images in each disease class.

Generate bar charts and pie charts to visualize the distribution of Apple and Grape images.

Run:

```bash
python3 Distribution.py leaves/images
```

The program displays:
- Number of images per disease.
- Total number of images.
- Bar charts.
- Pie charts.

To analyze the balanced dataset:

```bash
python3 Distribution.py augmented_directory
```

## Part 2: Data Augmentation

Implement six data augmentation techniques:

- Flip
- Rotate
- Skew
- Shear
- Crop
- Distortion

The program identifies the class with the highest number of images and calculates how many additional images each remaining class needs.

It generates new images using the six augmentation techniques until all classes have the same number of images.

Original images are copied to the output directory, while the original dataset remains unchanged.

Run:

```bash
python3 Augmentation.py leaves/images
```

The balanced dataset is saved in `augmented_directory`:

```text
augmented_directory/
├── Apple_Black_rot/
├── Apple_healthy/
├── Apple_rust/
├── Apple_scab/
├── Grape_Black_rot/
├── Grape_Esca/
├── Grape_healthy/
└── Grape_spot/
```

Check the resulting distribution:

```bash
python3 Distribution.py augmented_directory
```

**Note:** Running Augmentation again removes and recreates the existing `augmented_directory`.

## Part 3: Image Transformation

Implement image processing techniques to extract and analyze visual characteristics of leaf images.

The implemented transformations are:

1. Gaussian blur
2. Mask
3. ROI objects
4. Analyze object
5. Pseudolandmarks
6. Color histogram

### Process a single image

```bash
python3 Transformation.py "leaves/images/Apple_healthy/image.jpg"
```

The program displays the original image and its transformations.

### Process a complete directory

```bash
python3 Transformation.py -src leaves/images -dst transformations
```

The program recursively processes the images in the source directory and saves the results in the specified destination directory.

The original directory is never modified.

Each processed image produces five transformed images and one color histogram.

Results are saved in a separate directory, preserving the original class structure:

```text
transformations/
├── Apple_Black_rot/
├── Apple_healthy/
├── Apple_rust/
├── Apple_scab/
├── Grape_Black_rot/
├── Grape_Esca/
├── Grape_healthy/
└── Grape_spot/
```

## Project structure

```text
Leaffliction/
├── leaves/
│   └── images/
│       ├── Apple_Black_rot/
│       ├── Apple_healthy/
│       ├── Apple_rust/
│       ├── Apple_scab/
│       ├── Grape_Black_rot/
│       ├── Grape_Esca/
│       ├── Grape_healthy/
│       └── Grape_spot/
│
├── augmented_directory/
├── transformations/
│
├── Distribution.py
├── Augmentation.py
├── Transformation.py
├── requirements.txt
└── README.md
```

## Important

- `leaves/images` contains the original dataset and is never modified.
- `augmented_directory` contains the original images and the additional images generated to balance the classes.
- `transformations` contains the results of image processing. These files are not additional training examples.
- The three output and source directories must remain separate.

## Classification preparation

Before training a classification model, the original dataset must be divided into Training and Validation sets.

Augmentation must only be applied to Training.

Validation must contain original images that have not been used during training or augmentation. This prevents data leakage and ensures a reliable accuracy evaluation.

The validation set must contain at least 100 images, and the target accuracy is above 90%.

## Deactivate the virtual environment

When finished, deactivate the virtual environment:

```bash
deactivate
```