# Leaffliction

Computer Vision project for leaf disease recognition.

## Requirements

- Python 3
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
pip install -r requirements.txt
```

## Dataset structure

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

## Distribution

Count images and generate bar and pie charts for each plant type:

```bash
python3 Distribution.py leaves/images
```
