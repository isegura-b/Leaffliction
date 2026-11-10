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

On Windows (PowerShell), use the virtual environment's Python directly:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe Distribution.py leaves/images
```

If `python` fails because it points to the Windows App execution alias, create
the environment using the installed interpreter:

```powershell
& "$env:LOCALAPPDATA\Python\bin\python.exe" -m venv .venv
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
