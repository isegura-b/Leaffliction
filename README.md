

\# Leaffliction



Computer Vision project for leaf disease recognition.



\## Requirements



\- Python 3

\- Matplotlib



\## Setup



Create virtual environment:



python3 -m venv venv



Activate virtual environment:



source venv/bin/activate



Install dependencies:



pip install -r requirements.txt



\## Dataset structure



leaves/

└── images/

&#x20;   ├── Apple\_Black\_rot/

&#x20;   ├── Apple\_healthy/

&#x20;   ├── Apple\_rust/

&#x20;   ├── Apple\_scab/

&#x20;   ├── Grape\_Black\_rot/

&#x20;   ├── Grape\_Esca/

&#x20;   ├── Grape\_healthy/

&#x20;   └── Grape\_spot/



\## Distribution



Count images and generate charts:



python3 Distribution.py leaves/images



