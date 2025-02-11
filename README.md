 Project Title
# CareerAdvisor

# Project Description

CareerAdvisor is a Python-based tool designed to help users find job listings tailored to their skills. It integrates with the Adzuna API to fetch real-time job listings and parses resumes to match relevant opportunities.

Features

## Features
- Fetches real-time job listings using the Adzuna API.
- Parses resumes to extract skills and keywords.
- Matches job descriptions with parsed resume content.
- Outputs matched jobs for easy review.

 # Project Setup

 ## Setup Instructions

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/CareerAdvisor.git
   cd CareerAdvisor

Requirements

## Requirements
- Python 3.8 or higher
- Dependencies (listed in `requirements.txt`):
  - requests
  - python-dotenv
  - pdfminer.six
  - pandas
  - numpy
sentence-transformers==3.3.1
python-dotenv==1.0.1
scikit-learn==1.3.0

Package               Version
--------------------- -----------
annotated-types       0.7.0
anyio                 4.7.0
certifi               2024.12.14
cffi                  1.17.1
charset-normalizer    3.4.0
colorama              0.4.6
cryptography          44.0.0
distro                1.9.0
filelock              3.16.1
fsspec                2024.12.0
h11                   0.14.0
httpcore              1.0.7
httpx                 0.28.1
huggingface-hub       0.27.0
idna                  3.10
Jinja2                3.1.5
jiter                 0.8.2
joblib                1.4.2
MarkupSafe            3.0.2
mpmath                1.3.0
networkx              3.4.2
numpy                 2.2.1
openai                1.58.1
packaging             24.2
pandas                2.2.3
pdfminer.six          20240706
pillow                11.0.0
pip                   24.3.1
pycparser             2.22
pydantic              2.10.4
pydantic_core         2.27.2
python-dateutil       2.9.0.post0
python-dotenv         1.0.1
pytz                  2024.2
PyYAML                6.0.2
regex                 2024.11.6
requests              2.32.3
safetensors           0.4.5
scikit-learn          1.3.0
scipy                 1.14.1
sentence-transformers 3.3.1
setuptools            75.6.0
six                   1.17.0
sniffio               1.3.1
sympy                 1.13.1
threadpoolctl         3.5.0
tokenizers            0.21.0
torch                 2.5.1
tqdm                  4.67.1
transformers          4.47.1
typing_extensions     4.12.2
tzdata                2024.2
urllib3               2.3.0

# Folder Structure

CareerAdvisor/
├── data/
│   ├── job_listings.json
│   ├── resume_parsed.txt
│   └── sample_resume.pdf
├── scripts/
│   ├── fetch_jobs.py
│   ├── parse_resume.py
├── main.py
├── requirements.txt
├── README.md
└── .env


