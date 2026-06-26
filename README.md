# LLM-Powered Resume Extraction Pipeline

An end-to-end document AI pipeline that converts unstructured resumes (PDFs including scanned documents) into structured candidate profiles using OCR, LLMs, and schema validation.

Works on both text-based PDFs and image/scanned PDFs using OCR. Detects text and formats it into structured JSON output per PDF file using an LLM API (currently supporting Groq). Implemented resilient LLM communication with automatic retry and rate-limit recovery. Validated and normalized model outputs into a deterministic JSON schema.

Traditional regex-based resume parsers are brittle because resume layouts vary significantly. This project instead uses an LLM to semantically identify candidate information after extracting text, making it robust to varying document formats.

## Pipeline Architecture
```
                       PDF
                        │
        ┌───────────────┴─────────────────┐
        │                                 │
  Text-based PDF                     Scanned PDF
        │                                 │
     PyMuPDF                          OCR Engine
        │                                 │
        └───────────────┬─────────────────┘
                        │
                Text Normalization
                        │
                 LLM Information
                    Extraction
                        │
                 JSON Validation
                        │
              Structured Candidate
                     Profile
```

## Usage Guide

1- Install the `Tesseract OCR` engine:

- Ubuntu:
```bash
sudo apt update
sudo apt install tesseract-ocr
```

- Windows
```shell
winget install UB-Mannheim.TesseractOCR
```

2- For the first time running: run `pip install -r requirements.txt`

3- Rename your `.env.example` to `.env`

4- Add your Groq API key to `.env`

5- Place all your PDFs at `data/`

6- Run the pipeline:

- From the notebook:
    - Press `Run All` on the notebook to run all cells

**OR**

- Run the script:
    ```bash
    python pipeline.py
    ```


7- Your output will be at `data/` called `output [date].json`

## Output format

- JSON file containing:
    - File Name
    - File Path
    - Name
    - Email
    - Phone
    - Skills (list of strings)
    - Education (list of strings)
    - Work Experience (list of strings)

- Example:

```json
{
    "File Name": {
        "0": "Muhammed Esmail.pdf"
    },
    "File Path": {
        "0": "/PDF Resume Parser/data/Muhammed Esmail.pdf"
    },
    "Name": {
        "0": "Muhammed A. Esmail"
    },
    "Email": {
        "0": "muhammedesmail101@gmail.com"
    },
    "Phone": {
        "0": "01208665973"
    },
    "Skills": {
        "0": [
            "Problem Solving & Algorithmic Thinking 2013 Specialist rank on Codeforces.",
            "Competitive Programming: Active participant in ECPC, IEEE Extreme, and Codeforces contests.",
            ...
        ]
    },
    "Work Experience": {
        "0": [
            "AI Engineering Intern 2013 EVA Limited (Feb 2026 - Present)",
            "Secured internship with a Top 20 finish in the 2026 Evai Hackathon by building an AI Customer Support Agent.",
            ...
        ]
    },
    "Education": {
        "0": [
            "Cairo University 2013 Giza, Giza Governorate",
            "Bachelor of Computers and Artificial Intelligence 2013 General",
            "Rank 1 student: GPA 3.84",
            "Expected graduation: August 2028",
            ...
        ]
    }
}


```
* Note the `"0"` means the index of the PDF. This is an example of only one PDF processed, but generally there will be `"0"`, `"1"`, `"2"`, etc. The index of the PDF this information relates to. Depends on the order of processing.

- The JSON fomat is extremely versatile and allows importing the data back into python Pandas for more data processing.

## Possible Future Improvements

> More supported resume file formats (e.g. docx)

> Batch processing

> FastAPI REST API

> Docker support

> Confidence score for extracted fields

> Automatic evaluation against benchmark dataset