# ON PAUSE (V2 will be available after AI-Resume-Generator is done)
# AI-Resume-Matching-Backend
Backend for AI-Based Resume Job Matching and Skill Gap Detection Using Transformer Embeddings and Cosine Similarity

A **FastAPI-based backend** for AI-driven **Resume–Job Matching** and **Skill Gap Detection** using **SBERT Transformer Embeddings** and **FAISS indexing**.

---

## Manual Assembly

Large binary files and private keys are excluded for security and performance. Set up the following manually:

### 1. AI Models
#### Automatic Download (Online)
- Simply create a folder named models in the root directory.
- The first time you run python main.py, the system will detect the missing files and automatically download the all-MiniLM-L6-v2 model into that folder.

#### Online (Manual)
- Create a folder named `models` in the root directory.
- Download the `all-MiniLM-L6-v2` model.
- Place it in:
  ```
  ./models/all-MiniLM-L6-v2
  ```

### 2. Poppler Binaries 
#### Zip File (Offline)
- Extract poppler-25.12.0.zip into the root directory.
- Ensure the following path exists:
  ```
  ./poppler-25.12.0/Library/bin
  ```

#### Online (Manual)
- Download **Poppler for Windows**.
- Extract it into the root directory.
- Ensure the following path exists:
  ```
  ./poppler-25.12.0/Library/bin
  ```

### 3. Firebase (To be implemented)
- If using cloud features:
  - Add your Firebase service key file:
  ```
  ./serviceAccountKey.json
  ```

### 4. Certificates (To be implemented)
- If using cloud features:
  - Add your SSL certificates:
  ```
  ./cert.pem
  ./key.pem
  ```

---

## Steps to Run

### 1. Clone the Repository
```bash
git clone https://github.com/jerohalili/Resume-Matching-Backend
cd Resume-Matching-Backend
```

### 2. Create and Activate Virtual Environment

**PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Execution

### Start the Backend Server
```bash
python main.py
```

---

## API Documentation

Once the server is running, open:
```
http://127.0.0.1:8000/docs
```

- Interactive Swagger UI for testing endpoints

---

## Main Endpoint

### `POST /validate-any-resume`

**Description:**
- Accepts:
  - PDF or Image file (resume)
  - JSON string of expected skills

**Purpose:**
- Matches resume content against required skills
- Identifies missing or weak skill areas (To be implemented)

---

## Core Stack

- **FastAPI** — High-performance backend framework for the API layer  
- **EasyOCR** — Python-based OCR engine used to extract text from resumes  
- **SBERT (Sentence Transformers)** — NLP engine that converts text into semantic vector embeddings  
- **FAISS** — Facebook AI Similarity Search for high-speed vector indexing  
- **RapidFuzz** — Fast string matching library for granular skill validation  
- **Uvicorn** — Lightning-fast ASGI server that powers the backend

---

## Notes

- Ensure all manual dependencies are correctly placed before running.
- Model loading and vector indexing depend on correct directory structure.
- Poppler is required for PDF processing.

---
