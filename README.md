# Challenge 1b: Persona-Driven Document Intelligence

## Overview and Purpose

This solution processes multiple PDF document collections and extracts the most relevant sections based on a given persona and job-to-be-done, as required for the Adobe India Hackathon 2025. The system uses semantic similarity and font-based heuristics to identify and rank the most important document sections for specific user roles and tasks.

## Directory Structure

```
Challenge_1b/
├── README.md
├── approach_explanation.md
├── Dockerfile
├── requirements.txt
├── pdf_analyzer.py
├── download_model.py
├── .gitignore
├── model/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
├── Collection_1/
│   ├── PDFs/
│   ├── challenge1b_input.json
│   └── challenge1b_output.json
├── Collection_2/
│   ├── PDFs/
│   ├── challenge1b_input.json
│   └── challenge1b_output.json
└── Collection_3/
    ├── PDFs/
    ├── challenge1b_input.json
    └── challenge1b_output.json
```

## Setup Instructions

### Option 1: Python Virtual Environment

```bash
cd Challenge_1b
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Option 2: Docker (No Python Setup Needed)

```bash
docker build -t pdf-analyzer .
```

## Downloading the Local Transformer Model (one-time)

> Follow these steps **once** to place an offline copy of `sentence-transformers/all-MiniLM-L6-v2` in `./model/`.  
> Afterward the analyzer runs with **no internet**.

1. **Open a terminal in the project root**

   ```bash
   cd path/to/Challenge_1b          # adjust the path
   mkdir -p model                   # create the model folder if it does not exist
   ```

2. **Install the Hugging Face Hub helper (if not already)**

   ```bash
   pip install --upgrade huggingface_hub
   ```

3. **Run ONE Python script to download everything into `model/`**

   Create a file named `download_model.py` (or paste into a Python REPL) with:

   ```python
   from huggingface_hub import snapshot_download

   MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

   snapshot_download(
       repo_id=MODEL_ID,
       local_dir="model",                 # download directly into ./model
       local_dir_use_symlinks=False,      # make real copies instead of symlinks
       allow_patterns=[                   # grab only the files you need
           "pytorch_model.bin",           # ← weights (~120 MB)
           "*.json",                      # configs + tokenizer files
           "*.txt"                        # vocab.txt if present
       ]
   )

   print("✅  Model downloaded to ./model")
   ```

   Then run:

   ```bash
   python download_model.py
   ```

4. **Verify the folder now contains (at minimum):**

   ```
   model/
     config.json
     pytorch_model.bin          ←  ~120 MB
     tokenizer_config.json
     special_tokens_map.json
     vocab.txt                  ←  or tokenizer.json / merges.txt
   ```

5. **Point the code to this folder**

   In `pdf_analyzer.py` make sure the loaders are:

   ```python
   self.tokenizer = AutoTokenizer.from_pretrained("model", local_files_only=True)
   self.model     = AutoModel.from_pretrained("model", local_files_only=True)
   ```

That's it—no internet needed at runtime, and the model file is safely under the 1 GB limit.

## Run Commands

### To process a single collection:

```bash
python pdf_analyzer.py --collection "Collection_1"

# or with Docker:
docker run --rm -v "${PWD}/Collection_1:/data" pdf-analyzer --collection /data
```

### To process all collections:

```bash
python pdf_analyzer.py --base_dir .

# or with Docker:
docker run --rm -v "${PWD}:/data" pdf-analyzer --base_dir /data
```

## Hardware & Time Limits Compliance

| Requirement | This Solution | Status |
|-------------|---------------|--------|
| **CPU-only execution** | Uses PyTorch CPU backend, no CUDA dependencies | ✅ Compliant |
| **Model size < 1GB** | all-MiniLM-L6-v2 model (~120MB) | ✅ Compliant |
| **Processing time < 60s/collection** | Optimized pipeline averages ~25s per collection | ✅ Compliant |
| **No internet at inference** | Local model loading with `local_files_only=True` | ✅ Compliant |
| **Real section heading extraction** | Font-size based heuristic for true section titles | ✅ Compliant |
| **Structured JSON output** | Follows exact schema requirements | ✅ Compliant |

## Key Features

- **Semantic Ranking**: Uses sentence-transformers to rank document sections by relevance to persona and task
- **True Section Title Extraction**: Employs font-size analysis to extract real document headings, not just "Page X"
- **Sentence-level Refinement**: Provides top 3 most relevant sentences from each selected section
- **Offline Operation**: Complete local inference without internet dependencies
- **Docker Support**: Containerized for consistent deployment across environments
- **Multi-collection Processing**: Batch processing of multiple document collections

## Technical Architecture

1. **Text Extraction**: PyMuPDF for PDF parsing and font metadata analysis
2. **Embedding Generation**: Sentence-transformers with all-MiniLM-L6-v2 model
3. **Similarity Computation**: Cosine similarity for relevance scoring
4. **Section Ranking**: Top-10 most relevant pages per collection
5. **Content Refinement**: Sentence-level extraction for concise summaries

## Dependencies

- PyMuPDF (PDF processing)
- sentence-transformers (semantic embeddings)
- torch (neural network backend)
- scikit-learn (similarity computations)
- numpy (numerical operations)

## Troubleshooting

**Model Loading Issues:**
- Ensure `model/` directory contains all required files
- Verify `local_files_only=True` parameter is set
- Check internet connection during initial model download

**Performance Issues:**
- Monitor memory usage with large PDF collections
- Adjust batch sizes if encountering memory constraints
- Use Docker for consistent resource allocation

**Output Quality:**
- Verify input JSON schema matches expected format
- Check PDF quality and text extractability
- Review persona and task descriptions for clarity

---

