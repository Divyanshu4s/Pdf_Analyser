FROM python:3.10-slim

# System deps for PyMuPDF
RUN apt-get update && apt-get install -y build-essential libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .    # pdf_analyzer.py, model/, collections, etc.

ENTRYPOINT ["python", "pdf_analyzer.py"]
CMD ["--help"]
