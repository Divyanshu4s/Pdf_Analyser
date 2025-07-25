import os
import json
import re
import time
import fitz  # PyMuPDF
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class PDFAnalyzer:
   def __init__(self, model_path="./model"):
    self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)  
    self.model = AutoModel.from_pretrained(model_path, local_files_only=True) 
        self.model.eval()  # Disable dropout for inference
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, texts):
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=256,
            return_tensors='pt'
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.numpy()

    def extract_page_text(self, file_path):
        try:
            doc = fitz.open(file_path)
            pages = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                # Extract real section title by font size & position
                info = page.get_text("dict")
                page_height = page.rect.height
                title_candidate = ""
                max_size = 0.0
                for block in info.get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            # only consider spans in upper 20% of page
                            if span["bbox"][1] < page_height * 0.2:
                                if span["size"] > max_size:
                                    max_size = span["size"]
                                    title_candidate = span["text"].strip()
                pages.append({
                    "text": text,
                    "page_number": page_num + 1,
                    "section_title": title_candidate or f"Page {page_num+1}"
                })
            return pages
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def rank_sections(self, pages, task_embedding):
        if not pages:
            return []
        texts = [page["text"] for page in pages]
        embs = self.get_embeddings(texts)
        sims = cosine_similarity([task_embedding], embs)[0]
        idxs = np.argsort(sims)[::-1]
        return [(pages[i], sims[i]) for i in idxs]

    def refine_content(self, text, task_embedding):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        if not sentences:
            return ""
        emb_sents = self.get_embeddings(sentences)
        sims = cosine_similarity([task_embedding], emb_sents)[0]
        top_idxs = np.argsort(sims)[::-1][:3]
        top_idxs.sort()
        return " ".join([sentences[i] for i in top_idxs])

    def process_collection(self, collection_path):
        input_file = os.path.join(collection_path, "challenge1b_input.json")
        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            return
        with open(input_file, 'r') as f:
            data = json.load(f)

        persona = data["persona"]["role"]
        task = data["job_to_be_done"]["task"]
        task_str = f"Persona: {persona}. Task: {task}"
        task_emb = self.get_embeddings([task_str])[0]

        all_pages = []
        pdfs_dir = os.path.join(collection_path, "PDFs")
        for doc in data["documents"]:
            path = os.path.join(pdfs_dir, doc["filename"])
            if not os.path.exists(path):
                print(f"PDF not found: {path}")
                continue
            pages = self.extract_page_text(path)
            for p in pages:
                p["document"] = doc["filename"]
            all_pages.extend(pages)

        ranked = self.rank_sections(all_pages, task_emb)[:10]
        extracted_sections = []
        subsection_analysis = []

        for rank, (page, score) in enumerate(ranked, start=1):
            extracted_sections.append({
                "document": page["document"],
                "section_title": page["section_title"],
                "importance_rank": rank,
                "page_number": page["page_number"]
            })
            refined = self.refine_content(page["text"], task_emb)
            subsection_analysis.append({
                "document": page["document"],
                "refined_text": refined,
                "page_number": page["page_number"]
            })

        output = {
            "metadata": {
                "input_documents": [d["filename"] for d in data["documents"]],
                "persona": persona,
                "job_to_be_done": task,
                "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }

        out_file = os.path.join(collection_path, "challenge1b_output.json")
        with open(out_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Output saved to: {out_file}")

def find_collections(base_dir):
    collections = []
    for entry in os.listdir(base_dir):
        full = os.path.join(base_dir, entry)
        if os.path.isdir(full):
            if os.path.exists(os.path.join(full, "challenge1b_input.json")) and \
               os.path.exists(os.path.join(full, "PDFs")):
                collections.append(full)
    return collections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PDF Analyzer with Section Title Extraction')
    parser.add_argument('--base_dir', type=str, default='.', help='Base folder containing collections')
    parser.add_argument('--collection', type=str, help='Specific collection to process')
    args = parser.parse_args()

    analyzer = PDFAnalyzer(model_path="model")
    if args.collection:
        analyzer.process_collection(args.collection)
    else:
        cols = find_collections(args.base_dir)
        if not cols:
            print("No valid collections found.")
        for col in cols:
            print(f"Processing: {os.path.basename(col)}")
            analyzer.process_collection(col)
