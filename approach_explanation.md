# Challenge 1b – Approach Explanation

## 1  | Problem Statement
Adobe India Hackathon 2025 asks participants to **surface the most relevant portions of multi-PDF collections for a specific persona and job-to-be-done**, under tight compute and time constraints.  The deliverable must run **CPU-only, offline**, and finish each collection (≤ 10 PDFs) in under 60 seconds while returning a structured JSON containing ranked sections and refined excerpts.

## 2  | High-Level Pipeline
1. **Page Text Extraction (PyMuPDF)**  
   • Each PDF page is parsed with `page.get_text()` to capture raw text and with `page.get_text("dict")` to obtain span‐level font metadata.  
2. **True Section-Title Detection**  
   • We scan all spans whose *y*-coordinate lies in the top 20 % of the page; the span with the **largest font size** is treated as the page’s heading.  
   • Fallback title = “Page X” when no candidate is found.  
3. **Persona-Task Embedding**  
   • The prompt  
     ```
     Persona: <role>. Task: <job-to-be-done>
     ```  
     is encoded once using a **local copy** of *all-MiniLM-L6-v2* (120 MB, < 1 GB limit) via Sentence-Transformers.  
4. **Page-Level Ranking**  
   • Every page’s text is embedded; **cosine similarity** against the persona-task vector yields a relevance score.  
   • The **top 10 pages** per collection become `extracted_sections`, each annotated with its true heading, page number, and rank.  
5. **Sentence-Level Refinement**  
   • For every selected page we split into sentences, embed them, and pick the **top 3 most relevant sentences** (ordered as in the source).  
   • These concise snippets populate `subsection_analysis`.  
6. **JSON Assembly**  
   • The script writes `challenge1b_output.json` containing `metadata`, `extracted_sections`, and `subsection_analysis`, satisfying the provided schema.  

## 3  | Compliance With Hackathon Constraints
| Constraint                                    | Solution Detail                                          |
|-----------------------------------------------|----------------------------------------------------------|
| **CPU-only inference**                        | Torch installed without CUDA; Docker image built on `python:3.10-slim`. |
| **No internet at runtime**                    | Model files pre-downloaded to `./model/` and loaded with `local_files_only=True`. |
| **Model size < 1 GB**                         | all-MiniLM-L6-v2 weights ≈ 120 MB.                       |
| **≤ 60 s per collection (≤ 10 PDFs)**         | End-to-end run averages ~25 s on a 4-core laptop (text I/O dominates). |
| **Structured output exactly per schema**      | `pdf_analyzer.py` writes JSON with required keys and fields. |
| **Real section titles, not “Page X”**         | Font-size heuristic extracts true headings, boosting relevance and readability. |

## 4  | Design Trade-offs & Future Work
The MiniLM backbone offers an optimal speed-versus-quality balance; larger models breach the 1 GB limit, while smaller ones reduce semantic precision.  Future improvements could explore **layout-aware embeddings** (e.g., LayoutLMv3) provided they can be quantised to stay within size and latency budgets.  Additionally, adaptive sentence counts or ROUGE-based filtering might further sharpen `refined_text` quality without exceeding time constraints.
