
## Features

- **Summarize** — extracts the most representative 15% of sentences using the TextRank algorithm (no LLM required, runs fully offline)
- **View Headings** — builds a navigable document index by detecting headings from font sizes (PDF) or Word styles (DOCX)
- **Search Inside** — finds the most semantically relevant sentences to any query using sentence embeddings and cosine similarity

---

## How it works (overview)

```
Upload file (PDF or DOCX)
        ↓
Parse text + structure
        ↓
Choose a task:
  ├── Summarize   → TextRank scores sentences by mutual similarity → top 15% returned
  ├── Headings    → PDF: font size heuristic / DOCX: style tags → indented index
  └── Search      → Encode all sentences + query → cosine similarity → top N results
```

---

## Summarization — how it actually works

This app uses **extractive summarization**, meaning it picks real sentences from the document rather than generating new ones. No LLM is involved. The algorithm is called **TextRank**.

### Step 1 — Tokenization

The raw text is passed into Sumy's `PlaintextParser` with an English `Tokenizer`. NLTK handles two things here:

- **Sentence tokenization** — splits the full text into individual sentences
- **Word tokenization + stopword removal** — breaks sentences into words and strips common filler words like "the", "is", "and" that carry no meaning for comparison

### Step 2 — Build a sentence list

Every sentence becomes a node. At this point the algorithm has a flat list: `[S1, S2, S3 ... Sn]`.

### Step 3 — Build a similarity matrix

Every sentence is compared to every other sentence. Similarity is measured by **shared word overlap** — how many meaningful words two sentences have in common. This produces an N×N matrix where each cell holds a similarity score between a pair of sentences.

```
         S1    S2    S3    S4
S1  [  1.0   0.8   0.1   0.0 ]
S2  [  0.8   1.0   0.2   0.1 ]
S3  [  0.1   0.2   1.0   0.7 ]
S4  [  0.0   0.1   0.7   1.0 ]
```

### Step 4 — PageRank scoring

TextRank treats the similarity matrix as a graph and runs a **PageRank-style algorithm** on it — the same idea Google uses to rank web pages, but applied to sentences.

The rule: a sentence that is highly similar to many other sentences scores higher, because being "agreed with" by the rest of the document means it likely captures a central idea.

A sentence that expresses a unique idea not echoed elsewhere scores lower — even if that idea is important.

### Step 5 — Pick the top 15%

The highest-scoring sentences are selected. The code extracts 15% of the total sentence count, with a minimum of 5 sentences so very short documents still get a usable summary.

```python
num_to_extract = int(total_count * 0.15)
if num_to_extract < 5:
    num_to_extract = 5
```

### Step 6 — Output in original order

The selected sentences are returned in the **order they appear in the document**, not ranked by score. This makes the summary read naturally as flowing text rather than jumping around.

### What TextRank does NOT do

| It does not... | What it does instead |
|---|---|
| Count word frequency | Measures sentence-to-sentence similarity |
| Understand meaning | Compares surface-level word overlap |
| Generate new text | Extracts existing sentences verbatim |
| Use a neural network | Pure graph algorithm, no model weights |

### Limitation

Because TextRank relies on word overlap, it can miss sentences that express a key idea in unique language not echoed elsewhere in the document. Modern LLM-based summarizers understand meaning rather than just counting shared words, making them better for nuanced content — but they require an API and have token costs. TextRank is free, fast, and fully offline.

---

## PDF vs DOCX — key differences

These two file types are handled differently throughout the pipeline.

### Parsing

| | PDF | DOCX |
|---|---|---|
| Library | PyMuPDF (fitz) | python-docx |
| Unit of iteration | Page | Paragraph |
| Page numbers | Tracked per sentence | Not available (shown as N/A) |

### Heading detection

**PDF** has no concept of headings — the app guesses by looking at font size:

| Font size | Heading level |
|---|---|
| 18px or above | H1 |
| 15px or above | H2 |
| 13px or above | H3 |

This is a heuristic and can incorrectly flag large decorative text as headings.

**DOCX** reads Word's built-in style tags directly ("Heading 1", "Heading 2", etc.), which is semantically accurate and reliable.

### What the `structure` variable holds

Both file types return three values — `raw_text`, `structure`, `sentences` — but `structure` contains fundamentally different data:

```python
# PDF: raw font/position block data — needs find_pdf_headings() to process it
raw_text, structure, sentences = get_text_from_pdf(file_data)

# DOCX: already-processed heading list — just reformatted for display
raw_text, structure, sentences = get_text_from_docx(file_data)
```

This is why the View Headings task branches on `is_pdf`:

```python
if is_pdf:
    results = find_pdf_headings(structure)       # extra processing needed
else:
    results = [{"level": h["level"], ...} for h in structure]  # already done
```

### Sentence tokenization

Both use the same regex pattern to split sentences:

```python
re.split(r"(?<=[.!?])\s+", text)
```

The difference is what gets stored alongside each sentence:

```python
# PDF — sentence stored with its page number
sentence_list.append((sentence, page_num))

# DOCX — no page number available
sentence_list.append((sentence, None))
```

This flows all the way to search results — PDF shows a real page number, DOCX shows "N/A".

### Summarization

Identical for both — `raw_text` is a plain string by the time it reaches TextRank, so the algorithm has no awareness of whether the source was a PDF or DOCX.

---

## Semantic search — how it works

The search feature does not match keywords. It matches **meaning**.

1. Every sentence in the document is encoded into a 384-dimensional vector using the `all-MiniLM-L6-v2` sentence transformer model
2. The user's query is encoded into the same vector space using the same model
3. Cosine similarity is computed between the query vector and every sentence vector
4. The top N sentences with the highest similarity scores are returned

This means a search for "financial performance" can surface a sentence containing "revenue grew significantly" even though no words overlap — because the model understands they mean similar things.

Embeddings are cached per document using `@st.cache_data`, so changing the query or the results slider does not re-encode the entire document.

---

## Requirements

Python 3.9+

Install dependencies:

```bash
pip install streamlit pymupdf python-docx sumy sentence-transformers scikit-learn numpy nltk
```

Or with `uv`:

```bash
uv add streamlit pymupdf python-docx sumy sentence-transformers scikit-learn numpy nltk
```

---

## Running the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project structure

```
.
├── app.py          # Main application file
└── README.md       # This file
```

---

## Key libraries

| Library | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pymupdf` (fitz) | PDF parsing — text and font metadata |
| `python-docx` | DOCX parsing — paragraphs and heading styles |
| `sumy` | TextRank extractive summarization |
| `nltk` | Tokenization and stopword removal for Sumy |
| `sentence-transformers` | Sentence embeddings for semantic search |
| `scikit-learn` | Cosine similarity scoring |
| `numpy` | Sorting similarity scores |

---

## Notes

- The sentence embedding model (`all-MiniLM-L6-v2`, ~80MB) is downloaded on first run and cached automatically via `@st.cache_resource`
- Embeddings are cached per document via `@st.cache_data` — re-typing a search query does not re-encode the document
- PDF heading detection is heuristic and may pick up large decorative text that is not actually a heading
- DOCX heading detection reads Word style tags directly and is more reliable
- Page numbers appear in PDF search results; DOCX results show "N/A" since python-docx does not expose page boundaries
