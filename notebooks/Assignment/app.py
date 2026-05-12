import streamlit as st
import fitz
import io
import re
import numpy as np
from docx import Document
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
 
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
 
 
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
 
  
def extract_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""
    sentences = []
    pages_data = []
 
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        full_text += text + "\n"
        blocks = page.get_text("dict")["blocks"]
        pages_data.append({"page": page_num, "blocks": blocks})
        for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
            if len(sent.split()) > 4:
                sentences.append((sent, page_num))
 
    return full_text, pages_data, sentences
 
 
def extract_docx(file_bytes):
    from docx import Document
    import io
    doc = Document(io.BytesIO(file_bytes))
    full_text = []
    headings = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text: continue
        
        # 1. Check official style name
        style_name = para.style.name if para.style else ""
        
        # 2. Check for "Manual" Headings (Bold + Short Text)
        is_bold = all(run.bold for run in para.runs if run.text.strip())
        is_short = len(text) < 60
        
        if "Heading" in style_name or (is_bold and is_short):
            headings.append(text)
            full_text.append(f"### {text}") # Mark as heading for the analyzer
        else:
            full_text.append(text)
            
    return "\n".join(full_text), headings
  
def summarize(full_text):
    parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    total = len(list(parser.document.sentences))
    count = max(5, int(total * 0.15))
    result = summarizer(parser.document, count)
    return " ".join(str(s) for s in result), total, count
 
  
def extract_headings_pdf(pages_data):
    from collections import Counter
    all_sizes = []
    for page in pages_data:
        for block in page["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        all_sizes.append(round(span["size"], 1))
 
    if not all_sizes:
        return []
 
    body_size = Counter(all_sizes).most_common(1)[0][0]
    headings = []
    seen = set()
 
    for page in pages_data:
        for block in page["blocks"]:
            for line in block.get("lines", []):
                text = " ".join(s["text"] for s in line.get("spans", [])).strip()
                if not text or text in seen or len(text.split()) > 12:
                    continue
                sizes = [s["size"] for s in line.get("spans", []) if s["text"].strip()]
                if not sizes:
                    continue
                max_size = max(sizes)
                if max_size >= body_size + 1.5:
                    level = 1 if max_size >= body_size + 6 else 2 if max_size >= body_size + 2 else 3
                    headings.append({"level": level, "text": text, "page": page["page"]})
                    seen.add(text)
 
    return headings
  
@st.cache_data
def get_embeddings(sent_texts):
    return load_model().encode(list(sent_texts))
 
 
def search(query, sentences, embeddings, top_k):
    query_vec = load_model().encode([query])
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"rank": i + 1, "score": round(float(scores[idx]), 3),
             "text": sentences[idx][0], "page": sentences[idx][1] or "—"}
            for i, idx in enumerate(top_indices)]
 
 
# UI 
 
st.title("Document Analyzer")
 
uploaded = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])
 
if not uploaded:
    st.stop()
 
file_bytes = uploaded.read()
file_type = uploaded.name.split(".")[-1].lower()
 
if file_type == "pdf":
    full_text, pages_data, sentences = extract_pdf(file_bytes)
    is_pdf = True
else:
    full_text, pages_data, sentences = extract_docx(file_bytes)
    is_pdf = False
 
st.write(f"**{uploaded.name}** — {len(full_text.split()):,} words, {len(sentences)} sentences")
st.divider()
 
col1, col2, col3 = st.columns(3)
with col1:
    btn1 = st.button("Summarize", use_container_width=True)
with col2:
    btn2 = st.button("Extract Headings", use_container_width=True)
with col3:
    btn3 = st.button("Semantic Search", use_container_width=True)
 
if btn1:
    st.session_state["panel"] = "summarize"
elif btn2:
    st.session_state["panel"] = "headings"
elif btn3:
    st.session_state["panel"] = "search"
 
panel = st.session_state.get("panel")
 
st.divider()
 
if panel == "summarize":
    st.subheader("Summary")
    with st.spinner("Summarizing..."):
        summary, total, count = summarize(full_text)
    st.caption(f"{count} sentences selected out of {total} ({round(count/total*100, 1)}% coverage)")
    st.write(summary)
 
elif panel == "headings":
    st.subheader("Headings & Table of Contents")
    headings = extract_headings_pdf(pages_data) if is_pdf else [
        {"level": h["level"], "text": h["text"], "page": "—"} for h in pages_data
    ]
    if not headings:
        st.warning("No headings detected.")
    else:
        for h in headings:
            indent = "　" * (h["level"] - 1)
            label = {1: "H1", 2: "H2", 3: "H3"}.get(h["level"], "H?")
            st.write(f"{indent}**[{label}]** {h['text']}  *(p. {h['page']})*")
 
elif panel == "search":
    st.subheader("Semantic Search")
    query = st.text_input("Enter a keyword or sentence:")
    top_k = st.slider("Results", 1, 10, 5)
    if query.strip():
        with st.spinner("Searching..."):
            sent_texts = tuple(s[0] for s in sentences)
            embeddings = get_embeddings(sent_texts)
            results = search(query, sentences, embeddings, top_k)
        for r in results:
            st.markdown(f"**#{r['rank']}** · score `{r['score']}` · page `{r['page']}`")
            st.write(r["text"])
            st.divider()