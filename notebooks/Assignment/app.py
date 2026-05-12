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
def load_nlp_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    combined_text = ""
    sentence_list = []
    structural_data = []

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        combined_text += page_text + "\n"
        blocks = page.get_text("dict")["blocks"]
        structural_data.append({"page": page_num, "blocks": blocks})
        raw_sentences = re.split(r"(?<=[.!?])\s+", page_text.strip())
        for s in raw_sentences:
            if len(s.split()) > 4:
                sentence_list.append((s, page_num))

    return combined_text, structural_data, sentence_list

def get_text_from_docx(docx_bytes):
    doc = Document(io.BytesIO(docx_bytes))
    combined_text = ""
    sentence_list = []
    heading_info = []

    for para in doc.paragraphs:
        clean_para = para.text.strip()
        if not clean_para:
            continue
            
        combined_text += clean_para + "\n"
        
        if para.style and hasattr(para.style, 'name') and para.style.name and para.style.name.startswith("Heading"):
            style_name = para.style.name
            level_match = re.search(r"\d+", style_name)
            level = int(level_match.group()) if level_match else 1
            heading_info.append({"level": level, "text": clean_para})
        else:
            raw_sentences = re.split(r"(?<=[.!?])\s+", clean_para)
            for s in raw_sentences:
                if len(s.split()) > 4:
                    sentence_list.append((s, None))

    return combined_text, heading_info, sentence_list

def run_summarizer(text_content):
    parser = PlaintextParser.from_string(text_content, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    
    all_sentences = list(parser.document.sentences)
    total_count = len(all_sentences)
    
    num_to_extract = int(total_count * 0.15)
    if num_to_extract < 5:
        num_to_extract = 5
        
    summary_result = summarizer(parser.document, num_to_extract)
    
    final_string = ""
    for sentence in summary_result:
        final_string += str(sentence) + " "
        
    return final_string, total_count, num_to_extract

def find_pdf_headings(doc_blocks):
    found_headings = []
    already_seen = set()

    for item in doc_blocks:
        page_no = item["page"]
        for block in item["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = line.get("spans", [])
                text_line = " ".join(s["text"] for s in spans).strip()
                
                if not text_line or text_line in already_seen or len(text_line.split()) > 15:
                    continue
                
                font_size = spans[0]["size"]
                
                if font_size >= 13: 
                    if font_size >= 18:
                        h_level = 1
                    elif font_size >= 15:
                        h_level = 2
                    else:
                        h_level = 3
                        
                    found_headings.append({
                        "level": h_level, 
                        "text": text_line, 
                        "page": page_no
                    })
                    already_seen.add(text_line)

    return found_headings

@st.cache_data
def create_embeddings(text_list):
    model = load_nlp_model()
    return model.encode(list(text_list))

def perform_search(user_query, original_sentences, doc_embeddings, top_n):
    model = load_nlp_model()
    query_vec = model.encode([user_query])
    
    scores = cosine_similarity(query_vec, doc_embeddings)[0]
    best_indices = np.argsort(scores)[::-1][:top_n]
    
    results_list = []
    for i, idx in enumerate(best_indices):
        results_list.append({
            "rank": i + 1,
            "confidence": round(float(scores[idx]), 3),
            "text": original_sentences[idx][0],
            "page": original_sentences[idx][1] if original_sentences[idx][1] else "N/A"
        })
    return results_list

st.set_page_config(page_title="AI Doc Analyzer")
st.title("📄 Document Analyzer")
st.write("Upload a file to summarize, extract structure, or search semantically.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

if uploaded_file is not None:
    file_data = uploaded_file.read()
    extension = uploaded_file.name.split(".")[-1].lower()
    
    if extension == "pdf":
        raw_text, structure, sentences = get_text_from_pdf(file_data)
        is_pdf = True
    else:
        raw_text, structure, sentences = get_text_from_docx(file_data)
        is_pdf = False

    st.info(f"File loaded: {uploaded_file.name} ({len(sentences)} sentences found)")

    task = st.radio("Choose a Task:", ["Summarize", "View Headings", "Search Inside"])
    st.divider()

    if task == "Summarize":
        st.subheader("Extractive Summary")
        with st.spinner("Processing text..."):
            summary_text, total, kept = run_summarizer(raw_text)
            st.write(summary_text)
            st.caption(f"Note: Kept {kept} sentences from a total of {total}.")

    elif task == "View Headings":
        st.subheader("Document Index")
        if is_pdf:
            results = find_pdf_headings(structure)
        else:
            results = [{"level": h["level"], "text": h["text"], "page": "—"} for h in structure]
        
        if not results:
            st.warning("No headings detected ")
        else:
            for item in results:
                space = "    " * (item["level"] - 1)
                st.markdown(f"{space}**L{item['level']}**: {item['text']} (p. {item['page']})")

    elif task == "Search Inside":
        st.subheader("Semantic Search")
        user_input = st.text_input("What are you looking for?")
        how_many = st.slider("Number of results", 1, 10, 3)
        
        if user_input:
            with st.spinner("Finding best matches..."):
                just_text = [s[0] for s in sentences]
                embeddings = create_embeddings(tuple(just_text))
                
                search_results = perform_search(user_input, sentences, embeddings, how_many)
                
                for res in search_results:
                    st.write(f"**Result #{res['rank']}** (Score: {res['confidence']}) - Page: {res['page']}")
                    st.write(res['text'])
                    st.markdown("---")