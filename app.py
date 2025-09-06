# rag_chroma_app_multiuser_cleanup_cloud.py
import streamlit as st
import os
import time
from newspaper import Article
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import chromadb
import io
import PyPDF2
import uuid

# -------------------------------
# CONFIG
# -------------------------------

API_KEY = st.secrets.get("GENAI_API_KEY")
if not API_KEY:
    st.error("Please set your GENAI_API_KEY in Streamlit secrets.")
    st.stop()

genai.configure(api_key=API_KEY)

EMBED_MODEL = "models/text-embedding-004"
GEN_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
DB_CLEANUP_DAYS = 7  # delete DB files older than 7 days

# -------------------------------
# Cleanup old user DBs
# -------------------------------
def cleanup_old_dbs(days=DB_CLEANUP_DAYS):
    now = time.time()
    for f in os.listdir("."):
        if f.startswith("chromastore_") and f.endswith(".db"):
            file_age_days = (now - os.path.getmtime(f)) / (24 * 3600)
            if file_age_days > days:
                os.remove(f)
                print(f"Deleted old DB: {f}")

cleanup_old_dbs()

# -------------------------------
# Ask user for username/email
# -------------------------------
user_id = st.sidebar.text_input("Enter your username/email to start:")
if not user_id:
    st.warning("Please enter a username/email to use the app.")
    st.stop()

# -------------------------------
# SQLite database per user
# -------------------------------
db_path = f"chromastore_{user_id}.db"
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection("documents")

# -------------------------------
# Helpers
# -------------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def embed_texts(texts):
    embeddings = []
    for t in texts:
        resp = genai.embed_content(
            model=EMBED_MODEL,
            content=t,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings.append(resp["embedding"])
    return embeddings

def add_document(source_name, text):
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metas,
    )
    st.success(f"Added {len(chunks)} chunks from {source_name}")

def retrieve_with_citations(query, k=TOP_K):
    q_emb = embed_texts([query])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas

# -------------------------------
# Sidebar: Add Documents
# -------------------------------
st.sidebar.header("Add Documents")
add_mode = st.sidebar.radio("Add by", ["URL", "PDF", "Paste Text"])
clear_docs = st.sidebar.checkbox("Clear your existing documents?", value=False)

if clear_docs:
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
        st.sidebar.info("Cleared your old documents.")

if add_mode == "URL":
    url = st.sidebar.text_input("Enter article URL:")
    if st.sidebar.button("Add URL"):
        try:
            article = Article(url)
            article.download()
            article.parse()
            add_document(url, article.text)
        except Exception as e:
            st.sidebar.error(f"Failed fetching {url}: {e}")

elif add_mode == "PDF":
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file and st.sidebar.button("Add PDF"):
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = "\n".join([p.extract_text() or "" for p in reader.pages if p.extract_text()])
            add_document(uploaded_file.name, text)
        except Exception as e:
            st.sidebar.error(f"Failed reading PDF: {e}")

elif add_mode == "Paste Text":
    pasted = st.sidebar.text_area("Paste text here")
    if st.sidebar.button("Add Pasted Text") and pasted.strip():
        add_document("Pasted Text", pasted)

# -------------------------------
# Tabs for features
# -------------------------------
tab1, tab2, tab3 = st.tabs(["🤖 Ask Questions", "📝 Summary", "☁️ WordCloud"])

# -------------------------------
# Tab 1: RAG Q&A
# -------------------------------
with tab1:
    st.subheader("Ask Questions (RAG + Citations)")
    question = st.text_input("Type your question:")

    if question:
        with st.spinner("Retrieving relevant contexts..."):
            docs, metas = retrieve_with_citations(question, k=TOP_K)

        if not docs:
            st.warning("No results found.")
        else:
            context = ""
            for d, m in zip(docs, metas):
                context += f"[Source: {m['source']} | chunk {m['chunk_index']}]\n{d}\n\n"

            try:
                prompt = (
                    "You are a helpful assistant. Answer the question using ONLY the context below. "
                    "Cite the sources when possible.\n\n"
                    f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                )
                model = genai.GenerativeModel(GEN_MODEL)
                resp = model.generate_content(prompt)
                st.subheader("Answer")
                st.write(resp.text)

                st.subheader("Citations")
                for d, m in zip(docs, metas):
                    clean_text = " ".join(d.split())
                    st.markdown(f"- **{m['source']} (chunk {m['chunk_index']})**: {clean_text[:300]}...")
            except Exception as e:
                st.error(f"Error generating answer: {e}")

# -------------------------------
# Tab 2: Summary
# -------------------------------
with tab2:
    st.subheader("Corpus Summary")
    if st.button("Generate Summary"):
        all_docs = collection.get()
        if not all_docs["documents"]:
            st.warning("No documents added yet.")
        else:
            combined = "\n\n".join(all_docs["documents"])
            try:
                prompt = (
                    "Summarize the following documents into 5–7 concise bullet points:\n\n"
                    f"{combined}"
                )
                model = genai.GenerativeModel(GEN_MODEL)
                resp = model.generate_content(prompt)
                st.markdown("**Summary:**")
                st.write(resp.text)
            except Exception as e:
                st.error(f"Error generating summary: {e}")

# -------------------------------
# Tab 3: WordCloud
# -------------------------------
with tab3:
    st.subheader("Corpus Keyword WordCloud")
    try:
        all_docs = collection.get()
        if all_docs["documents"]:
            text_all = " ".join(all_docs["documents"])
            vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
            vectorizer.fit_transform([text_all])
            keywords = vectorizer.get_feature_names_out()
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(keywords))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.info("No documents added yet.")
    except Exception as e:
        st.error(f"Error generating WordCloud: {e}")
