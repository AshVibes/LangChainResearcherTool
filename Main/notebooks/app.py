import os
import streamlit as st

# LangChain imports (modern package layout)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Simple loader for URL or text
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="LangChain Researcher Tool", layout="wide")

# ---------- Helpers ----------
def load_text_from_url(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        # naive: join paragraph text
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return "\n\n".join(paragraphs)
    except Exception as e:
        return f"[Error loading URL] {e}"

def create_vectorstore_from_text(text: str, openai_api_key: str):
    # split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # embeddings
    os.environ["OPENAI_API_KEY"] = openai_api_key
    embeddings = OpenAIEmbeddings()

    # create FAISS index from chunks
    vectordb = FAISS.from_texts(chunks, embeddings)
    return vectordb

# ---------- UI ----------
st.title("LangChain Researcher Tool — Streamlit")

# API Key input (use secrets in production)
api_key = st.text_input("OpenAI API Key (paste here or set as env var)", type="password")
if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        st.warning("Please provide your OpenAI API key to use embeddings/LLM.")

mode = st.radio("Input type", ("Paste text", "Enter URL"))
input_content = st.text_area("Paste text or URL here", height=200)

if st.button("Prepare Document"):
    if not input_content:
        st.error("Paste a URL or text first.")
    else:
        with st.spinner("Loading and preparing..."):
            if mode == "Enter URL" or input_content.startswith("http"):
                text = load_text_from_url(input_content.strip())
            else:
                text = input_content
            if not text or text.startswith("[Error"):
                st.error("Could not load text. See below.")
                st.write(text)
            else:
                st.success("Text loaded. Creating vector store (this uses OpenAI embeddings).")
                vectordb = create_vectorstore_from_text(text, api_key)
                # save vectorstore to session_state
                st.session_state["vectordb"] = vectordb
                st.session_state["text_preview"] = text[:5000]
                st.write("Document prepared. Preview (first 5000 chars):")
                st.text_area("Preview", st.session_state["text_preview"], height=300)

st.markdown("---")
if "vectordb" in st.session_state:
    query = st.text_input("Ask a question about the prepared document")
    if query:
        with st.spinner("Running retrieval + LLM..."):
            llm = OpenAI(model="gpt-3.5-turbo")  # change model if you wish
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state["vectordb"].as_retriever())
            answer = qa.run(query)
            st.markdown("*Answer:*")
            st.write(answer)
