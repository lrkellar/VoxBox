import os
from langchain.text_splitter import CharacterTextSplitter 
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
import streamlit as st
import pickle

PDF_FOLDER = 'data/pdfs/'
DB_FOLDER = 'data/db/'
os.makedirs(DB_FOLDER, exist_ok=True) 

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]

OUTPUT_DIR = "data\db"

def save_output(output, filename):
    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(output, f)

pdf_files = [
    r"data\pdfs\Emergency Maintenance SOP.pdf",
    r"data\pdfs\Kascha's Operations Processes.pdf",
    r"data\pdfs\Operations SOP .pdf"]
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(api_key=st.secrets['OPENAI_API_KEY'])
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

raw_text = get_pdf_text(pdf_files)
text_chunks = get_text_chunks(raw_text)
vectorstore = get_vectorstore(text_chunks)
vectorstore.save_local("faiss_index")
