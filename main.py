# voxbox venv

import streamlit as st
from langchain.llms import OpenAI
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain

st.title("VoxBox")

openai_api_key = st.secrets["OPENAI_API_KEY"]
PINECONE_ENV = st.secrets['fair_pinecone_env']
pinecone_api_key = st.secrets['ritter_pinecone_api']
pinecone_index = st.secrets["fair_pinecone_index"]

@st.cache_resource
def embedding_db():
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key = pinecone_api_key,
        environment=PINECONE_ENV
    )
    index = pinecone.Index(pinecone_index)
    vector_store = Pinecone(index, embeddings.embed_query, "text")

    return vector_store

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
vector_store = embedding_db()

def rag_answer(query):
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )
    result = qa_with_sources(query)
    st.info(result)

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))

with st.form('my_form'):
    text = st.text_area('Enter text: ', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        if len(text)>0:
            # generate_response(text)
            rag_answer(text)