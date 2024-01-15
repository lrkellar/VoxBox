# voxbox venv
from PyPDF2 import PdfReader
import streamlit as st
from langchain_community.llms import OpenAI
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

st.title("VoxBox")
st.text("A minimal interface to an AI with domain knowledge - RAG AI")
mode = 1
openai_api_key = st.secrets["OPENAI_API_KEY"]
PINECONE_ENV = st.secrets['fair_pinecone_env']
pinecone_api_key = st.secrets['ritter_pinecone_api']
pinecone_index = st.secrets["fair_pinecone_index"]

if mode == 1:
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

def rag_answer(query, vector_store):
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )
    result = qa_with_sources(query)
    return(result)

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


with st.form('my_form'):
    text = st.text_area('Enter text: ', 'What questions do you have about fair housing law in Indiana?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        if len(text) > 0:
            if mode == 1:
                llm = ChatOpenAI(
                    openai_api_key=openai_api_key,
                    model_name='gpt-3.5-turbo',
                    temperature=0.2
                )
                vector_store = embedding_db()
                response_dict = rag_answer(text, vector_store=vector_store)  # Store the entire response dictionary

                # Extract values from the dictionary
                question = response_dict['question']
                answer = response_dict['answer']
                sources = response_dict['sources']

                # Create a layout with columns for question/answer and sources
                col1, col2 = st.columns([3, 1])  # Adjust column widths as needed

                with col1:
                    st.subheader("Question:")
                    st.write(question)
                    st.subheader("Answer:")
                    st.write(answer, unsafe_allow_html=True)  # Allow HTML formatting if applicable

                with col2:
                    st.subheader("Sources:")
                    st.write(sources, unsafe_allow_html=True)  # Allow HTML formatting if applicable

            if mode == 3:
                embeddings = OpenAIEmbeddings()
                llm = ChatOpenAI(
                    openai_api_key=openai_api_key,
                    model_name='gpt-3.5-turbo',
                    temperature=0.2
                )
                vector_store = embedding_db()

                new_db = FAISS.load_local(r"data\faiss_index", embeddings)

                response_dict = rag_answer(text, vector_store=vector_store)  # Store the entire response dictionary

                # Extract values from the dictionary
                question = response_dict['question']
                answer = response_dict['answer']
                sources = response_dict['sources']

                # Create a layout with columns for question/answer and sources
                col1, col2 = st.columns([3, 1])  # Adjust column widths as needed

                with col1:
                    st.subheader("Question:")
                    st.write(question)
                    st.subheader("Answer:")
                    st.write(answer, unsafe_allow_html=True)  # Allow HTML formatting if applicable

                with col2:
                    st.subheader("Sources:")
                    st.write(sources, unsafe_allow_html=True)  # Allow HTML formatting if applicable
