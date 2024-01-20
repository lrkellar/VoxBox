# voxbox venv

import streamlit as st
from langchain_community.llms import OpenAI
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma

# sqlite3 specification for streamlit cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

st.title("VoxBox")
st.text("A minimal interface to an AI with domain knowledge - RAG AI")
mode = "Fair Housing"
openai_api_key = st.secrets["OPENAI_API_KEY"]
PINECONE_ENV = st.secrets['fair_pinecone_env']
pinecone_api_key = st.secrets['ritter_pinecone_api']
pinecone_index = st.secrets["fair_pinecone_index"]


with st.sidebar:
    st.write("Please select what kind of knowledge you'd like the AI to have")
    mode = st.radio('Choose Mode: ', ["Fair Housing","SOP", "Cited SOP"] )

if mode == "Fair Housing":
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

if mode == "SOP":
    @st.cache_resource
    def chroma_hookup():
        embedding_function = OpenAIEmbeddings()
        vector_store = Chroma(
            persist_directory="db", 
            embedding_function=embedding_function
        )
        return vector_store

def rag_answer(query, vector_store):
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )
    result = qa_with_sources(query)
    return(result)

def chroma_rag_answer(query, vector_store):
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = vector_store.as_retriever(search_type="mmr")
    )
    result = qa_with_sources(query)
    return(result)

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))


def get_conversation_chain_og(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_response(conversation, question):
    result = conversation.predict(question)
    answer = result["answer"]
    docs = result.pop("source_documents")
    
    return answer, docs

from htmlTemplates import css, bot_template, user_template
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

if mode == "Fair Housing":
    with st.form('my_form'):
        text = st.text_area('Enter text: ', 'What questions do you have about fair housing law in Indiana?')
        submitted = st.form_submit_button('Submit')
        if submitted:
            if len(text) > 0:
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

if mode == "SOP":

    # Database Setup
    persist_directory = 'db'
    embedding = OpenAIEmbeddings()
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    # create conversation chain
    st.session_state.conversation = get_conversation_chain(vector_store)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("SOP discussion")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

if mode == "Cited SOP":
    with st.form('my_form'):
        # Database Setup
        persist_directory = 'db'
        embedding = OpenAIEmbeddings()
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding)

        text = st.text_area('Enter text: ', 'What questions do you have about SOPs?')
        submitted = st.form_submit_button('Submit')

        if submitted:
            if len(text) > 0:
                llm = ChatOpenAI(
                    openai_api_key=openai_api_key,
                    model_name='gpt-3.5-turbo',
                    temperature=0.2
                )
                # OG) vector_store = embedding_db()
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