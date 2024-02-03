# voxbox venv

# sqlite3 specification for streamlit cloud
commited = 1
if commited == 1:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

debug = 1

import streamlit as st
from langchain_community.llms import OpenAI
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
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
import time
from  icecream import ic
import re
from langchain.chains import create_citation_fuzzy_match_chain


st.title("VoxBox")
st.text("A minimal interface to an AI with domain knowledge - RAG AI")
mode = "SOP Citations"
openai_api_key = st.secrets["OPENAI_API_KEY"]
PINECONE_ENV = st.secrets['fair_pinecone_env']
pinecone_api_key = st.secrets['ritter_pinecone_api']
pinecone_index = st.secrets["fair_pinecone_index"]

def streamlit_debug_window(error_message : str):
    with st.spinner(error_message):
        time.sleep(3)  # Delay for 3 seconds

# Clear the spinner and error message after the delay
st.empty()

with st.sidebar:
    st.write("Please select what kind of knowledge you'd like the AI to have")
    mode = st.radio('Choose Mode: ', ["SOP Citations", "Fair Housing","SOP", "-Cited SOP- Under development"] )

if mode == "Fair Housing":
    @st.cache_resource
    def embedding_db():
        if debug == 1:
            streamlit_debug_window("embedding_db called")
        embeddings = OpenAIEmbeddings()
        pinecone.init(
            api_key = pinecone_api_key,
            environment=PINECONE_ENV
        )
        index = pinecone.Index(pinecone_index)
        vector_store = Pinecone(index, embeddings.embed_query, "text")

        return vector_store

#if mode == "SOP" or mode == "SOP Citations":
@st.cache_resource
def chroma_hookup():
    print("Chroma hookup called")
    if debug == 1:
        streamlit_debug_window("Chromahookup called")

    embedding_function = OpenAIEmbeddings()
    vector_store = Chroma(
        persist_directory="db", 
        embedding_function=embedding_function
    )
    return vector_store
vectordb = chroma_hookup()

@st.cache_resource
def db_prep(texts, persist_directory = "db"):
    embedding = OpenAIEmbeddings(api_key=st.secrets['OPENAI_API_KEY'])
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    return vectordb

def rag_answer(query, vector_store):
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )
    result = qa_with_sources.invoke(query)
    return(result)

def chroma_rag_answer(query, vector_store):
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = vector_store.as_retriever(search_type="mmr")
    )
    result = qa_with_sources.invoke(query)
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

def get_conversation_chain_and(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents = True
    )
    with st.text(ic(conversation_chain)):
        time.sleep(10)
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
            
def handle_userinput2(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    # Three columns with different widths
    col1, col2, col3 = st.columns([3,1,1])
    # col1 is wider

    # Using 'with' notation:
    with col1:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    with col2:
        st.text(response.keys())
        st.text(response['context'])

def cleaner(inputa : str) -> str:
    regex = r".*\\(.*)"

    match = re.search(regex, inputa) # Access the first group (entire match)
    return match

def highlight(text, span):
    return (
        "..."
        + text[span[0] - 20 : span[0]]
        + "*"
        + "\033[91m"
        + text[span[0] : span[1]]
        + "\033[0m"
        + "*"
        + text[span[1] : span[1] + 20]
        + "..."
    )

def unpack_answer(incoming):
    staging = ""
    for x in range(len(incoming['text'].answer)):
        stage2 = incoming['text'].answer[x].fact
        staging = f"{staging} \n\n {x+1}. {stage2}"
    return staging

def unpack_citations(incoming):
    staging = ""
    for x in range(len(incoming['text'].answer)):
        stage2 = incoming['text'].answer[x].substring_quote
        staging = f"{staging}  \n\n {x+1}:{stage2}"
    return staging

def citation_chain(question, context, debug_mode = 0):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", api_key=st.secrets['OPENAI_API_KEY'])


    chain = create_citation_fuzzy_match_chain(llm)

    result2 = chain.invoke({'question': question, 'context' : context})
    result_staging = result2['text']
    ic(result_staging)
    #ic(result2)
    return(result2)


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
    user_question = st.text_input("I am the on call technician. What do I do about a leak?:")
    if user_question:
        handle_userinput(user_question)

if mode == "-Cited SOP- Under development":
    with st.form('my_form'):
        # Database Setup
        persist_directory = 'db'
        embedding = OpenAIEmbeddings()
        vector_store = chroma_hookup()
        text = st.text_area('Enter text: ', 'I am the on call technician. What do I do about a leak?')
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

if mode == "SOP Citations":
    vectordb = db_prep()
    print("db_prep run")
    db_check = vectordb.get()
    ic(db_check)
    conversation = ["Welcome to your SOP guide"]
    chat_window = st.text(conversation)

    text_input = st.text_input(label="What would you like help with?",value="....")


    if text_input:
        query = text_input
        context = vectordb.similarity_search(query)
        results = citation_chain(question=query, context=context)
        citations = unpack_citations(results)

        st.subheader("Results")
        st.markdown(unpack_answer(results))
        tab1, tab2 = st.tabs(["Citations", "Sources"])
        tab1.write(citations)

        if 'context' in results and len(results['context']) > 1:
            source1_raw = results['context'][1].metadata['source']
            source1 = cleaner(source1_raw)
            tab2.write(source1[1])

        # st.markdown("[google](www.google.com)") # Example of markdown hyperlink