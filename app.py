import streamlit as st
import os
import time
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

def initialize_directories():
    for directory in ['pdfFiles', 'vectorDB']:
        if not os.path.exists(directory):
            os.makedirs(directory)

def initialize_session_variables():
    if 'template' not in st.session_state:
        st.session_state['template'] = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

        Context: {context}
        History: {history}

        User: {question}
        Chatbot:"""

    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=st.session_state['template'],
        )

    if 'memory' not in st.session_state:
        st.session_state['memory'] = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            input_key="question",
        )

    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = Chroma(
            persist_directory='vectorDB',
            embedding_function=OllamaEmbeddings(base_url='http://localhost:11434', model="llama3")
        )

    if 'llm' not in st.session_state:
        st.session_state['llm'] = Ollama(
            base_url="http://localhost:11434",
            model="llama3",
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

def upload_and_process_pdf():
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        file_path = f'pdfFiles/{uploaded_file.name}'
        if not os.path.exists(file_path):
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            process_pdf(file_path)

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    all_splits = text_splitter.split_documents(data)

    st.session_state.vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=OllamaEmbeddings(model="llama3")
    )
    st.session_state.vectorstore.persist()

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    initialize_qa_chain()

def initialize_qa_chain():
    if 'retriever' in st.session_state and 'llm' in st.session_state:
        st.session_state['qa_chain'] = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

def handle_chat_interaction():
    user_input = st.text_input("Ask me something about the PDF content:", key="user_input", on_change=handle_chat_interaction)
    if user_input:
        response = st.session_state.qa_chain(user_input)
        
        user_message = {"role": "user", "content": user_input}
        st.session_state.chat_history.append(user_message)
        
        chatbot_response = response['result']
        chatbot_message = {"role": "assistant", "content": chatbot_response}
        st.session_state.chat_history.append(chatbot_message)
        
        for message in st.session_state.chat_history:
            if message['role'] == "user":
                st.markdown("**You:** " + message['content'])
            else:
                st.markdown("**Assistant:** " + message['content'])

def main():
    st.title("Chat with your Files - Llama 3 8b")
    initialize_directories()
    initialize_session_variables()

    upload_and_process_pdf()

    if 'qa_chain' in st.session_state:
        handle_chat_interaction()
    else:
        st.write("Please upload a PDF file to start the chatbot.")

if __name__ == "__main__":
    main()