from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import VectorDBQA
from langchain_community.document_loaders import DirectoryLoader
import gradio as gr
import os

persist_dir = './db'
documents_dir = './documents'
embeddings = OpenAIEmbeddings()

# If we've already persisted the db to disk, load that
if os.path.exists(persist_dir):
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# Otherwise, make new Chroma DB
else:
    loader = DirectoryLoader(documents_dir)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_dir)

# Define object to send RAG queries to
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)

# Define gradio UI logic
def response(message, history):
    h = ''

    for d in history:
        h += 'User message: \'' + d[0] + '\', '
        h += 'Bot message: \'' + d[1] + '\' \n'

    m = 'You are an chatbot meant to answer participant questions about TreeHacks, a hackathon. Here is the prior message history: \n' + h + '\nHere is the message you have just been given: ' + message
    yield qa.invoke(m)

# Launch gradio UI
gr.ChatInterface(response).launch()
