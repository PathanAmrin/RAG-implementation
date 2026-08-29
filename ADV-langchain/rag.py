# A = Loading All Packages

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Gemini packages
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA


# B = Google Gemini API Key

import os
from dotenv import load_dotenv

load_dotenv()

my_gemini_api_key = os.getenv("GOOGLE_API_KEY")


# C = Creating RAG

# 1. Load Markdown File

loader = TextLoader(
    "documents\knowledge.md",
    encoding="utf-8"
)

documents = loader.load()


# 2. Split Markdown into Chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)


# 3. Create Gemini Embeddings and Store in FAISS

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=my_gemini_api_key
)

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)


# 4. Create Retriever

my_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# 5. Create RAG Chain

qa = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=my_gemini_api_key,
        temperature=0
    ),
    retriever=my_retriever
)


# 6. Ask Question

my_question = "what is the remote policy?"

result = qa.invoke({
    "query": my_question
})

print(result["result"])