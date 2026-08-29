# 📚 RAG Project using LangChain, Google Gemini & FAISS

This project implements a simple **Retrieval-Augmented Generation (RAG)** system using:

* 🦜 LangChain
* 🤖 Google Gemini
* 🔎 FAISS Vector Database
* 🧠 Gemini Embeddings
* 📄 Markdown knowledge file

The system loads information from a Markdown file, splits it into smaller chunks, converts the chunks into vector embeddings, stores them in FAISS, retrieves the most relevant information, and then uses Google Gemini to generate an answer.

---

## 🏗️ Project Architecture

```text
Knowledge File
     │
     ▼
TextLoader
     │
     ▼
Document Splitting
     │
     ▼
RecursiveCharacterTextSplitter
     │
     ▼
Gemini Embeddings
     │
     ▼
FAISS Vector Store
     │
     ▼
Retriever
     │
     ▼
Relevant Documents
     │
     ▼
Gemini LLM
     │
     ▼
Final Answer
```

---

## 📁 Project Structure

```text
rag-project/
│
├── documents/
│   └── knowledge.md
│
├── rag.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

### File Description

| File                     | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| `documents/knowledge.md` | Contains the knowledge used by the RAG system            |
| `rag.py`                 | Main Python program                                      |
| `requirements.txt`       | Python dependencies                                      |
| `.env`                   | Stores the Google Gemini API key                         |
| `.gitignore`             | Prevents sensitive/unnecessary files from being uploaded |
| `README.md`              | Project documentation                                    |

---

# ⚙️ Technologies Used

### LangChain

LangChain is used to build the RAG pipeline and connect the document loader, text splitter, embeddings, vector store, retriever, and LLM.

### Google Gemini

Google Gemini is used for:

* Generating embeddings
* Generating the final answer

Embedding model:

```text
gemini-embedding-001
```

LLM:

```text
gemini-3.6-flash
```

### FAISS

FAISS is used as the vector store.

It stores document embeddings and performs similarity search to find relevant chunks for a user's question.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rag-project
```

Replace:

```text
<YOUR_GITHUB_REPOSITORY_URL>
```

with your GitHub repository URL.

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux / Ubuntu

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not been created yet, use:

```text
langchain
langchain-community
langchain-text-splitters
langchain-google-genai
langchain-classic
faiss-cpu
python-dotenv
```

Then install:

```bash
pip install -r requirements.txt
```

---

# 🔑 Google Gemini API Key

You need a Google Gemini API key.

Create a `.env` file in the project root:

```text
GOOGLE_API_KEY=your_google_api_key_here
```

Example:

```text
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXX
```

**Do not upload your API key to GitHub.**

---

# 🔒 .gitignore

Create a `.gitignore` file:

```text
venv/
.env
__pycache__/
*.pyc
```

This prevents your virtual environment, API key, and Python cache files from being uploaded to GitHub.

---

# 📄 Knowledge File

Create:

```text
documents/knowledge.md
```

Example:

```markdown
# Company Remote Work Policy

Employees can work remotely up to three days per week.

Remote work must be approved by the employee's manager.

Employees must be available during core working hours from 9 AM to 6 PM.

Employees working remotely must have a reliable internet connection.

Employees must follow all company security policies while working remotely.
```

---

# 🧠 How the RAG System Works

## Step 1: Load the Document

The project uses `TextLoader` to load the Markdown knowledge file.

```python
loader = TextLoader(
    "documents\knowledge.md",
    encoding="utf-8"
)

documents = loader.load()
```

The document is loaded into LangChain's document format.

---

## Step 2: Split the Document

The document is divided into smaller chunks:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)
```

### Why split the document?

Large documents cannot always be sent directly to the LLM.

Splitting allows the system to:

* Search smaller pieces of information
* Retrieve relevant content
* Reduce unnecessary context
* Improve retrieval accuracy

### Chunk Configuration

```text
chunk_size = 500
chunk_overlap = 100
```

This means each chunk can contain approximately 500 characters, with 100 characters overlapping with the next chunk.

---

# 🔢 Step 3: Create Embeddings

The project uses Google's Gemini embedding model:

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=my_gemini_api_key
)
```

An embedding converts text into a numerical vector.

For example:

```text
"Remote work policy"
        ↓
[0.023, -0.154, 0.762, ...]
```

These vectors allow the system to perform semantic similarity searches.

---

# 🗄️ Step 4: Store Embeddings in FAISS

```python
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)
```

FAISS stores the document embeddings and allows fast similarity searching.

---

# 🔍 Step 5: Create a Retriever

```python
my_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)
```

The value:

```text
k = 3
```

means the system retrieves the **3 most relevant document chunks** for the question.

---

# 🤖 Step 6: Create the RAG Chain

The Gemini model is used as the LLM:

```python
ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=my_gemini_api_key,
    temperature=0
)
```

The retriever provides relevant information to Gemini.

The overall process is:

```text
User Question
      │
      ▼
Retriever
      │
      ▼
Top 3 Relevant Chunks
      │
      ▼
Gemini
      │
      ▼
Generated Answer
```

---

# ❓ Step 7: Ask a Question

The question is defined here:

```python
my_question = "what is the remote policy?"
```

Then the RAG chain is executed:

```python
result = qa.invoke({
    "query": my_question
})
```

Finally:

```python
print(result["result"])
```

prints the generated answer.

---

# ▶️ Run the Project

Activate your virtual environment first.

### Windows

```bash
venv\Scripts\activate
```

Then run:

```bash
python rag.py
```

### Linux / Ubuntu

```bash
source venv/bin/activate
python3 rag.py
```

---

# 📌 Example

### Knowledge

```text
Employees can work remotely up to three days per week.
```

### Question

```text
What is the remote policy?
```

### RAG Process

```text
Question
   ↓
Convert question into embedding
   ↓
Search FAISS
   ↓
Retrieve relevant chunks
   ↓
Send context + question to Gemini
   ↓
Generate answer
```

### Example Answer

```text
Employees can work remotely up to three days per week,
with manager approval.
```

---

# 🔄 Complete RAG Pipeline

```text
              ┌─────────────────────┐
              │  knowledge.md       │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    TextLoader       │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Text Splitter     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Gemini Embeddings   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │       FAISS         │
              │    Vector Store     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │     Retriever       │
              └──────────┬──────────┘
                         │
                    Top K Chunks
                         │
                         ▼
              ┌─────────────────────┐
              │    Gemini LLM       │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    Final Answer     │
              └─────────────────────┘
```

---

# 🎯 Key Concepts

## RAG

**RAG = Retrieval-Augmented Generation**

Instead of asking the LLM to answer only from its training knowledge, RAG first retrieves relevant information from your own documents.

---

## Embedding

An embedding converts text into a numerical vector that represents its semantic meaning.

---

## Vector Database

A vector store stores embeddings and allows similarity searches.

In this project:

```text
FAISS
```

is used as the vector store.

---

## Retriever

The retriever searches the vector store and returns the most relevant document chunks.

This project uses:

```text
Top K = 3
```

---

## LLM

The Large Language Model generates the final response.

This project uses:

```text
Gemini
```

---

# ⚠️ Important Notes

### Windows File Path

For better cross-platform compatibility, use:

```python
loader = TextLoader(
    "documents/knowledge.md",
    encoding="utf-8"
)
```

instead of:

```python
"documents\knowledge.md"
```

Using `/` works reliably on Windows and Linux.

---

### Protect Your API Key

Never write your API key directly inside `rag.py`.

Use:

```text
.env
```

and:

```python
os.getenv("GOOGLE_API_KEY")
```

---

# 🛠️ Future Improvements

This basic RAG project can be extended with:

* PDF document loading
* Multiple document support
* Web document loading
* Metadata filtering
* Hybrid search
* Re-ranking
* Conversation memory
* Streaming responses
* FastAPI backend
* React frontend
* Persistent FAISS index
* Supabase / pgvector
* ChromaDB
* Document upload functionality
* Chat interface

---

# 📚 Learning Outcome

After completing this project, you will understand:

1. What RAG is
2. How document loading works
3. How text chunking works
4. What embeddings are
5. How FAISS works
6. What vector similarity search means
7. How a retriever works
8. How Gemini generates answers
9. How LangChain connects all components
10. How to build a basic RAG application in Python

---

# 👩‍💻 Author

**Amreen Pathan**

B.Tech – Computer Science

---

# ⭐ Project Summary

This project demonstrates a basic **Retrieval-Augmented Generation application** using **LangChain, Google Gemini embeddings, Gemini LLM, and FAISS**.

The system retrieves relevant information from a private Markdown knowledge base and uses Gemini to generate a natural-language answer based on that retrieved context.

