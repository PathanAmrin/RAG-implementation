# RAG Implementation Project

This project is a simple Retrieval-Augmented Generation (RAG) application built using Python. It demonstrates how to integrate a vector database, an embedding model, and a Large Language Model (LLM) to answer questions based on a provided text document.

## How it Works

1.  **Document Loading:** The application reads text from `documents/college.txt`.
2.  **Chunking:** The document is split into smaller chunks (by lines).
3.  **Embedding:** The `sentence-transformers` library (`all-MiniLM-L6-v2` model) converts text chunks into vector embeddings.
4.  **Vector Database:** These embeddings are stored locally in a persistent ChromaDB database (`chroma_db` directory).
5.  **Retrieval:** When a user asks a question, the question is also converted into a vector. ChromaDB is queried to find the most similar document chunks.
6.  **Generation:** The retrieved context and the original question are sent to an LLM via OpenRouter API (using the `openai` python package) to generate a final, context-aware answer.

## Prerequisites

-   Python 3.7+
-   An API key from [OpenRouter](https://openrouter.ai/)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PathanAmrin/RAG-implementation.git
    cd RAG-project
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory of the project and add your OpenRouter API key:
    ```env
    OPENROUTER_API_KEY=your_openrouter_api_key_here
    ```

5.  **Prepare your document:**
    Ensure you have a text file at `documents/college.txt` with the content you want the model to answer questions about.

## Usage

Run the main script:

```bash
python rag.py
```

The script will:
1. Load the embedding model.
2. Initialize the ChromaDB collection.
3. Chunk and embed the document, storing the vectors.
4. Prompt you to ask a question.
5. Retrieve relevant context and generate an answer using the OpenRouter LLM.

## Dependencies

-   `chromadb`: Vector database for storing and querying embeddings.
-   `sentence-transformers`: Local generation of text embeddings.
-   `openai`: Used as the client to interact with the OpenRouter API.
-   `python-dotenv`: Loads environment variables from the `.env` file.
