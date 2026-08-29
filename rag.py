import os

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv


# ==================================================
# 1. Load environment variables
# ==================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")


# ==================================================
# 2. Create OpenRouter client
# ==================================================

llm = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


# ==================================================
# 3. Load embedding model
# ==================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# 4. Create ChromaDB
# ==================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


# ==================================================
# 5. Create collection
# ==================================================

collection = chroma_client.get_or_create_collection(
    name="college"
)


# ==================================================
# 6. Read document
# ==================================================

with open("documents/college.txt", "r") as file:
    text = file.read()


# ==================================================
# 7. Split document into chunks
# ==================================================

chunks = [
    line.strip()
    for line in text.split("\n")
    if line.strip()
]


print("\nDocument chunks:")

for i, chunk in enumerate(chunks):
    print(f"{i}: {chunk}")


# ==================================================
# 8. Create embeddings
# ==================================================

print("\nCreating embeddings...")

embeddings = embedding_model.encode(
    chunks
).tolist()


# ==================================================
# 9. Store documents and vectors in ChromaDB
# ==================================================

collection.upsert(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

print("Documents stored in ChromaDB.")


# ==================================================
# 10. Get question from user
# ==================================================

question = input("\nAsk a question: ")


# ==================================================
# 11. Convert question into vector
# ==================================================

query_embedding = embedding_model.encode(
    [question]
).tolist()


# ==================================================
# 12. Search ChromaDB
# ==================================================

results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)


# ==================================================
# 13. Get retrieved context
# ==================================================

retrieved_documents = results["documents"][0]

context = "\n".join(retrieved_documents)


print("\nRetrieved context:")

for document in retrieved_documents:
    print("-", document)


# ==================================================
# 14. Create prompt for LLM
# ==================================================

prompt = f"""
You are a helpful college assistant.

Answer the user's question using ONLY the context provided below.

Context:
{context}

Question:
{question}

If the answer is not available in the context,
say "I don't know based on the provided information."
"""


# ==================================================
# 15. Send context + question to OpenRouter
# ==================================================

print("\nGenerating answer...")

response = llm.chat.completions.create(
    model="openai/gpt-4.1",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens=500
)


# ==================================================
# 16. Display final answer
# ==================================================

answer = response.choices[0].message.content

print("\n==============================")
print("FINAL ANSWER")
print("==============================")
print(answer)