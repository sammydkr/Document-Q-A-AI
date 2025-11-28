import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

def embed(texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-large"
    )
    return np.array([d.embedding for d in response.data]).astype("float32")

# Load and process
text = read_pdf("data/document.pdf")
chunks = chunk_text(text)

embeddings = embed(chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "vectorstore.faiss")

with open("chunks.txt", "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(c + "\n---\n")

print("Vector store created.")
