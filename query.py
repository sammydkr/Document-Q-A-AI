import faiss
import numpy as np
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

index = faiss.read_index("vectorstore.faiss")

with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("---\n")

def ask(question):
    q_embed = client.embeddings.create(
        input=[question],
        model="text-embedding-3-large"
    ).data[0].embedding

    q_embed = np.array(q_embed).astype("float32").reshape(1, -1)

    D, I = index.search(q_embed, 3)
    context = "\n".join([chunks[i] for i in I[0]])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Use the context to answer accurately."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    print("\nANSWER:\n", response.choices[0].message.content)

ask("Give me a summary of this document.")
