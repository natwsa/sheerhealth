import numpy as np
import pandas as pd
import faiss
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load your saved chunk embeddings
df = pd.read_pickle("embeddings.pkl")
embeddings = np.array(df["embedding"].tolist()).astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
index.add(embeddings)


def retrieve_chunks(query, k=3):
    # Call OpenAI Embedding API
    response = openai.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    query_embedding = np.array(response.data[0].embedding, dtype="float32")

    # Search the FAISS index
    D, I = index.search(query_embedding.reshape(1, -1), k)
    return df.iloc[I[0]]["chunk"].tolist()


