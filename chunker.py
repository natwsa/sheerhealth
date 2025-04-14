from PyPDF2 import PdfReader
import pandas as pd
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import time

pdf_path = "/Users/sandalowash/Downloads/purd.pdf"  #put your path here
reader = PdfReader(pdf_path)
text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

# Fixed-size chunking with overlap
def chunk_text_with_overlap(text, max_words=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words - overlap  # move forward with overlap
    return chunks

chunks_with_overlap = chunk_text_with_overlap(text)

df = pd.DataFrame({"chunk": chunks_with_overlap})

embeddings = []
for i, chunk in enumerate(df["chunk"]):
    try:
        response = client.embeddings.create(input=chunk,
        model="text-embedding-ada-002")
        embeddings.append(response.data[0].embedding)
        time.sleep(0.1)  # polite delay to avoid rate limits
    except Exception as e:
        print(f"Error on chunk {i}: {e}")
        embeddings.append(None)

df["embedding"] = embeddings
df.to_pickle("embeddings.pkl")


