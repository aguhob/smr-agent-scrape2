
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
import numpy as np

# Load scraped data
scraped_df = pd.read_csv("scraped_smr_sources.csv")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Chunk text into smaller pieces
def chunk_text(text, max_tokens=500):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Prepare all chunks
documents = []
metadata = []

for idx, row in scraped_df.iterrows():
    chunks = chunk_text(row['content'])
    for chunk in chunks:
        documents.append(chunk)
        metadata.append({
            'original_url': row['original_url'],
            'archive_url': row['archive_url']
        })

# Embed all chunks
embeddings = model.encode(documents)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# Save for future use
faiss.write_index(index, "faiss_index.idx")
pd.DataFrame(metadata).to_csv("metadata.csv", index=False)

# Streamlit App
st.title("SMR Risk Report Generator 🚀")
metadata_df = pd.read_csv("metadata.csv")
index = faiss.read_index("faiss_index.idx")

query = st.text_input("Enter your question about SMRs:")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)

    st.subheader("Relevant Findings:")
    for idx in I[0]:
        st.write(documents[idx])
        st.caption(f"Source: {metadata_df.iloc[idx]['original_url']}")

    st.subheader("Auto-Generated Summary:")
    full_context = "\n".join([documents[idx] for idx in I[0]])
    st.write("(Summarizer Placeholder)\n" + full_context[:2000] + "...")
