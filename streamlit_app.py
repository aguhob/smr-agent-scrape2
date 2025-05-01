# streamlit_app.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
import numpy as np

# Load scraped data
scraped_df = pd.read_csv("scraped_smr_sources.csv")

# Normalize column names in case of extra spaces or format issues
scraped_df.columns = scraped_df.columns.str.strip()

# Check if 'content' column exists
if 'content' not in scraped_df.columns:
    st.error("❌ The 'content' column is missing from scraped_smr_sources.csv. Please check the file format.")
    st.stop()

# CLEANUP step: Drop rows with missing or invalid content
scraped_df = scraped_df.dropna(subset=['content'])
scraped_df = scraped_df[scraped_df['content'].apply(lambda x: isinstance(x, str) and x.strip() != '')]

# Final hardening: Force 'content' to be string
scraped_df['content'] = scraped_df['content'].astype(str)

# Stop the app if no valid content remains
if scraped_df.empty:
    st.error("⚠️ No valid nuclear-related articles found. Please check your data or rerun the scraper.")
    st.stop()

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

# Prepare documents and metadata
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

# Embed documents
embeddings = model.encode(documents)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# Save embeddings (optional)
faiss.write_index(index, "faiss_index.idx")
pd.DataFrame(metadata).to_csv("metadata.csv", index=False)

# Streamlit App UI
st.title("SMR Risk Report Generator 🚀")

st.subheader("Tailor your report:")
audience = st.selectbox("Choose your audience:", ["Investor", "Community", "Industrial Real Estate"])

query = st.text_input("Enter your question about SMRs:")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)

    st.subheader("Relevant Findings:")
    context_chunks = []
    for idx in I[0]:
        st.write(documents[idx])
        st.caption(f"Source: {metadata[idx]['original_url']}")
        context_chunks.append(documents[idx])

    st.subheader("Customized Risk Summary:")

    # Customize summary by audience
    if audience == "Investor":
        focus = "Focus on financial risks, ROI, asset impacts, and regulatory hurdles."
    elif audience == "Community":
        focus = "Focus on safety, health, environmental risks, and community acceptance."
    else:
        focus = "Focus on zoning, siting issues, industrial insurance, and property valuation."

    full_context = "\n".join(context_chunks)
    st.write(f"**({audience} Focused Summary Placeholder)**\n\n{focus}\n\n" + full_context[:2000] + "...")
