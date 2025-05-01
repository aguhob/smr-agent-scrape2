# Archive.org Scraper + Chunking + Embedding + Streamlit RAG Demo for SMR Hackathon

import requests
import pandas as pd
import time
import os
from bs4 import BeautifulSoup
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# List of URLs to retrieve from Archive.org
urls = [
    "https://www.scientificamerican.com",
    "https://www.bnnbloomberg.ca",
    "https://www.axios.com",
    "https://techcrunch.com",
    "https://nuclearstreet.com/",
    "https://www.pewresearch.org",
    "https://www.doctorsfornuclearenergy.org",
    "https://grist.org/",
    "https://thenarwhal.ca",
    "http://www.bostonglobe.com",
    "https://holtecinternational.com",
    "https://nanonuclearenergy.com/",
    "https://kairospower.com",
    "https://www.terrapower.com",
    "https://www.aalo.com",
    "https://www.energy.gov/ne/office-nuclear-energy-news",
    "https://www.energy.gov/ne/listings/ne-press-releases",
    "https://www.anl.gov",
    "https://thoriumenergyalliance.com",
    "https://www.nucnet.org"
]

# Function to get the latest snapshot URL from archive.org
def get_latest_archive_url(site_url):
    api_url = f"http://archive.org/wayback/available?url={site_url}"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        try:
            return data['archived_snapshots']['closest']['url']
        except KeyError:
            return None
    return None

# Function to scrape text from a webpage
def scrape_text_from_url(archive_url):
    try:
        response = requests.get(archive_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join([p.get_text() for p in paragraphs if p.get_text()])
        return text[:5000]  # Limit to 5000 characters for now
    except Exception as e:
        print(f"Error scraping {archive_url}: {e}")
        return ""

# Function to split text into smaller chunks
def chunk_text(text, max_chunk_size=500):
    sentences = text.split(".")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + "."
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + "."
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Main scraping loop
scraped_data = []

if not os.path.exists("scrapedchunked_smr_sources.csv"):
    for url in urls:
        print(f"Processing: {url}")
        archive_url = get_latest_archive_url(url)
        if archive_url:
            print(f" -> Archive found: {archive_url}")
            text = scrape_text_from_url(archive_url)
            scraped_data.append({
                'original_url': url,
                'archive_url': archive_url,
                'content': text
            })
        else:
            print(f" -> No archive found.")
        time.sleep(1)  # Be nice to archive.org

    scraped_df = pd.DataFrame(scraped_data)
    scraped_df.to_csv("scraped_smr_sources.csv", index=False)
    print("Scraping complete! Saved to scraped_smr_sources.csv")
else:
    scraped_df = pd.read_csv("scraped_smr_sources.csv")

# Chunk and embed
print("Chunking and embedding...")
model = SentenceTransformer('all-MiniLM-L6-v2')
all_texts = []
all_chunks = []
original_urls = []

for idx, row in scraped_df.iterrows():
    chunks = chunk_text(row['content'])
    all_chunks.extend(chunks)
    original_urls.extend([row['original_url']] * len(chunks))

embeddings = model.encode(all_chunks, convert_to_numpy=True)

# Save to FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "smr_index.faiss")

# Save chunks metadata
chunks_df = pd.DataFrame({
    'chunk': all_chunks,
    'source': original_urls
})
chunks_df.to_csv("chunks_metadata.csv", index=False)

print("Chunking and embedding complete!")

# Simple Streamlit RAG App
def load_index_and_chunks():
    index = faiss.read_index("smr_index.faiss")
    chunks_df = pd.read_csv("chunks_metadata.csv")
    return index, chunks_df

def search_index(query, index, chunks_df, model, top_k=5):
    query_vec = model.encode([query]).astype('float32')
    D, I = index.search(query_vec, top_k)
    results = chunks_df.iloc[I[0]]
    return results

st.title("SMR Risk Report Generator (Hackathon Demo)")

query = st.text_input("Enter your question (e.g., What are real estate risks with SMRs?)")

if query:
    with st.spinner("Searching knowledge base..."):
