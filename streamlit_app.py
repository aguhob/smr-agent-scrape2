# streamlit_app.py – Fresh Start (Smart Scraped Viewer Only)
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SMR Article Viewer", layout="wide")

# Load and validate data
try:
    scraped_df = pd.read_csv("scraped_smr_sources.csv")
except Exception as e:
    st.error(f"❌ Failed to load scraped_smr_sources.csv: {e}")
    st.stop()

# Normalize and clean
scraped_df.columns = scraped_df.columns.str.strip()
if 'content' not in scraped_df.columns:
    st.error("❌ Missing 'content' column in the CSV file.")
    st.stop()

# Ensure all content is a string
scraped_df['content'] = scraped_df['content'].fillna('').astype(str)
scraped_df = scraped_df[scraped_df['content'].str.strip() != '']
if scraped_df.empty:
    st.warning("⚠️ No usable articles found. Try re-running the scraper.")
    st.stop()

# UI
st.title("🔍 SMR Smart-Scraped Article Viewer")
st.markdown("Explore scraped articles containing **'nuclear'**, pulled from Archive.org.")

# Sidebar filters
sources = scraped_df['original_url'].dropna().unique()
selected_source = st.sidebar.selectbox("Choose source", sources)

filtered = scraped_df[scraped_df['original_url'] == selected_source]

for _, row in filtered.iterrows():
    st.subheader(f"[{row['original_url']}]({row['archive_url']})")
    st.write(row['content'][:3000] + ("..." if len(row['content']) > 3000 else ""))
