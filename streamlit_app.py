import pandas as pd
import streamlit as st

st.set_page_config(page_title="SMR Article Viewer", layout="wide")

# Load scraped data
try:
    df = pd.read_csv("scraped_smr_sources.csv")
except Exception as e:
    st.error(f"❌ Could not load 'scraped_smr_sources.csv': {e}")
    st.stop()

# Normalize column names and ensure 'content' exists
df.columns = df.columns.str.strip()
if "content" not in df.columns:
    st.error("❌ Missing 'content' column in your CSV. Please check the file.")
    st.stop()

# Clean data
df["content"] = df["content"].astype(str).fillna("")
df = df[df["content"].str.strip().str.lower() != "nan"]
if df.empty:
    st.warning("⚠️ No usable articles found. Re-run your scraper or check your data.")
    st.stop()

# UI
st.title("🔍 SMR Nuclear Article Explorer")
st.markdown("This app displays smart-s
