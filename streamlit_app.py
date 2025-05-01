# streamlit_app.py – Smart Scraping Viewer Only
import pandas as pd
import streamlit as st

# Load scraped data
scraped_df = pd.read_csv("scraped_smr_sources.csv")

# Normalize column names
scraped_df.columns = scraped_df.columns.str.strip()

# Validate 'content' column
if 'content' not in scraped_df.columns:
    st.error("❌ The 'content' column is missing from scraped_smr_sources.csv. Please check the file format.")
    st.stop()

# Clean and convert
scraped_df['content'] = scraped_df['content'].astype(str)
scraped_df = scraped_df[scraped_df['content'].apply(lambda x: x.strip() != '' and x.lower() != 'nan')]

# Stop if empty
if scraped_df.empty:
    st.error("⚠️ No valid nuclear-related articles found. Please re-run the scraper.")
    st.stop()

# Streamlit UI
st.title("SMR Smart Scraping Explorer")
st.markdown("View the most recent scraped articles that mention **nuclear**.")

# Article filter
selected_url = st.selectbox("Choose a source to view:", scraped_df['original_url'].unique())
article_rows = scraped_df[scraped_df['original_url'] == selected_url]

for _, row in article_rows.iterrows():
    st.markdown(f"### [{row['original_url']}]({row['archive_url']})")
    st.write(row['content'][:3000] + ("..." if len(row['content']) > 3000 else ""))
