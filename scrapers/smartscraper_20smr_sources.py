# smart_archive_scraper.py

import requests
import pandas as pd
import time
from bs4 import BeautifulSoup

# Your Top 20 SMR-related URLs
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

# YEARS to prioritize
TARGET_YEARS = ["2025", "2024", "2023", "2022"]

# Function to get the closest snapshot in target years
def get_archive_url(site_url, years=TARGET_YEARS):
    for year in years:
        timestamp = year + "0101"  # January 1st of the year
        api_url = f"http://archive.org/wayback/available?url={site_url}&timestamp={timestamp}"
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            try:
                return data['archived_snapshots']['closest']['url']
            except KeyError:
                continue  # Try next year if no archive found
    return None

# Function to scrape full article text smartly
def scrape_text_from_url(archive_url):
    try:
        response = requests.get(archive_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try grabbing <article> tag first
        article = soup.find('article')
        if article:
            text = article.get_text(separator="\n")
        else:
            # Fallback: all <p> and <div> tags
            paragraphs = soup.find_all(['p', 'div'])
            text = "\n".join([p.get_text() for p in paragraphs if p.get_text()])

        text = text.strip()

        # Also check the headline if needed
        headline = soup.find('h1')
        headline_text = headline.get_text() if headline else ""

        combined_text = headline_text + "\n" + text

        # Only keep if "nuclear" appears at least twice
        if combined_text.lower().count("nuclear") >= 2:
            return combined_text[:10000]  # Limit size for efficiency
        else:
            print(f"Filtered out (nuclear too rare): {archive_url}")
            return ""
    except Exception as e:
        print(f"Error scraping {archive_url}: {e}")
        return ""

# Main loop
scraped_data = []

for url in urls:
    print(f"Processing: {url}")
    archive_url = get_archive_url(url)
    if archive_url:
        print(f" -> Found archive: {archive_url}")
        text = scrape_text_from_url(archive_url)
        if text:
            scraped_data.append({
                'original_url': url,
                'archive_url': archive_url,
                'content': text
            })
    else:
        print(f" -> No archive found for {url}")
    time.sleep(1)  # Be nice to Archive.org

# Save results
scraped_df = pd.DataFrame(scraped_data)
scraped_df.to_csv("smartscrape_smr_sources.csv", index=False)

print("✅ Scraping complete! Saved to scraped_smr_sources.csv")
