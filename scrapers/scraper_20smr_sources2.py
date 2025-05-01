# archive_scraper.py

import requests
import pandas as pd
import time
from bs4 import BeautifulSoup

# List of URLs you want to scrape (the real top 20 you selected)
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

# Function to get latest archived snapshot
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

# Function to scrape text from an archive URL
def scrape_text_from_url(archive_url):
    try:
        response = requests.get(archive_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join([p.get_text() for p in paragraphs if p.get_text()])
        return text[:5000]  # Limit to 5000 characters per site
    except Exception as e:
        print(f"Error scraping {archive_url}: {e}")
        return ""

# Main scraping
scraped_data = []

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
    time.sleep(1)  # Be nice to Archive.org!

# Save to CSV
scraped_df = pd.DataFrame(scraped_data)
scraped_df.to_csv("scraped_smr_sources.csv", index=False)

print("✅ Scraping complete! File saved: scraped222_smr_sources.csv")
