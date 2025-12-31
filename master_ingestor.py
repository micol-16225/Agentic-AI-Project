import requests
import urllib.parse
import pandas as pd
import xml.etree.ElementTree as ET
import os

# --- 1. YOUR ORIGINAL FDA LOGIC (Restored) ---
BASE_URL = "https://api.fda.gov/transparency/crl.json"
BIMO_CITATIONS = "(312.60 OR 312.50)"
STATISTICAL_VIOLATIONS = "(ineligible+OR+unblinded+OR+stratification+OR+sample+size+OR+protocol+deviation)"
RECIPIENT_FILTER = "(recipient:Sponsor+OR+recipient:Company)"
SEARCH_QUERY = f'citation:{BIMO_CITATIONS}+AND+text:{STATISTICAL_VIOLATIONS}+AND+{RECIPIENT_FILTER}' 

def fetch_fda_original():
    print(f"📡 Using your original FDA query...")
    # Note: We don't double-encode here, requests.get handles it
    params = {'search': SEARCH_QUERY, 'limit': 5}
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status() 
        results = response.json().get('results', [])
        
        processed = []
        for r in results:
            processed.append({
                'source': 'FDA_CRL',
                'date': r.get('letter_date', 'N/A'),
                'title': f"CRL to {r.get('recipient', 'N/A')}",
                'content': r.get('text', '')[:2000]
            })
        return processed
    except Exception as e:
        print(f"❌ Original FDA Script Error: {e}")
        return []

# --- 2. THE PUBMED ADDITION ---
def fetch_pubmed_simple():
    print("📚 Fetching PubMed context...")
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": "biostatistics[Title]", "retmode": "json", "retmax": 5}
    
    try:
        # Get IDs
        res = requests.get(url, params=params).json()
        ids = res.get("esearchresult", {}).get("idlist", [])
        if not ids: return []

        # Get Abstracts
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f_res = requests.get(fetch_url, params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
        root = ET.fromstring(f_res.content)
        
        processed = []
        for article in root.findall(".//PubmedArticle"):
            title = article.find(".//ArticleTitle").text
            processed.append({
                'source': 'PubMed_Academic',
                'date': '2025',
                'title': title,
                'content': title # Keeping it light for the first success
            })
        return processed
    except Exception as e:
        print(f"❌ PubMed Error: {e}")
        return []

# --- 3. THE UNIFIED SYNC ---
def run_sync():
    # Step A: Get Live Data
    fda_data = fetch_fda_original()
    pubmed_data = fetch_pubmed_simple()
    
    # Step B: Save Live Data to CSV
    combined_live = fda_data + pubmed_data
    pd.DataFrame(combined_live).to_csv("master_regulatory_library.csv", index=False)
    print(f"📡 Scraped {len(combined_live)} live records.")

    # Step C: TRIGGER HYDRATION (This adds your wisdom)
    print("🧪 Merging your 12 Wisdom Sections into the library...")
    try:
        import hydrate_library
        hydrate_library.hydrate_full()
    except Exception as e:
        print(f"❌ Could not find hydrate_library.py: {e}")

if __name__ == "__main__":
    run_sync()