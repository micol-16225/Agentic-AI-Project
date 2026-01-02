import requests
import pandas as pd
import os
import time
from datetime import datetime

# --- CONFIGURATION (Matches your Hydrate script) ---
HISTORICAL_FILE = "fda_letters2.csv"
BASE_URL = "https://api.fda.gov/drug/warningletter.json" # Using the stable endpoint

# The keywords your Hydrator/Auditor cares about
SAP_KEYWORDS = ["STATISTICAL", "SAMPLE SIZE", "UNBLINDED", "ESTIMAND", "MULTIPLICITY"]

def fetch_and_map_precedents():
    print(f"📡 Starting Ingestor: Targeting SAP Precedents...")
    
    # We query for 'clinical trial' to keep it relevant to your SAP audit
    query = 'text:"clinical+trial"+AND+(text:"statistical"+OR+text:"analysis")'
    params = {'search': query, 'limit': 50}
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        results = data.get('results', [])
        
        processed = []
        for r in results:
            # --- MAPPING TO MATCH YOUR HYDRATE SCRIPT ---
            processed.append({
                'source': 'FDA_API_Live',
                'type': 'Precedent',            # Matches 'type' in Hydrator
                'title': f"WL: {r.get('subject', 'Unknown')}", # Matches 'title'
                'content': r.get('text', ''),   # Matches 'content'
                'date': r.get('date', 'N/A')    # Matches 'date'
            })
        
        new_df = pd.DataFrame(processed)

        # Merge with existing file if it exists to prevent data loss
        if os.path.exists(HISTORICAL_FILE):
            print(f"📂 Merging with existing {HISTORICAL_FILE}...")
            existing_df = pd.read_csv(HISTORICAL_FILE)
            final_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=['content'])
        else:
            final_df = new_df

        final_df.to_csv(HISTORICAL_FILE, index=False)
        print(f"✅ Success! {HISTORICAL_FILE} updated. Ready for Hydration.")
        
    except Exception as e:
        print(f"❌ Ingestor failed: {e}")
        print("💡 Tip: If API is 500/404, your Hydrate script will still work using the Verbatim records.")

if __name__ == "__main__":
    fetch_and_map_precedents()