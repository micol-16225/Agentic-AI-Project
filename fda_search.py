import requests
import urllib.parse
import json
import pandas as pd
import os

# --- 1. Configuration and Constants ---
BASE_URL = "https://api.fda.gov/transparency/crl.json"

# BIMO Citations (Investigator and Sponsor Responsibilities)
BIMO_CITATIONS = "(312.60 OR 312.62 OR 312.50 OR 312.57)"

# Execution and Statistical Failure Keywords
STATISTICAL_VIOLATIONS = "(ineligible+OR+unblinded+OR+stratification+OR+sample+size+OR+dose+error+OR+protocol+deviation+OR+statistical+validity+OR+unreliable+data)"

# Recipient Filter (Targeting Clinical Sponsors and Organizations)
RECIPIENT_FILTER = "(recipient:MD+OR+recipient:Investigator+OR+recipient:Sponsor+OR+recipient:Pharmaceutical+OR+recipient:Company)"

# Composite Search Query
SEARCH_QUERY = f'citation:{BIMO_CITATIONS}+AND+text:{STATISTICAL_VIOLATIONS}+AND+{RECIPIENT_FILTER}' 
encoded_query = urllib.parse.quote(SEARCH_QUERY)

params = {
    'search': encoded_query,
    'limit': 100,  # Starting with 100. Max is 1000 per request.
    'skip': 0 # Start at the beginning
    # 'api_key': 'YOUR_API_KEY' # Use your API key for higher volume
}

# --- 2. Data Fetching Function ---
def fetch_fda_data(base_url, params):
    all_records = []
    
    # We will use a loop to simulate retrieving more than the default 'limit' (e.g., more than 100)
    # A complete solution would loop and increment 'skip' until 'total' is reached, 
    # but we'll stick to a simple single call for now.
    
    print(f"Executing search with query: {params['search']}")
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() 
        data = response.json()
        
        if 'results' in data and data['results']:
            all_records.extend(data['results'])
            print(f"Successfully retrieved {len(all_records)} records.")
        
        return all_records
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the API request: {e}")
        return []

# --- 3. Data Processing and Saving Function ---
def process_and_save_data(data_records, json_filename="fda_letters.json", csv_filename="fda_letters.csv"):
    if not data_records:
        print("No data to save.")
        return

    # --- Step A: Save to JSON (Preserves the original structure) ---
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(data_records, f, ensure_ascii=False, indent=4)
    print(f"\n✅ Data successfully saved to {json_filename} (Full JSON).")

    # --- Step B: Prepare and Save to CSV (For easy viewing and labeling) ---
    
    # 1. Extract only the key fields and add the custom 'Doc Type'
    processed_data = []
    for letter in data_records:
        full_text = letter.get('text', '').upper()
        
        # Categorize the document type (for your knowledge base)
        doc_type = ""
        if "WARNING LETTER" in full_text:
            doc_type = "TYPE A: Execution/Compliance Failure (WL)"
        elif "COMPLETE RESPONSE" in full_text:
            doc_type = "TYPE B: Statistical/Regulatory Consequence (CRL)"
        else:
            doc_type = "TYPE C: Other BIMO Correspondence"

        processed_data.append({
            'doc_type': doc_type,
            'recipient': letter.get('recipient', 'N/A'),
            'date': letter.get('letter_date', 'N/A'),
            'citations': ', '.join(letter.get('citation', [])),
            # Truncate the text for easy CSV viewing, but keep the full text in JSON
            'text_snippet': letter.get('text', '')[:1000].replace('\n', ' '), 
            'full_text': letter.get('text', '')
        })

    # 2. Convert to Pandas DataFrame and save to CSV
    df = pd.DataFrame(processed_data)
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    print(f"✅ Data successfully saved to {csv_filename} (CSV for analysis).")


# --- 4. Main Execution Block ---
if __name__ == "__main__":
    # 1. Fetch the Data
    data_records = fetch_fda_data(BASE_URL, params)
    
    # 2. Process and Save
    process_and_save_data(data_records)
    
    # 3. Inform the user of completion
    print("The files 'fda_letters.json' and 'fda_letters.csv' now contain your knowledge base.")
