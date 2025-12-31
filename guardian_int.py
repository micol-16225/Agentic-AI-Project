import os
import json
import time
from google import genai
from google.genai import errors # Import this to catch the specific quota error

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "models/gemini-2.5-flash"

def analyze_with_retry(record, index, retries=3):
    """Tries to analyze, but waits if the quota is hit."""
    letter_content = record.get('full_text', record.get('text', ''))[:30000]
    prompt = f"Analyze this FDA correspondence (Record #{index}). Extract Sponsor Name and Failure Profile. [JSON Output Required]"
    
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            return json.loads(response.text)
        
        except Exception as e:
            if "429" in str(e):
                wait_time = (attempt + 1) * 30 # Wait 30, then 60, then 90 seconds
                print(f"⚠️ Quota hit. Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
            else:
                print(f"❌ Permanent Error on Record {index}: {e}")
                return None
    return None

def main():
    with open("fda_letters.json", "r") as f:
        records = json.load(f)

    # NEW: Check if we have an existing progress file to resume from
    kb_file = "guardian_kb_final.json"
    if os.path.exists(kb_file):
        with open(kb_file, "r") as f:
            kb = json.load(f)
        start_index = len(kb)
        print(f"🔄 Resuming from record {start_index + 1}...")
    else:
        kb = []
        start_index = 0

    print(f"🚀 Processing records {start_index + 1} to {len(records)}...")

    for i in range(start_index, len(records)):
        print(f"({i+1}/{len(records)}) Analyzing...")
        
        analysis = analyze_with_retry(records[i], i+1)
        
        if analysis:
            found_name = analysis.get('identified_sponsor', 'Unknown')
            print(f"   ✅ Identified: {found_name}")
            
            kb.append({
                "meta": {"company": found_name, "date": records[i].get('date')},
                "audit_insights": analysis
            })
            
            # SAVE PROGRESS after every successful record
            with open(kb_file, "w") as f:
                json.dump(kb, f, indent=4)
        
        # Fundamental "Politeness" Delay (6 seconds = 10 requests per minute)
        time.sleep(6) 

    print("\n✅ MASTER KNOWLEDGE BASE COMPLETE!")

if __name__ == "__main__":
    main()