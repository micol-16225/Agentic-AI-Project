import os
import json
from google import genai

# Setup
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "models/gemini-2.5-flash"

def run_spot_check():
    # 1. Load your existing data
    if not os.path.exists("fda_letters.json"):
        print("❌ fda_letters.json not found!")
        return

    with open("fda_letters.json", "r") as f:
        records = json.load(f)

    # 2. Grab the first record
    first_letter = records[0]
    content = first_letter.get('full_text', '') or first_letter.get('text', '')
    
    print(f"--- 🔎 SPOT CHECKING RECORD #1 ---")
    print(f"Recipient: {first_letter.get('recipient')}")
    print(f"Date: {first_letter.get('letter_date') or first_letter.get('date')}")
    print("-" * 30)

    # 3. Ask Gemini for a high-level audit
    prompt = f"""
    You are a Clinical Trial Auditor. Read this FDA correspondence and summarize the 
    'Biological/Statistical Risk' in 3 bullet points. 
    Then, provide a 'Severity Score' from 1-10.

    TEXT:
    {content[:10000]} 
    """

    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        print("🤖 AI ANALYSIS:")
        print(response.text)
    except Exception as e:
        print(f"❌ API Error: {e}")

if __name__ == "__main__":
    run_spot_check()