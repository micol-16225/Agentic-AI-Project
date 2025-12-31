import json
import os
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "models/gemini-2.5-flash"

def load_kb():
    with open("guardian_kb_final.json", "r") as f:
        return json.load(f)

def audit_protocol(user_protocol, kb_data):
    # 1. Load the "Expert Rules" from the text file
    try:
        with open("regulatory_library.txt", "r") as f:
            stat_law = f.read()
    except FileNotFoundError:
        return "❌ Error: 'regulatory_library.txt' not found. Please create it in the same folder."

    # 2. Gather the 81 historical failure patterns (Defining the missing variable)
    historical_lessons = "\n".join([
        f"- {item.get('audit_insights', {}).get('auditor_warning', 'N/A')}" 
        for item in kb_data[:40] 
    ])

    prompt = f"""
    You are a Senior FDA Statistical Reviewer. You are auditing a protocol from a TOP-TIER pharma company. 
    They don't make rookie mistakes; they make COMPLEX ones. You must use BOTH the provided Law and the 81 Past Failure Records.
    
    --- STEP 1: LEGAL COMPLIANCE ---
    Use this library to find violations: 
    {stat_law}

    --- STEP 2: HISTORICAL PRECEDENT ---
    Search these past rejections for matches: 
    {historical_lessons}
    
    --- TASK ---
    Scan the protocol for these 4 "Professional-Grade" Red Flags:
    
    1. ESTIMAND MISALIGNMENT: Does the way they handle 'Intercurrent Events' (like rescue meds) actually answer the clinical question?
    2. THE ALPHA PENALTY: If they are doing interim looks or adaptive changes, are they 'spending' their Alpha correctly?
    3. SEPARATION OF POWERS: Is there a risk that the people cleaning the data might be 'unblinded'? 
    4. DOCUMENT CONSISTENCY: Look for 'Footnote Conflicts'—do windows in text match the tables?

    Identify subtle flaws. You MUST support your findings by citing the 'Academic Arsenal' 
    provided in the library (e.g., Benjamini & Hochberg 1995 for multiplicity, Rubin 1976 for missing data).

    If you find a 'Selection Bias' issue, cite Pearl (2011) or Rothwell (2005). 
    This makes your report legally and academically defensible.
    
    Audit the protocol. If a risk matches a past failure from the database, 
    start that sentence with: "HISTORICAL MATCH FOUND:"

    PROTOCOL TO AUDIT:
    {user_protocol}
    
    Format your response with:
    1. RISK ASSESSMENT (High/Medium/Low)
    2. HISTORICAL MATCHES (Mention specific patterns found in the KB)
    3. RECOMMENDED MITIGATION
    """

    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        return response.text
    except Exception as e:
        return f"❌ AI Audit failed: {e}"

if __name__ == "__main__":
    kb = load_kb()
    print("🛡️ Guardian Auditor Agent Online.")
    print("-" * 30)
    
    my_trial = input("Describe the trial protocol or biostat plan to audit: \n> ")
    
    if my_trial:
        print("\n🔍 Scanning Knowledge Base and Auditing...")
        report = audit_protocol(my_trial, kb)
        print("\n📝 AUDIT REPORT:")
        print(report)