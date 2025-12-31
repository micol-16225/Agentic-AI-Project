import pandas as pd
import json
import time
import os
import re
import google.generativeai as genai

# -------------------------------
# Configure Gemini API
# -------------------------------
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# -------------------------------
# Configuration
# -------------------------------
CSV_FILENAME = "fda_letters.csv"
OUTPUT_ANALYSIS_FILENAME = "llm_master_analysis.json"
CHUNK_SIZE = 4000
RATE_LIMIT_DELAY = 1  # seconds between API calls

# -------------------------------
# Prompt Template
# -------------------------------
PROMPT_TEMPLATE = """
ROLE: You are an expert regulatory affairs specialist and clinical biostatistician. Your task is to analyze the provided segment of an FDA document (Warning Letter or Complete Response Letter).

OBJECTIVE: Identify and categorize all specific instances of protocol deviations, sponsor oversight failures, or statistical methodology flaws related to clinical trial integrity.

FOUR PILLARS OF KNOWLEDGE (Use one of these for the 'Pillar_Category'):
1. Statistical_Validity: ICH E9 R1 / Estimands, Analysis Choice, Hypothesis Testing.
2. Execution_Risk: 21 CFR 312.60/62 Investigator Protocol Deviations, Documentation Failures.
3. Methodological_Solutions: Missing Data Handling, Sensitivity Analysis flaws, Biased Imputation.
4. Trial_Design: Sample Size, Stratification, Generalizability (I/E) Flaws.

DOCUMENT CONTEXT:
Document Type: {DOC_TYPE}
Document Date: {DOC_DATE}

TEXT SEGMENT TO_ANALYZE:
---
{TEXT_CHUNK}
---

OUTPUT FORMAT:
Generate ONLY a single JSON array of objects for all findings in the segment.
[
    {{
        "finding_id": "unique_identifier",
        "violation_text": "The precise sentence(s) describing the specific flaw.",
        "Pillar_Category": "One of the four categories listed above.",
        "CFR_Section_Cited": "The specific CFR section if mentioned (e.g., 312.60).",
        "Statistical_Consequence": "Describe the statistical impact (e.g., Loss of Power, Selection Bias, Data Unreliability)."
    }},
    ...
]
"""

# -------------------------------
# Gemini API Call (updated for latest client)
# -------------------------------
def call_llm_api(prompt):
    print(f"  [API Call] Sending {len(prompt)} characters to Gemini...")
    time.sleep(RATE_LIMIT_DELAY)

    try:
        response = genai.generate_text(
            model="gemini-1.5-pro",
            prompt=prompt,
            temperature=0.2
        )
        llm_response_text = response.text if hasattr(response, "text") else "[]"
        # Clean up code fences
        llm_response_text = re.sub(r"```json|```", "", llm_response_text, flags=re.IGNORECASE).strip()
        return llm_response_text
    except Exception as e:
        print(f"  [Error] Gemini API call failed: {e}")
        return "[]"

# -------------------------------
# Chunking function
# -------------------------------
def chunk_text(text, chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# -------------------------------
# Main Pipeline
# -------------------------------
def run_llm_analysis_pipeline(csv_filename):
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: {csv_filename} not found.")
        return

    master_analysis_list = []

    print(f"Starting LLM analysis on {len(df)} documents...")

    for index, row in df.iterrows():
        full_text = str(row['full_text'])
        doc_type = row['doc_type']
        doc_date = row['date']
        recipient = row['recipient'] if pd.notna(row['recipient']) else "Unknown"

        chunks = chunk_text(full_text, CHUNK_SIZE)
        doc_findings = []

        print(f"\nAnalyzing Document {index + 1}/{len(df)}: {recipient} ({doc_type}). Chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            structured_prompt = PROMPT_TEMPLATE.format(
                DOC_TYPE=doc_type,
                DOC_DATE=doc_date,
                TEXT_CHUNK=chunk
            )

            llm_response_text = call_llm_api(structured_prompt)

            # Safely parse JSON
            try:
                findings = json.loads(llm_response_text)
            except json.JSONDecodeError:
                print(f"  [Warning] Failed to parse JSON for chunk {i+1}. Preview:")
                print(llm_response_text[:500])
                findings = []

            # Add metadata
            for finding in findings:
                finding['source_doc_id'] = index
                finding['source_recipient'] = recipient
                finding['source_chunk'] = i + 1
                doc_findings.append(finding)

        master_analysis_list.extend(doc_findings)
        print(f"Total findings extracted so far: {len(master_analysis_list)}")

    # Save master JSON
    print(f"\nAnalysis complete. Saving to {OUTPUT_ANALYSIS_FILENAME}")
    with open(OUTPUT_ANALYSIS_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(master_analysis_list, f, ensure_ascii=False, indent=4)

    print(f"✅ Done! Total findings: {len(master_analysis_list)}")

# -------------------------------
# Run script
# -------------------------------
if __name__ == "__main__":
    run_llm_analysis_pipeline(CSV_FILENAME)
