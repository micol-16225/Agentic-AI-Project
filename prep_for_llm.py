import pandas as pd
import json

# --- Configuration ---
CSV_FILENAME = "fda_letters.csv"
OUTPUT_PROMPTS_FILENAME = "llm_analysis_prompts.json"
CHUNK_SIZE = 4000  # A typical safe chunk size for processing large documents

# --- 1. The Structured Prompt Template ---
# This prompt defines the LLM's task and the desired output schema.
PROMPT_TEMPLATE = """
ROLE: You are an expert regulatory affairs specialist and clinical biostatistician. Your task is to analyze the provided segment of an FDA document (Warning Letter or Complete Response Letter).

OBJECTIVE: Identify all specific instances of protocol deviations, sponsor oversight failures, or statistical methodology flaws related to clinical trial integrity.

FOUR PILLARS OF KNOWLEDGE (Use these for the 'Pillar_Category'):
1. Statistical_Validity (ICH E9 R1 / Estimands, Analysis Choice)
2. Execution_Risk (21 CFR 312.60/62 Investigator Protocol Deviations, Documentation)
3. Methodological_Solutions (Missing Data Handling, Sensitivity Analysis flaws)
4. Trial_Design (Sample Size, Stratification, Generalizability Flaws)

DOCUMENT CONTEXT:
Document Type: {DOC_TYPE}
Document Date: {DOC_DATE}

TEXT SEGMENT TO ANALYZE:
---
{TEXT_CHUNK}
---

OUTPUT FORMAT:
Generate ONLY a single JSON array containing objects for each finding.
[
    {{
        "finding_id": "unique_id_for_this_finding",
        "violation_text": "The precise sentence(s) describing the specific flaw.",
        "Pillar_Category": "One of the four categories listed above.",
        "CFR_Section_Cited": "The specific CFR section if mentioned (e.g., 312.60).",
        "Statistical_Consequence": "Describe the statistical impact (e.g., Loss of Power, Selection Bias, Data Unreliability)."
    }},
    ...
]
"""

# --- 2. Text Chunking Function ---
def chunk_text(text, chunk_size):
    """Splits a long string into smaller, manageable chunks."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# --- 3. Main Processing Function ---
def prepare_data_for_llm(csv_filename):
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: {csv_filename} not found. Please run the data fetching script first.")
        return

    llm_prompts_list = []
    
    # Iterate through each letter in the dataframe
    for index, row in df.iterrows():
        full_text = str(row['full_text'])
        doc_type = row['doc_type']
        doc_date = row['date']
        
        # Split the document into chunks
        chunks = chunk_text(full_text, CHUNK_SIZE)
        
        print(f"Processing document {index + 1} ({doc_type}). Total chunks: {len(chunks)}")
        
        # Create a unique prompt for each chunk
        for i, chunk in enumerate(chunks):
            
            # Populate the template with the current chunk and context
            structured_prompt = PROMPT_TEMPLATE.format(
                DOC_TYPE=doc_type,
                DOC_DATE=doc_date,
                TEXT_CHUNK=chunk
            )
            
            llm_prompts_list.append({
                'document_id': f"doc_{index}_{row['recipient']}",
                'chunk_id': i + 1,
                'prompt': structured_prompt
            })
            
            # NOTE: In a real system, you would send structured_prompt to the LLM API here.
            
    # Save the resulting list of structured prompts
    with open(OUTPUT_PROMPTS_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(llm_prompts_list, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Successfully prepared {len(llm_prompts_list)} structured prompts.")
    print(f"The file '{OUTPUT_PROMPTS_FILENAME}' contains the input for your LLM analysis.")


# --- Execution Block ---
if __name__ == "__main__":
    prepare_data_for_llm(CSV_FILENAME)