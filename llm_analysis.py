import os
import json
import google.generativeai as genai
from google.generativeai import types
from pydantic import BaseModel, Field
from typing import List

# --- Configuration ---
RAW_LETTERS_DIR = 'warning_letters_text'  # Directory where your letters are saved
OUTPUT_JSON_DIR = 'llm_analyzed_json'      # Directory to save the structured JSON
PHASE_1_COUNT = 50 

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# Initialize Gemini Client. It automatically reads the GEMINI_API_KEY environment variable.
try:
    client = Client()
except Exception as e:
    print("FATAL ERROR: Could not initialize Gemini client.")
    print("Ensure 'pip install google-genai' and the GEMINI_API_KEY environment variable is set.")
    exit()

# --- 2. Define the Structured Output Schema using Pydantic ---
class Violation(BaseModel):
    Violation_Category: str = Field(
        description="Broad area: cGMP, Misbranding, or Clinical."
    )
    Specific_Regulation_Code: str = Field(
        description="The exact CFR or U.S.C. code cited (e.g., '21 CFR 211.22(d)')."
    )
    CFR_Code_Snippet: str = Field(
        description="The relevant, direct text quote from the letter where the code is cited."
    )
    Actionable_Pattern: str = Field(
        description="A concise, human-readable pattern of the failure (e.g., 'Failure to establish written procedures')."
    )
    Bias_Classification: str = Field(
        description="Statistical classification: Procedural, Technical, or Personnel."
    )

class WarningLetterAnalysis(BaseModel):
    """A list of violations found in a single warning letter."""
    Violations: List[Violation]

# --- 3. The Prompt Template ---
SYSTEM_PROMPT = """
You are an expert FDA regulatory analyst. Your sole task is to analyze the provided Warning Letter text and extract a structured list of every unique violation cited, strictly adhering to the provided JSON schema.

**CLASSIFICATION RULES:**
* **Violation_Category:** Must be one of: "cGMP", "Misbranding", or "Clinical".
* **Bias_Classification:** Must be one of: "Procedural" (documentation/SOPs), "Technical" (equipment/testing/methods), or "Personnel" (staffing/training/authority).
"""

def analyze_letter_with_llm(letter_text: str, filename: str) -> bool:
    """Sends a single letter to the LLM for structured analysis and saves the result."""
    
    user_message = f"""
    Please analyze the following FDA Warning Letter text and extract the violations.
    
    **WARNING LETTER TEXT TO ANALYZE:**
    ---
    {letter_text}
    ---
    """
    
    print(f"-> Analyzing {filename}...")

    try:
        # Request structured JSON output using the Pydantic schema
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[SYSTEM_PROMPT, user_message],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=WarningLetterAnalysis,
            ),
        )

        output_json = response.text
        
        # Save the structured data
        output_path = os.path.join(OUTPUT_JSON_DIR, filename.replace('.txt', '.json'))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json.loads(output_json), f, indent=4)
        
        print(f"<- SUCCESS: Saved structured analysis to {output_path}")
        return True

    except Exception as e:
        print(f"<- ERROR processing {filename}: {e}")
        return False

# --- 4. Main Execution ---
if __name__ == "__main__":
    letter_files = [f for f in os.listdir(RAW_LETTERS_DIR) if f.endswith('.txt')]
    
    # Use only the first PHASE_1_COUNT letters
    phase_1_files = sorted(letter_files)[:PHASE_1_COUNT]
    
    print(f"\n--- Starting Phase 1 LLM Analysis of {len(phase_1_files)} Letters ---")
    
    success_count = 0
    
    for filename in phase_1_files:
        filepath = os.path.join(RAW_LETTERS_DIR, filename)
        
        # Read the content of the scraped letter
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Run the analysis
        if analyze_letter_with_llm(content, filename):
            success_count += 1
            
    print("\n--- Phase 1 Analysis Complete ---")
    print(f"Successfully processed and structured {success_count} letters.")