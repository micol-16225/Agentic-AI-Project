import pandas as pd
import time
import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv

# Ensure these files exist in your directory
from adversarial_generator import RedTeamAgent
from lifecycle_agent3 import BiostatLifecycleAgent3

# Load API keys
load_dotenv()

# --- 1. THE VALIDATOR (The Judge) ---
class AuditorValidator:
    def __init__(self, library_path="master_regulatory_library.csv"):
        self.library = pd.read_csv(library_path)
        self.library.columns = [c.strip() for c in self.library.columns]
        self.log_file = "audit_validation_performance.jsonl"
        
        print(f"--- VALIDATOR INITIALIZED ---")
        print(f"Library Rows Loaded: {len(self.library)}")
        print("-" * 30)

    def validate_report(self, protocol_name, report_text):
        diagnostics = []
        metrics = {"passed_checks": 0, "total_checks": 5}
        
        # Regex to find anything inside double quotes
        quoted_phrases = re.findall(r'"([^"]*)"', report_text)
        
        print(f"\n--- VERIFYING EVIDENCE FOR: {protocol_name} ---")
        valid_quotes = []
        
        for phrase in quoted_phrases:
            clean = phrase.strip()
            if len(clean.split()) >= 5: # Valid quotes are usually 5+ words
                def normalize(text):
                    return re.sub(r'[^\w\s]', '', str(text)).lower().strip()

                norm_phrase = normalize(clean)
                # Check if this exact phrase exists in your 107-record library
                match_exists = self.library['content'].apply(lambda x: norm_phrase in normalize(x)).any()
                
                if match_exists:
                    print(f"✅ VERIFIED: '{clean[:50]}...'")
                    valid_quotes.append(clean)
                else:
                    print(f"❌ HALLUCINATED: '{clean[:50]}...'")
                    diagnostics.append(f"HALLUCINATION: {clean}")

        # SCORING LOGIC
        if valid_quotes and not any("HALLUCINATION" in d for d in diagnostics):
            metrics["passed_checks"] += 3 # Truthfulness is 60% of the grade
        
        if "SECTION" in report_text.upper() or "RISK" in report_text.upper():
            metrics["passed_checks"] += 2 # Formatting is 40%

        final_score = (metrics["passed_checks"] / metrics["total_checks"]) * 100
        status = "PASS" if final_score >= 80 else "FAIL"

        result = {"timestamp": datetime.now().strftime("%Y-%m-%d"), "score": f"{final_score}%", "status": status}
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps({**result, "protocol": protocol_name}) + "\n")
            
        return result

# --- 2. THE EVALUATOR (The Orchestrator) ---
class AuditorEvaluator:
    def __init__(self):
        library_path = "master_regulatory_library.csv"
        api_key = os.environ.get("GROQ_API_KEY")
        
        # Initialize the components correctly as attributes
        self.auditor = BiostatLifecycleAgent3(api_key=api_key, library_path=library_path)
        self.red_team = RedTeamAgent() 
        self.validator = AuditorValidator(library_path=library_path)

        self.test_suite = [
            {
                "name": "Phase 3 Confirmatory: Legacy LOCF Trap",
                "text": "The primary analysis will use LOCF to handle missing data at Week 24.",
                "type": "Legacy LOCF Bias"
            },
            {
                "name": "Oncology: Multiplicity Alpha Inflation",
                "text": "Each of the 14 secondary endpoints will be tested at a significance level of 0.05.",
                "type": "Multiplicity Alpha Inflation"
            }
        ]

    def run_audit_with_truth_constraint(self, protocol_text, lessons):
        """Forces the Auditor to use double quotes so the validator can see them."""
        prompt = f"""
        AUDIT THE FOLLOWING PROTOCOL:
        {protocol_text}

        HISTORICAL LESSONS TO APPLY:
        {lessons}

        CRITICAL REQUIREMENT: You must provide evidence by quoting the historical lessons 
        VERBATIM inside double quotes like this: "The use of LOCF is generally not acceptable".
        Failure to use double quotes for citations will result in a failed audit.
        """
        return self.auditor.audit_protocol(prompt, lessons)

# --- 3. THE EXECUTION ---
if __name__ == "__main__":
    eval = AuditorEvaluator()
    print("\n🚀 STARTING MARATHON VALIDATION\n" + "="*50)

    # LOOP 1: STATIC
    for case in eval.test_suite:
        print(f"\n[TESTING]: {case['name']}")
        lessons = eval.auditor._load_library(case['type'])
        report = eval.run_audit_with_truth_constraint(case['text'], lessons)
        result = eval.validator.validate_report(case['name'], report)
        print(f"RESULT: {result['status']} ({result['score']})")

    # LOOP 2: DYNAMIC (Red Team)
    for flaw in ["Multiplicity Alpha Inflation", "Legacy LOCF Bias"]:
        print(f"\n[DYNAMIC RED TEAM]: {flaw}")
        try:
            time.sleep(3) # Rate limit protection
            poison = eval.red_team.generate_poison_pill_protocol(flaw)
            lessons = eval.auditor._load_library(flaw)
            report = eval.run_audit_with_truth_constraint(poison, lessons)
            verdict = eval.validator.validate_report(f"Dynamic_{flaw}", report)
            print(f"DYNAMIC RESULT: {verdict['status']} ({verdict['score']})")
        except Exception as e:
            print(f"⚠️ Error: {e}")

    print("\n🏁 MARATHON COMPLETE. Check audit_validation_performance.jsonl")