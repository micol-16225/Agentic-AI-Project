import pandas as pd
import time
import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv

# Updated imports to reflect your refactored classes
from adversarial_generator import RedTeamAgent  
from lifecycle_agent4 import BiostatLifecycleAgent4 

load_dotenv()

# --- 1. THE UPDATED VALIDATOR ---
class AuditorValidator:
    def __init__(self, statutory_path="statutory_truth.csv", intel_path="optimizer_intelligence.csv"):
        # Load both tiers to ensure every quote has a source
        s_df = pd.read_csv(statutory_path)
        i_df = pd.read_csv(intel_path)
        self.library = pd.concat([s_df, i_df], ignore_index=True)
        self.library.columns = [c.strip() for c in self.library.columns]
        self.log_file = "audit_validation_performance.jsonl"
        
        print(f"--- VALIDATOR INITIALIZED (TIERED) ---")
        print(f"Total Truth Records: {len(self.library)}")
        print("-" * 30)

    def validate_report(self, protocol_name, report_text):
        diagnostics = []
        metrics = {"passed_checks": 0, "total_checks": 5}
        
        # CHANGE: Look for BOLD QUOTES **"..."** to match the new Auditor prompt
        quoted_phrases = re.findall(r'\*\*"([^"]*)"\*\*', report_text)
        
        # Fallback to standard quotes if bolding is missing
        if not quoted_phrases:
            quoted_phrases = re.findall(r'"([^"]*)"', report_text)
        
        print(f"\n--- VERIFYING EVIDENCE: {protocol_name} ---")
        valid_count = 0
        
        for phrase in quoted_phrases:
            clean = phrase.strip()
            if len(clean.split()) >= 4:
                def normalize(text):
                    return re.sub(r'[^\w\s]', '', str(text)).lower().strip()

                norm_phrase = normalize(clean)
                match_exists = self.library['content'].apply(lambda x: norm_phrase in normalize(x)).any()
                
                if match_exists:
                    valid_count += 1
                    print(f"✅ VERIFIED: '{clean[:50]}...'")
                else:
                    diagnostics.append(f"HALLUCINATION: {clean}")
                    print(f"❌ FAILED: '{clean[:50]}...'")

        # SCORING LOGIC (Hardened for Task #3)
        if len(quoted_phrases) > 0 and (valid_count / len(quoted_phrases)) >= 0.9:
            metrics["passed_checks"] += 3 # 90%+ accuracy required for the 60% grade
        
        if "INTERNAL MONOLOGUE" in report_text.upper():
            metrics["passed_checks"] += 2 # Professional structure check

        final_score = (metrics["passed_checks"] / metrics["total_checks"]) * 100
        status = "PASS" if final_score >= 80 else "FAIL"

        result = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "score": f"{final_score}%", "status": status}
        with open(self.log_file, "a") as f:
            f.write(json.dumps({**result, "protocol": protocol_name, "failures": diagnostics}) + "\n")
            
        return result

# --- 2. THE UPDATED EVALUATOR ---
class AuditorEvaluator:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        # Initialize Agent with Tiered Logic
        self.auditor = BiostatLifecycleAgent4(
            api_key=self.api_key, 
            library_path="statutory_truth.csv" # Defaulting to truth
        )
        self.validator = AuditorValidator()
        # self.red_team = RedTeamAgent() 

        # Task #3: Define common regulatory failure modes
        self.test_suite = [
            {
                "name": "Statutory: Estimand Alignment",
                "text": "We will use ITT but treat drop-outs as missing without defining intercurrent events.",
                "type": "Statutory"
            },
            {
                "name": "Academic: MMRM Structure",
                "text": "Primary analysis will use MMRM with an UN structured covariance matrix.",
                "type": "Academic_Rigor"
            }
        ]

    def run_dynamic_test(self, protocol_text):
        """Standardizes the run for both Static and Red Team loops."""
        # Use the auditor's orchestrator which handles the tiered loading internally
        return self.auditor.audit_protocol(protocol_text)

# --- 3. THE EXECUTION ---
if __name__ == "__main__":
    eval = AuditorEvaluator()
    print("\n🚀 STARTING TIERED VALIDATION MARATHON\n" + "="*50)

    # LOOP 1: STATIC TESTS
    for case in eval.test_suite:
        print(f"\n[TESTING]: {case['name']}")
        report = eval.run_dynamic_test(case['text'])
        result = eval.validator.validate_report(case['name'], report)
        print(f"RESULT: {result['status']} ({result['score']})")
        time.sleep(12) # TPM Protection

    # LOOP 2: DYNAMIC (Red Team - if available)
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

    print("\n🏁 MARATHON COMPLETE. Performance logged in audit_validation_performance.jsonl")