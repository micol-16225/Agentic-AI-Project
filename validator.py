import pandas as pd
import re
import json
from datetime import datetime
from lifecycle_agent3 import BiostatLifecycleAgent3 
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# --- 1. THE EVALUATOR CLASS (The Judge) ---
class AuditorValidator:
    def __init__(self, library_path="master_regulatory_library.csv"):
        self.library = pd.read_csv(library_path)
        self.library.columns = [c.strip() for c in self.library.columns]
        self.log_file = "audit_validation_performance.jsonl"
        
        # --- NEW: Print Library Stats on Init ---
        print(f"--- VALIDATOR INITIALIZED ---")
        print(f"Library Rows Loaded: {len(self.library)}")
        print(f"Sample 'Truths' expected: {self.library['content'].head(3).tolist()}")
        print("-" * 30)

    def validate_report(self, protocol_name, report_text):
        diagnostics = []
        metrics = {"passed_checks": 0, "total_checks": 5}
        
        # --- NEW LOGIC: Extract text inside " " instead of ** ** ---
        quoted_phrases = re.findall(r'"([^"]*)"', report_text)
        
        print(f"\n--- VERIFYING QUOTED EVIDENCE ---")
        valid_quotes = []
        for phrase in quoted_phrases:
            clean = phrase.strip()
            if len(clean.split()) >= 6:
                # NORMALIZATION: Remove punctuation and lower-case for the check
                def normalize(text):
                    return re.sub(r'[^\w\s]', '', text).lower().strip()

                norm_phrase = normalize(clean)
                # Check if the normalized phrase exists in any normalized library entry
                match_exists = self.library['content'].apply(lambda x: norm_phrase in normalize(str(x))).any()
                
                if match_exists:
                    print(f"✅ VERIFIED QUOTE: '{clean[:50]}...'")
                    valid_quotes.append(clean)
                else:
                    print(f"❌ HALLUCINATED QUOTE: '{clean[:50]}...'")
                    diagnostics.append(f"HALLUCINATION: {clean}")

        if valid_quotes and not any("HALLUCINATION" in d for d in diagnostics):
            metrics["passed_checks"] += 2 # Heavy weight on truthfulness

        # Monologue Check
        monologue_passed = "DOES THE PROTOCOL MENTION" in report_text.upper()
        if monologue_passed: metrics["passed_checks"] += 2

        final_score = (metrics["passed_checks"] / metrics["total_checks"]) * 100
        status = "PASS" if final_score >= 80 else "FAIL"

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "protocol": protocol_name,
            "score": f"{final_score}%",
            "status": status,
            "failures": diagnostics
        }
        
        # Append to log
        with open(self.log_file, "a") as f:
            f.write(json.dumps(result) + "\n")
            
        return result

# --- 2. THE TEST EXECUTION (How you run it) ---

# In your Runner script:
fake_protocol_mdd = """
TITLE: A Phase 3, Multicenter, Randomized, Double-Blind, Placebo-Controlled Study to Evaluate the Efficacy and Safety of Zenziva (ZN-202) in Subjects with Moderate-to-Severe Major Depressive Disorder (MDD).

6. STUDY ENDPOINTS AND OBJECTIVES
6.1 Primary Objective
The primary objective of this study is to evaluate the clinical efficacy of Zenziva 20mg once daily compared to placebo in reducing depressive symptoms. The primary endpoint is the change from baseline in the Montgomery-Asberg Depression Rating Scale (MADRS) total score at Week 24.

6.2 Secondary Objectives (The Multiplicity Trap)
To provide a comprehensive assessment of the Zenziva clinical profile, the following secondary objectives will be assessed at Week 24. To ensure maximum sensitivity to clinical signals in this underserved population, each will be evaluated at a nominal significance level of 0.05:
1. Change from baseline in the Hamilton Rating Scale for Depression (HAM-D 17).
2. Proportion of subjects achieving clinical remission (MADRS ≤ 10).
3. Change from baseline in the Sheehan Disability Scale (SDS) total score.
4. Improvement in the Pittsburgh Sleep Quality Index (PSQI).
5. Change in the Generalized Anxiety Disorder 7-item (GAD-7) scale.
6. Evaluation of the Quality of Life Enjoyment and Satisfaction Questionnaire (QLESQ).

9. STATISTICAL METHODOLOGY AND SAMPLE SIZE
9.1 Estimand Definition (The Contradiction Trap)
Consistent with a 'Patient-Centric' approach, the primary estimand follows a Treatment-Policy Strategy. This ensures that the treatment effect estimated reflects the benefit for all randomized patients, regardless of their adherence to the study medication or the initiation of rescue psychiatric therapy.

9.4 Handling of Missing Data (The Legacy Trap)
While every effort will be made to retain subjects in the trial, some attrition is anticipated. For the primary analysis of the MADRS change from baseline, missing values at Week 24 will be handled via the Last Observation Carried Forward (LOCF) method. This approach ensures that the statistical power calculated at N=450 is maintained and that every patient randomized contributes a final value to the efficacy analysis, providing a conservative and stable estimate of the drug effect.

9.5 Statistical Model (The Estimand Conflict)
Notwithstanding Section 9.1, the primary analysis will utilize a 'While-on-Treatment' logic. Data points collected after the discontinuation of the study drug or following the administration of prohibited rescue medications (as defined in Section 5.4) will be excluded from the primary efficacy set and treated as missing. This aligns the analysis with the physiological impact of the drug.
"""

# 1. Run your Auditor

api_key = os.getenv("API_KEY")
auditor = BiostatLifecycleAgent3(
    api_key=api_key, 
    library_path="master_regulatory_library.csv",
)
validator = AuditorValidator()

report = auditor.audit_protocol(fake_protocol_mdd, historical_lessons="")

# 2. Run the Validator on the resulting report
verdict = validator.validate_report("ZEN-301_High_Complexity_Draft", report)

# 3. Save the actual report for inspection
with open("latest_auditor_report.txt", "w") as f:
    f.write(report)

print(f"Validation Complete. Score: {verdict['score']}")
print("Full report saved to 'latest_auditor_report.txt' for inspection.")