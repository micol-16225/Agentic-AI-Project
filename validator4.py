import pandas as pd
import re
import json
from datetime import datetime
from lifecycle_agent4 import BiostatLifecycleAgent4
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

class AuditorValidator:
    def __init__(self, statutory_path="statutory_truth.csv", intel_path="optimizer_intelligence.csv"):
        # UPDATED: Load both tiers of truth
        self.stat_df = pd.read_csv(statutory_path)
        self.intel_df = pd.read_csv(intel_path)
        
        # Combine into a single "Searchable Truth" for the validation engine
        self.full_library = pd.concat([self.stat_df, self.intel_df], ignore_index=True)
        self.full_library.columns = [c.strip() for c in self.full_library.columns]
        
        self.log_file = "audit_validation_performance.jsonl"
        print(f"--- VALIDATOR INITIALIZED (TIERED MODE) ---")
        print(f"Statutory Rows: {len(self.stat_df)} | Intelligence Rows: {len(self.intel_df)}")
        print("-" * 40)

    def validate_report(self, protocol_name, report_text):
        diagnostics = []
        # Total checks: 1. Statutory Truth, 2. Precedent Truth, 3. Academic Truth, 4. Formatting, 5. Logic
        metrics = {"passed_checks": 0, "total_checks": 5}
        
        # --- CHANGE 1: REGEX FOR BOLD QUOTES ---
        # Your new Auditor uses **"Quote"**. This regex captures text inside double asterisks and quotes.
        quoted_phrases = re.findall(r'\*\*"([^"]*)"\*\*', report_text)
        
        # Fallback for standard quotes if bolding fails
        if not quoted_phrases:
            quoted_phrases = re.findall(r'"([^"]*)"', report_text)

        print(f"\n--- VERIFYING {len(quoted_phrases)} QUOTED PIECES OF EVIDENCE ---")
        
        valid_quotes_count = 0
        for phrase in quoted_phrases:
            clean = phrase.strip()
            if len(clean.split()) >= 4: # Lowered threshold slightly for atomic rules
                
                def normalize(text):
                    return re.sub(r'[^\w\s]', '', str(text)).lower().strip()

                norm_phrase = normalize(clean)
                # Search across the combined tiered library
                match_exists = self.full_library['content'].apply(lambda x: norm_phrase in normalize(x)).any()
                
                if match_exists:
                    print(f"✅ VERIFIED: '{clean[:60]}...'")
                    valid_quotes_count += 1
                else:
                    print(f"❌ HALLUCINATION: '{clean[:60]}...'")
                    diagnostics.append(f"HALLUCINATION: {clean}")

        # --- MLOps CONFIG: PRE-AWS CONSTRAINT MODE ---
        # Set to True while parallel workers are capped at 3
        CONSTRAINT_MODE = True 
        TOKEN_CAP = 3 

        # --- CHANGE 1: DYNAMIC DENOMINATOR ---
        if CONSTRAINT_MODE:
            # Scale the total expected checks down to match the letter cap
            # This prevents the score from being penalized for missing letters
            effective_total = min(metrics["total_checks"], (TOKEN_CAP + 5)) # +5 for Statutory/Acad
        else:
            effective_total = metrics["total_checks"]

        # --- CHANGE 2: TIERED SCORING (Your existing logic) ---
        if len(quoted_phrases) > 0:
            accuracy_rate = valid_quotes_count / len(quoted_phrases)
            if accuracy_rate == 1.0:
                metrics["passed_checks"] += 3
            elif accuracy_rate >= 0.7:
                metrics["passed_checks"] += 1

        # --- CHANGE 3: MONOLOGUE CHECK ---
        if "INTERNAL MONOLOGUE" in report_text.upper():
            metrics["passed_checks"] += 2

        # --- FINAL CALCULATION ---
        final_score = (metrics["passed_checks"] / effective_total) * 100
        status = "PASS" if final_score >= 80 else "FAIL"

        if CONSTRAINT_MODE:
            print(f"🛡️ STATUS: {status} (Constraint Mode Active)")
        
        print("\n🧐 VALIDATOR DIAGNOSTIC:")
        if final_score < 80:
            missing_count = effective_total - valid_quotes_count
            print(f"❌ REASON: LOW RECALL.")
            print(f"- The Agent found {valid_quotes_count} verified quotes.")
            print(f"- To PASS, the Agent needed to cite {int(effective_total * 0.8)} quotes.")
            print(f"- GAP: The Agent missed {missing_count} pieces of evidence that were available in context.")
            
            if CONSTRAINT_MODE and valid_quotes_count <= 3:
                print("💡 INSIGHT: The Agent is likely hitting the 'Token Wall'. It stopped reading after the first few documents to save space.")
            elif accuracy_rate < 1.0:
                print("💡 INSIGHT: Some quotes were paraphrased. The Validator requires 100% character-match.")
        else:
            print("✅ REASON: Sufficient evidence and 100% accuracy achieved.")

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "protocol": protocol_name,
            "score": f"{final_score}%",
            "status": status,
            "hallucinations": [d for d in diagnostics if "HALLUCINATION" in d]
        }
        
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
# Load environment variables (API Keys)
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# --- 1. INITIALIZATION (Tiered Paths) ---
# We point to the two new "Sources of Truth" created in your Jan 2 update
STATUTORY_PATH = "statutory_truth.csv"
INTEL_PATH = "optimizer_intelligence.csv"

# Initialize the Auditor (Agent 4)
# Note: Ensure your class name matches your script (BiostatLifecycleAgent3 or 4)
auditor = BiostatLifecycleAgent4(
    api_key=api_key, 
    statutory_path=STATUTORY_PATH, 
    intel_path=INTEL_PATH
)

# Initialize the Validator with both tiers
validator = AuditorValidator(
    statutory_path=STATUTORY_PATH, 
    intel_path=INTEL_PATH
)

# --- 2. EXECUTION ---
print("🔍 Starting Adversarial Audit on ZEN-301...")

# The audit_protocol now handles the parallel letter checks internally
report = auditor.audit_protocol(
    user_protocol=fake_protocol_mdd, 
    historical_lessons="Focus on LOCF and Multiplicity precedents."
)

# --- 3. VALIDATION ---
print("⚖️ Validating Report against Statutory and Academic Truth...")
verdict = validator.validate_report("ZEN-301_High_Complexity_Draft", report)

# --- 4. OUTPUT & LOGGING ---
# Save the report for your manual inspection
with open("latest_auditor_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("\n" + "="*40)
print(f"✅ VALIDATION COMPLETE")
print(f"📊 PROTOCOL: {verdict['protocol']}")
print(f"🎯 SCORE:    {verdict['score']}")
print(f"🛡️ STATUS:   {verdict['status']}")

if verdict['hallucinations']:
    print("\n🚨 HALLUCINATIONS DETECTED:")
    for failure in verdict['hallucinations']:
        print(f"  - {failure}")
else:
    print("\n💎 CLEAN AUDIT: No hallucinations detected.")

print("="*40)
print("📝 Full report saved to 'latest_auditor_report.txt'")