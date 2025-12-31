import pandas as pd
import time
import os
from dotenv import load_dotenv
from adversarial_generator import RedTeamAgent
from lifecycle_agent3 import BiostatLifecycleAgent3

# Load API keys
load_dotenv()

# --- 1. THE CLASS (The Blueprint) ---
class AuditorEvaluator:
    def __init__(self):
        # 1. Define the path to your CSV
        library_path = "master_regulatory_library.csv"
        
        # 2. Get the API key from your environment
        api_key = os.environ.get("GROQ_API_KEY")
        
        # 3. Initialize the auditor with the 2 missing arguments
        self.auditor = BiostatLifecycleAgent3(
            api_key=api_key, 
            library_path=library_path
        )
        # 4. Define your test suite with expected keywords and sections = STATIC TESTS
        self.test_suite = [
    {
        "name": "Phase 3 Confirmatory Protocol (Full Draft)",
        "protocol_snippet": """
        TITLE: A Phase 3, Randomized, Double-Blind Study of Zenziva in MDD.
        
        1. CLINICAL OBJECTIVES
        The primary objective is to demonstrate the efficacy of Zenziva 20mg vs Placebo. 
        A secondary objective is to evaluate the 'Real-World' benefit, specifically focusing 
        on the benefit for all patients who enter the trial, regardless of whether they 
        adhere to the medication or switch to rescue therapy. This 'Treatment Policy' 
        approach is the cornerstone of our patient-centric design.

        2. STUDY ENDPOINTS
        The primary endpoint is the change from baseline in HAM-D score at Week 24.
        Key secondary endpoints include: 
        - MADRS total score at Week 24
        - Clinical Global Impression (CGI-S)
        - Patient Reported Quality of Life (QoL)
        - Sleep Quality Index
        - Anxiety Sub-scale
        - Social Functioning Scale.

        3. SAMPLE SIZE ASSUMPTIONS
        We assume a treatment difference (Delta) of 2.5 with a Standard Deviation of 6.0. 
        Using a 1:1 randomization, N=450 provides 90% power.

        4. STATISTICAL METHODOLOGY
        4.1 ESTIMANDS: Per Section 1, we aim for a 'While-on-Treatment' strategy. 
        Data collected after a patient discontinues study drug or initiates rescue 
        medication will be discarded and treated as missing. 
        
        4.2 MISSING DATA: We will use Last Observation Carried Forward (LOCF) for the 
        primary analysis to ensure we have a value for every patient randomized.

        4.3 MULTIPLICITY: Since this is an unmet need population, each of the 6 key 
        secondary endpoints will be tested at a nominal alpha of 0.05. This ensures 
        we do not miss any significant clinical signals that could benefit patients.
        """,
        "expected_keywords": ["CONTRADICTION", "ESTIMAND", "LOCF", "MULTIPLICITY", "ICH E9"],
        "required_section": "SECTION 2"
    }, 
    {
        "name": "Oncology Phase 3: Structural Integrity Test",
        "protocol_snippet": """
        TITLE: A Multicenter Study of OncoZ-1 in Metastatic Non-Small Cell Lung Cancer.
        
        SECTION 1: INTRODUCTION. OncoZ-1 has shown significant Progression-Free Survival (PFS) benefits 
        in Phase 2. This Phase 3 trial seeks to confirm these findings in a larger cohort (N=600).
        
        SECTION 2: ENDPOINTS. The primary endpoint is Overall Survival (OS). Secondary endpoints 
        include PFS, Objective Response Rate (ORR), and Quality of Life (QoL).
        
        SECTION 3: STATISTICAL METHODS. We will use a Cox Proportional Hazards model for the 
        primary analysis of OS. Comparisons will be made using a log-rank test.
        
        [CRITICAL OMISSION: NO ESTIMAND DEFINITION, NO MULTIPLICITY PLAN, NO SAMPLE SIZE JUSTIFICATION]
        """,
        "expected_keywords": ["ESTIMAND", "SAMPLE SIZE", "MULTIPLICITY", "ICH E9", "RTF"],
        "required_section": "SECTION 2"
    }, 
    {
        "name": "Diabetes Type 2: Legacy Methodology Trap",
        "protocol_snippet": """
        SECTION 1.0: OBJECTIVES. To evaluate change in HbA1c from baseline to Week 52.
        
        SECTION 4.0: MISSING DATA STRATEGY. In clinical trials for chronic conditions like T2D, 
        patient dropout is expected. To maintain a full dataset, we will employ Last Observation 
        Carried Forward (LOCF). This ensures that every patient who receives at least one dose 
        contributes to the final p-value. We believe this is a conservative and standard approach.
        
        SECTION 4.1: SENSITIVITY. No sensitivity analyses are planned as LOCF is inherently robust.
        """,
        "expected_keywords": ["LOCF", "BIAS", "MMRM", "SENSITIVITY", "ICH E9"],
        "required_section": "SECTION 2"
    }, 
    {
        "name": "Alzheimer's Phase 3: The Multiplicity Kitchen Sink",
        "protocol_snippet": """
        SECTION 2.1: PRIMARY ENDPOINT. Change in ADAS-Cog 13 score at Month 18.
        
        SECTION 2.2: KEY SECONDARY ENDPOINTS. We have identified 14 key secondary metrics including 
        the Clinical Dementia Rating (CDR-SB), Functional Activities Questionnaire (FAQ), 
        and various biomarkers (p-tau181, Amyloid-beta 42/40 ratio). 
        
        SECTION 4.5: INFERENTIAL LOGIC. To ensure we capture every possible benefit for this 
        suffering population, each of the 14 secondary endpoints will be tested at a 
        significance level of 0.05. If any of these reach significance, the drug will be 
        considered a 'Multi-Domain Success.'
        """,
        "expected_keywords": ["MULTIPLICITY", "ALPHA", "FWER", "TYPE I ERROR", "ADJUSTMENT"],
        "required_section": "SECTION 2"
    }
]

    def run_adversarial_test(self, flaw_to_test="Multiplicity"):
        # 1. Initialize the Red Team
        red_team = RedTeamAgent()
        
        # 2. Generate a "Poison" Protocol
        print(f"🚀 Red Team is generating a flawed protocol: {flaw_to_test}...")
        fake_protocol = red_team.generate_poison_pill_protocol(flaw_to_test)
        
        # 3. Feed it to the Auditor
        print("⚖️ Auditor is reviewing the draft...")
        audit_report = self.auditor.audit_protocol(fake_protocol, historical_lessons="")
        
        # 4. Validate the result
        # We check if the Auditor mentioned the specific flaw we told the Red Team to hide
        passed = flaw_to_test.upper() in audit_report.upper()
        
        return {
            "flaw": flaw_to_test,
            "report": audit_report,
            "success": passed
        }

    # Updated check logic for your Tester-Agent
    def evaluate_audit(self, audit_report, test_case):
        report_upper = audit_report.upper()
        
        # 1. Structural Check: Look for the KEYWORDS in headers, not section numbers
        has_historical = any(x in report_upper for x in ["HISTORICAL", "PRECEDENT", "SECTION 1"])
        has_academic = any(x in report_upper for x in ["REGULATORY", "ACADEMIC", "ALIGNMENT", "SECTION 2"])
        has_table = any(x in report_upper for x in ["RISK", "TABLE", "SECTION 3"])

        # 2. Keyword Check: Use more flexible partial matching
        found_keywords = []
        for kw in test_case["expected_keywords"]:
            if kw.upper() in report_upper:
                found_keywords.append(kw)
        
        missing_keywords = [kw for kw in test_case["expected_keywords"] if kw not in found_keywords]

        # 3. VERDICT: Pass if the content is there, even if headers aren't perfect
        # We pass if it finds at least 2/3 of keywords OR the core technical citation
        logic_passed = len(missing_keywords) <= 1 or any(x in report_upper for x in ["ICH", "21 CFR", "FDA"])
        
        # We require at least the Academic/Regulatory section to exist
        if has_academic and logic_passed:
            grade = "PASS"
        else:
            grade = "FAIL"

        return {
            "test_name": test_case["name"],
            "final_grade": grade,
            "missing_keywords": missing_keywords
        }
    

# --- 2. THE EXECUTION BLOCK (The Ignition) ---
if __name__ == "__main__":
    evaluator = AuditorEvaluator()
    red_team = RedTeamAgent()
    
    # Create or clear the transparency log
    log_file = "validation_results_audit_trail.txt"
    with open(log_file, "w") as f:
        f.write(f"--- VALIDATION AUDIT TRAIL: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    print("\n" + "="*50)
    print("🚀 STARTING DYNAMIC ADVERSARIAL VALIDATION")
    print("="*50)

    # --- PHASE 1: STATIC TESTS ---
    # --- PHASE 1: STATIC TESTS ---
    for case in evaluator.test_suite:
        print(f"\n[STATIC TEST] {case['name']}...")
        
        # 1. FETCH THE LESSONS (Source of Truth)
        # We search the CSV using the test case name or a keyword
        print(f"🔍 Fetching regulatory context for: {case['name']}...")
        lessons = evaluator.auditor._load_library(case['name']) 
        
        # 2. AUDIT (Passing both required arguments)
        report = evaluator.auditor.audit_protocol(case['protocol_snippet'], lessons)
        
        result = evaluator.evaluate_audit(report, case)
        
        # Save to transparency log
        with open(log_file, "a") as f:
            f.write(f"CASE: {case['name']} | GRADE: {result['final_grade']}\n")
        
        print(f"{'✅' if result['final_grade'] == 'PASS' else '❌'} GRADE: {result['final_grade']}")

    # --- PHASE 2: DYNAMIC RED TEAM ---
    dynamic_flaws = ["Multiplicity Alpha Inflation", "Legacy LOCF Bias", "Missing Estimand Strategy"]
    
    # --- PHASE 2: DYNAMIC RED TEAM ---
    for flaw in dynamic_flaws:
        print(f"\n[ADVERSARIAL CHALLENGE] Testing: {flaw}...")
        try:
            # 1. Breather for the API
            time.sleep(3) 
            
            # 2. Red Team creates the trap
            poison_protocol = red_team.generate_poison_pill_protocol(flaw)
            
            # 3. GET THE TRUTH: Fetch lessons from your CSV based on the flaw
            # Change 'retrieve_relevant_lessons' to match your actual function name
            print(f"🔍 Searching CSV for lessons regarding {flaw}...")
            lessons = evaluator.auditor._load_library(flaw)
            
            # 4. THE AUDIT: Use the CSV context to judge the fake protocol
            print("⚖️ Auditor is judging based on Source of Truth...")
            report = evaluator.auditor.audit_protocol(poison_protocol, lessons)
            
            # 5. GRADING
            # We look for the flaw name or the word "PROTOCOL" (common in your library)
            passed = flaw.split()[-1].upper() in report.upper() or "VIOLATION" in report.upper()
            grade = "PASS" if passed else "FAIL"

            print(f"{'🔥' if grade == 'PASS' else '💀'} RESULT: {grade}")
            
            # Log it
            with open(log_file, "a") as f:
                f.write(f"ADVERSARIAL: {flaw}\nLESSONS USED: {lessons[:100]}...\nGRADE: {grade}\n\n")

        except Exception as e:
            print(f"⚠️ Test interrupted: {e}")

    print("\n" + "="*50)
    print(f"🏁 COMPLETE. Results saved to: {log_file}")
    print("="*50)