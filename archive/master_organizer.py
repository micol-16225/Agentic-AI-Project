import pandas as pd
import os

MASTER_FILE = "master_regulatory_library.csv"
HISTORICAL_FILE = "fda_letters.csv"

def build_ultimate_library():
    print("🚀 Building the Full-Scale Regulatory & Academic Knowledge Base...")
    library_data = []

    # --- TRACK 1: STATUTORY & REGULATORY LAW (The Legal Floor) ---
    statutory = [
        {"title": "ICH E9 (R1)", "source_type": "Statutory", "content": "The Estimand Framework: Requires precise definition of 'what' is being estimated. Must specify strategies for Intercurrent Events: Treatment-policy, Hypothetical, Composite, While-on-treatment, and Stratum-plus-variable."},
        {"title": "ICH E9", "source_type": "Statutory", "content": "Statistical Principles for Clinical Trials: MANDATES pre-specification of primary/secondary endpoints, ITT population usage, and Type I Error (Alpha) control."},
        {"title": "21 CFR Part 314.126", "source_type": "Statutory", "content": "Defines 'adequate and well-controlled' studies. Failure to provide valid statistical evidence is grounds for a Refusal to File (RTF)."},
        {"title": "21 CFR Part 312.62", "source_type": "Statutory", "content": "Investigator Recordkeeping: Requires accurate, complete, and verifiable case histories. Statistical validity is void if source docs don't match database."},
        {"title": "FDA Multiplicity Guidance", "source_type": "Statutory", "content": "Requires Family-Wise Error Rate (FWER) control. Failure to adjust for multiplicity in secondary endpoints is considered data-dredging."},
        {"title": "BIMO Compliance", "source_type": "Statutory", "content": "Bioresearch Monitoring: Focuses on data integrity, operational bias, and ensuring a 'Firewall' between unblinded personnel and trial leads."}
    ]

    # --- TRACK 2: ELITE 100 ACADEMIC INDEX (The Methodological Rigor) ---
    academic = [
        {"title": "Rubin (1976) Biometrika", "source_type": "Academic", "content": "Inference and Missing Data: Defines MAR, MCAR, and MNAR. If drop-outs are treatment-related, MAR assumptions are invalid."},
        {"title": "Benjamini & Hochberg (1995)", "source_type": "Academic", "content": "Controlling the False Discovery Rate (FDR): Critical for trials with multiple secondary endpoints or subgroups."},
        {"title": "Mehta & Pocock (2011) Stat Med", "source_type": "Academic", "content": "Adaptive Designs: The 'Promising Zone' logic for Sample Size Re-estimation (SSR) without statistical penalty."},
        {"title": "O'Brien & Fleming (1979)", "source_type": "Academic", "content": "Group Sequential Designs: Alpha-spending functions to allow interim looks without inflating Type I Error."},
        {"title": "Akacha et al. (2017)", "source_type": "Academic", "content": "Estimands in practice: Standardizes the handling of rescue medication and treatment discontinuation in longitudinal models."},
        {"title": "Rothwell (2005) Lancet", "source_type": "Academic", "content": "External Validity: Warning against 'Sanitized' trial populations that don't represent real-world clinical practice."},
        {"title": "Rosenbaum & Rubin (1983)", "source_type": "Academic", "content": "Propensity Scoring: Required for causal inference when analyzing non-randomized treatment switches."}
    ]

    # --- TRACK 3: ARCHITECT DESIGN PRINCIPLES (The Agent Intelligence) ---
    architect = [
        {"title": "Rule of Parsimony", "source_type": "Architect", "content": "Minimize exclusion criteria to maximize generalizability. Prioritize clinical endpoints over surrogates."},
        {"title": "HITL Verification Protocol", "source_type": "Architect", "content": "Mandatory Gates: G-01 Feasibility, G-02 Risk Appetite, G-03 Ethics, G-04 MCID Assumptions."}
    ]

    # Combine Statutory, Academic, and Architect tracks
    for entry in statutory + academic + architect:
        library_data.append(entry)

    # --- TRACK 4: HISTORICAL FDA LETTERS (The Precedents) ---
    if os.path.exists(HISTORICAL_FILE):
        hist_df = pd.read_csv(HISTORICAL_FILE)
        for _, row in hist_df.iterrows():
            library_data.append({
                "title": f"FDA CRL: {row['recipient']}",
                "source_type": "FDA_Letter",
                "content": f"Date: {row['date']} | Issue: {row['full_text']}"
            })
        print(f"✅ Integrated {len(hist_df)} Historical FDA Precedents.")

    # Final Unification
    final_df = pd.DataFrame(library_data)
    final_df = final_df.drop_duplicates(subset=['content'])
    final_df.to_csv(MASTER_FILE, index=False)
    print(f"🏁 MASTER LIBRARY COMPLETE: {len(final_df)} records across 4 specialized tracks.")

if __name__ == "__main__":
    build_ultimate_library()