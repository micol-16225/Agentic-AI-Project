
import pandas as pd
import os

# --- FILE PATHS (ML Ops Structure) ---
STATUTORY_FILE = "statutory_truth.csv"       # The "Hard Gate" for Auditor/Validator
INTELLIGENCE_FILE = "optimizer_intelligence.csv" # The "Brain" for Architect/Optimizer
HISTORICAL_FILE = "fda_letters.csv"          # Source for CRLs/Warning Letters

# 1. VERBATIM RECORDS (Statutory Law)
# Added 'source' and 'date' to ensure Auditor has full citation capability
verbatim_records = [
    {
        "title": "ICH E9 (R1) Section 3.3.1",
        "type": "Statutory",
        "source": "ICH E9 (R1)",
        "date": "2019-11-01",
        "content": "An estimand is a precise description of the treatment effect reflecting the clinical question posed by a given clinical trial objective. It summarises at a population level what the outcomes would be in the same patients under different treatment conditions being compared."
    },
    {
        "title": "ICH E9 (R1) Section 5.2",
        "type": "Statutory",
        "source": "ICH E9 (R1)",
        "date": "2019-11-01",
        "content": "The use of the last observation carried forward (LOCF) approach is generally not acceptable because it can result in biased estimates of the treatment effect and can lead to misleading conclusions."
    },
    {
        "title": "ICH E9 (R1) Section 5.1",
        "type": "Statutory",
        "source": "ICH E9 (R1)",
        "date": "2019-11-01",
        "content": "Strategies for handling intercurrent events include: Treatment-policy, Hypothetical, Composite, While-on-treatment, and Stratum-plus-variable."
    },
    {
        "title": "21 CFR 314.126(a)",
        "type": "Statutory",
        "source": "FDA 21 CFR",
        "date": "2024-04-01",
        "content": "Reports of adequate and well-controlled investigations provide the primary basis for determining whether there is 'substantial evidence' to support the claims of effectiveness for new drugs and antibiotics."
    },
    {
        "title": "21 CFR 314.126(b)(6)",
        "type": "Statutory",
        "source": "FDA 21 CFR",
        "date": "2024-04-01",
        "content": "The protocol and report of the study should describe the procedures used to accomplish minimization of bias on the part of the subjects, observers, and analysts of the data."
    },
    {
        "title": "FDA Multiplicity Guidance (2022) Section III",
        "type": "Statutory",
        "source": "FDA Guidance",
        "date": "2022-10-01",
        "content": "Procedures to control the Type I error rate should be pre-specified to ensure that the probability of making at least one Type I error does not exceed the desired level of significance."
    },
    {
        "title": "ICH E9 Section 5.2.1",
        "type": "Statutory",
        "source": "ICH E9",
        "date": "1998-02-05",
        "content": "The Full Analysis Set (FAS) should be used in the primary analysis. The FAS is as close as possible to the intention-to-treat ideal and includes all randomized subjects who received at least one dose of study medication."
    },
    {
        "title": "ICH E9 Section 5.7",
        "type": "Statutory",
        "source": "ICH E9",
        "date": "1998-02-05",
        "content": "The study protocol should identify those subgroups that are expected to be of primary interest and the statistical methods for examining them. Interactions between treatment and subgroups should be explored."
    },
    {
        "title": "FDA Multiplicity Guidance Section V",
        "type": "Statutory",
        "source": "FDA Guidance",
        "date": "2022-10-01",
        "content": "A common method to control the FWER is a hierarchical (or fixed-sequence) testing procedure where the primary and secondary endpoints are tested in a pre-specified order."
    },
    {
        "title": "ICH E3 Section 11.4.2.7",
        "type": "Statutory",
        "source": "ICH E3",
        "date": "1995-11-30",
        "content": "The SAP should describe the statistical methods for handling dropouts and missing data. If any 'ad hoc' analyses were performed, they must be clearly identified and justified."
    },
    {
        "title": "ICH E6 (R2) Section 5.5.3",
        "type": "Statutory",
        "source": "ICH E6 (R2)",
        "date": "2016-11-09",
        "content": "When using electronic trial data handling systems, the sponsor should ensure and document that the systems are designed to permit data changes in such a way that the data changes are documented and that there is no deletion of entered data (i.e., maintain an audit trail)."
    }
]

# 2. PITFALLS (Kept at Statutory level as requested)
wisdom_statutory = [
    {
        "source": "BIMO_Pitfalls", 
        "type": "Pitfall", 
        "title": "Common BIMO Pitfalls", 
        "date": "2025-01-01",
        "content": '''- Source Verification: 100% of data in the final report must match the "Source Docs" (hospital notes). '''
    }
]

# 3. INTELLIGENCE SECTIONS (Academic/Architecture/Process)
intelligence_records = [
    {
        "source": "Big_Pharma_Models", 
        "type": "Technical", 
        "title": "Statistical Formulas", 
        "content": "- MMRM (Mixed Model for Repeated Measures): $Y_i = X_i\\beta + Z_i b_i + \\epsilon_i$\n- Cox Proportional Hazards: $h(t|x) = h_0(t) \\exp(\\beta^T x)$"
    },
    {
        "source": "Design_Principles",
        "type": "Architecture",
        "title": "The Architect: Design Principles",
        "content": "Rule of Parsimony: Minimize exclusion criteria. Endpoint Hierarchy: Prioritize hard clinical endpoints."
    },
    {
        "source": "academic_Index",
        "type": "Academic_Rigor",
        "title": "Multiplicity & P-Hacking",
        "content": '''DOMAIN: MULTIPLICITY & P-HACKING (Logic: Controlling False Positives)
- Key Citations: Benjamini & Hochberg (1995) JRSSB; Westfall & Young (1993) JASA; Adda et al. (PNAS 2020).
- Expert Rule: Flag "Spikes" in p-values near 0.05. If multiple secondary endpoints are claimed as "Successes" without FDR control, cite Benjamini & Hochberg.'''
    },
    {
        "source": "academic_Index",
        "type": "Academic_Rigor",
        "title": "Missing Data & Estimands",
        "content": '''DOMAIN: MISSING DATA & ESTIMANDS (Logic: Handling Drop-outs)
- Key Citations: Rubin (1976) Biometrika; Hernán & Robins (2024) Causal Inference; ICH E9(R1).
- Expert Rule: Distinguish between MAR/MNAR. If drop-outs are treatment-related (e.g., side effects), "Missing at Random" assumptions fail. Demand sensitivity analyses per Rubin (1976).'''
    },
    {
        "source": "academic_Index",
        "type": "Academic_Rigor",
        "title": "Adaptive Designs",
        "content": '''DOMAIN: ADAPTIVE DESIGNS (Logic: The "Promising Zone")
- Key Citations: Mehta & Pocock (2011) Stat. Med.; O'Brien & Fleming (1979) Biometrics.
- Expert Rule: Sample size re-estimation (SSR) is only valid without penalty if the interim result is in the "Promising Zone." Ad-hoc doubling of N is a "Statistical Penalty" event.'''
    },
    {
        "source": "academic_Index",
        "type": "Academic_Rigor",
        "title": "External Validity",
        "content": '''DOMAIN: EXTERNAL VALIDITY (Logic: Generalizability)
- Key Citations: Rothwell (2005) Lancet; Pearl (2011) JASA.
- Expert Rule: Flag "Sanitized" populations. If exclusion criteria make the study group unrepresentative of real-world patients, cite Rothwell (2005).'''
    },
    {
        "source": "academic_Index",
        "type": "Academic_Rigor",
        "title": "Causal Inference",
        "content": '''DOMAIN: CAUSAL INFERENCE (Logic: Fairness)
- Key Citations: Rosenbaum & Rubin (1983) Biometrika; Robins et al. (2000) Epidemiology.
- Expert Rule: Ensure "Fair Comparison." Check for Propensity Score usage or G-estimation if treatment switching occurred.'''
    }
]

def hydrate_tiered():
    print("🚀 Starting Tiered Knowledge Unification...")
    
    # --- PHASE A: BUILD STATUTORY TRUTH ---
    # 1. Load Historical FDA Letters (Precedents)
    historical_df = pd.DataFrame()
    if os.path.exists(HISTORICAL_FILE):
        raw_hist = pd.read_csv(HISTORICAL_FILE)
        historical_df = pd.DataFrame({
            'source': 'FDA_Historical',
            'type': 'Precedent',
            'title': raw_hist['recipient'],
            'content': raw_hist['full_text'],
            'date': raw_hist['date']
        })
        print(f"✅ Found {len(historical_df)} Precedents in {HISTORICAL_FILE}.")

    # 2. Combine Law (Verbatim) + Pitfalls + Historical Precedents
    stat_static_df = pd.DataFrame(verbatim_records + wisdom_statutory)
    stat_final_df = pd.concat([stat_static_df, historical_df], ignore_index=True)
    stat_final_df = stat_final_df.drop_duplicates(subset=['content'])
    
    stat_final_df.to_csv(STATUTORY_FILE, index=False)
    print(f"⚖️ STATUTORY TRUTH: {len(stat_final_df)} records saved to {STATUTORY_FILE}.")

    # --- PHASE B: BUILD OPTIMIZER INTELLIGENCE ---
    # 3. Create Intelligence DataFrame
    intel_final_df = pd.DataFrame(intelligence_records)
    intel_final_df.to_csv(INTELLIGENCE_FILE, index=False)
    print(f"🧠 INTELLIGENCE: {len(intel_final_df)} records saved to {INTELLIGENCE_FILE}.")

    print("🏁 SUCCESS: Knowledge Base Layering Complete.")

if __name__ == "__main__":
    hydrate_tiered()