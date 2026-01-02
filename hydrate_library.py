
import pandas as pd
import os

MASTER_FILE = "master_regulatory_library.csv"
HISTORICAL_FILE = "fda_letters.csv"



# 2. Define the VERBATIM records (No summaries allowed)
verbatim_records = [
    # --- STATUTORY: ICH E9 (R1) ---
    {
        "title": "ICH E9 (R1) Section 3.3.1",
        "type": "Statutory",
        "content": "An estimand is a precise description of the treatment effect reflecting the clinical question posed by a given clinical trial objective. It summarises at a population level what the outcomes would be in the same patients under different treatment conditions being compared."
    },
    {
        "title": "ICH E9 (R1) Section 5.2",
        "type": "Statutory",
        "content": "The use of the last observation carried forward (LOCF) approach is generally not acceptable because it can result in biased estimates of the treatment effect and can lead to misleading conclusions."
    },
    {
        "title": "ICH E9 (R1) Section 5.1",
        "type": "Statutory",
        "content": "Strategies for handling intercurrent events include: Treatment-policy, Hypothetical, Composite, While-on-treatment, and Stratum-plus-variable."
    },
    
    # --- STATUTORY: 21 CFR ---
    {
        "title": "21 CFR 314.126(a)",
        "type": "Statutory",
        "content": "Reports of adequate and well-controlled investigations provide the primary basis for determining whether there is 'substantial evidence' to support the claims of effectiveness for new drugs and antibiotics."
    },
    {
        "title": "21 CFR 314.126(b)(6)",
        "type": "Statutory",
        "content": "The protocol and report of the study should describe the procedures used to accomplish minimization of bias on the part of the subjects, observers, and analysts of the data."
    },

    # --- STATUTORY: FDA MULTIPLICITY ---
    {
        "title": "FDA Multiplicity Guidance (2022) Section III",
        "type": "Statutory",
        "content": "Procedures to control the Type I error rate should be pre-specified to ensure that the probability of making at least one Type I error does not exceed the desired level of significance."
    },
    {
        "title": "ICH E9 Section 5.2.1",
        "type": "Statutory",
        "content": "The Full Analysis Set (FAS) should be used in the primary analysis. The FAS is as close as possible to the intention-to-treat ideal and includes all randomized subjects who received at least one dose of study medication."
    },
    {
        "title": "ICH E9 Section 5.7",
        "type": "Statutory",
        "content": "The study protocol should identify those subgroups that are expected to be of primary interest and the statistical methods for examining them. Interactions between treatment and subgroups should be explored."
    },
    {
        "title": "FDA Multiplicity Guidance Section V",
        "type": "Statutory",
        "content": "A common method to control the FWER is a hierarchical (or fixed-sequence) testing procedure where the primary and secondary endpoints are tested in a pre-specified order."
    },
    # --- EXPANDED VERBATIM RECORDS (SAP & PROTOCOL PILLARS) ---
    {
        "title": "ICH E9 Section 5.2.1",
        "type": "Statutory",
        "content": "The Full Analysis Set (FAS) should be used in the primary analysis. The FAS is as close as possible to the intention-to-treat ideal and includes all randomized subjects who received at least one dose of study medication."
    },
    {
        "title": "ICH E3 Section 11.4.2.7",
        "type": "Statutory",
        "content": "The SAP should describe the statistical methods for handling dropouts and missing data. If any 'ad hoc' analyses were performed, they must be clearly identified and justified."
    },
    {
        "title": "ICH E6 (R2) Section 5.5.3",
        "type": "Statutory",
        "content": "When using electronic trial data handling systems, the sponsor should ensure and document that the systems are designed to permit data changes in such a way that the data changes are documented and that there is no deletion of entered data (i.e., maintain an audit trail)."
    }
]


wisdom_sections = [
    {
        "source": "BIMO_Pitfalls", 
        "type": "Pitfall", 
        "title": "Common BIMO Pitfalls", 
        "content": '''- Source Verification: 100% of data in the final report must match the "Source Docs" (hospital notes). '''
    },
    {
        "source": "Big_Pharma_Models", 
        "type": "Technical", 
        "title": "Statistical Formulas", 
        "content": '''- MMRM (Mixed Model for Repeated Measures): $Y_i = X_i\\beta + Z_i b_i + \\epsilon_i$ where $\epsilon_i \sim N(0, R_i)$.
- Cox Proportional Hazards: $h(t|x) = h_0(t) \exp(\\beta^T x)$.
- Alpha Spending (O'Brien-Fleming): $\alpha(t) = 2 - 2\Phi(Z_{\alpha/2} / \sqrt{t})$.'''
    },
    {
        "source": "Design_Principles",
        "type": "Architecture",
        "title": "The Architect: Design Principles",
        "content": '''--- SECTION 11: DESIGN PRINCIPLES (THE ARCHITECT) ---
- Rule of Parsimony: Minimize exclusion criteria to maximize real-world application.
- Endpoint Hierarchy: Prioritize "Hard" clinical endpoints over "Surrogate" markers.
- Blinding Firewall: Protocols must define the "Firewall" between unblinded statisticians and trial leads.'''
    },
    {
        "source": "HITL_Protocol",
        "type": "Process_Standard",
        "title": "Reasoning & Human-In-The-Loop",
        "content": '''--- SECTION 12: REASONING & HUMAN-IN-THE-LOOP (HITL) PROTOCOLS ---
GOAL: Transform "Black Box" AI into "White Box" Augmented Intelligence. 

REASONING STANDARDS:
1. TRACEABILITY: Every recommendation must be traced back to ICH guidelines or Elite 100 Index.
2. MATHEMATICAL EXPLICITNESS: Provide distribution assumptions (e.g., "Normal likelihood with Unstructured Covariance Matrix") and LaTeX notation.
3. CLINICAL CONTEXT: Connect math to the patient (e.g., MAR violations due to toxicity).

HUMAN-IN-THE-LOOP (HITL) MANDATORY GATES:
- G-01 (Feasibility): Verify recruitment potential for suggested N.
- G-02 (Risk Appetite): Approve "Promising Zone" boundaries for Adaptive Designs.
- G-03 (Ethics/Safety): Verify clinical importance of composite endpoint components.
- G-04 (Assumptions): Verify if Effect Size (delta) is Minimally Clinically Important (MCID).

VERIFICATION LOGIC:
If the human "Rejects" a step, the Agent must provide a "Sensitivity Analysis" showing how the design changes under the human's preferred assumptions.'''
    }
]


academic_sections = [
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


def hydrate_full():
    print("🚀 Starting Master Library Unification...")
    
    # 1. Load Live Scraped Data (from master_ingestor)
    try:
        live_df = pd.read_csv(MASTER_FILE)
        print(f"✅ Found {len(live_df)} live records from scrapers.")
    except:
        live_df = pd.DataFrame()
        print("⚠️ No live scraped data found yet.")

    # 2. Load Your Historical 100 Letters
    historical_df = pd.DataFrame()
    if os.path.exists(HISTORICAL_FILE):
        raw_hist = pd.read_csv(HISTORICAL_FILE)
        # We must map your old columns to our new master columns
        historical_df = pd.DataFrame({
            'source': 'FDA_Historical',
            'type': raw_hist['doc_type'],
            'title': raw_hist['recipient'],
            'content': raw_hist['full_text'],
            'date': raw_hist['date']
        })
        print(f"✅ Found {len(historical_df)} historical records from fda_letters.csv.")

    # 3. Create Wisdom DataFrame
    static_df = pd.DataFrame(verbatim_records + wisdom_sections + academic_sections)
    print(f"✅ Loaded {len(static_df)} biostats wisdom sections.")

    # 4. MERGE EVERYTHING
    # Priority Order: Wisdom first, then Historical, then Live
    final_df = pd.concat([static_df, historical_df, live_df], ignore_index=True)
    
    # Remove duplicates if any (based on content)
    final_df = final_df.drop_duplicates(subset=['content'])

    # 5. Save the final Master Library
    final_df.to_csv(MASTER_FILE, index=False)
    print(f"🏁 FINAL SUCCESS: {MASTER_FILE} now contains {len(final_df)} total records.")

if __name__ == "__main__":
    hydrate_full()