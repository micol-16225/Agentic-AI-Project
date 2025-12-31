
import pandas as pd
import os

MASTER_FILE = "master_regulatory_library.csv"
HISTORICAL_FILE = "fda_letters.csv"


wisdom_sections = [
    {
        "source": "ICH_E9", 
        "type": "Requirement", 
        "title": "Multiplicity & Endpoints", 
        "content": '''- Requirement: All primary and secondary endpoints must be pre-specified in the protocol.
- Multiplicity: If multiple outcomes or interim analyses are used, the Type I Error (Alpha) must be controlled. Failure to adjust for multiple comparisons (e.g., Bonferroni, Hochberg) is a major reason for rejection.
- Subgroup Analysis: Results of subgroup analyses must be interpreted with caution. They are generally hypothesis-generating, not confirmatory.'''
    },
    {
        "source": "ICH_E9", 
        "type": "Requirement", 
        "title": "Missing Data & Bias", 
        "content": '''- ITT Principle: The "Intention-To-Treat" (ITT) population must be the primary analysis set. All randomized patients must be included.
- Missing Data: The protocol must pre-specify how missing data will be handled. Simple "Last Observation Carried Forward" (LOCF) is often considered biased. Sensitivity analyses must be performed to see if missing data changed the outcome.
- Blinding: Breaking the blind or having "unblinded" personnel involved in data cleaning is a critical violation (BIMO citation 312.60).'''
    },
    {
        "source": "ICH_E9_R1", 
        "type": "Framework", 
        "title": "Estimand Framework", 
        "content": '''- Concept: An 'Estimand' defines exactly "what" is being estimated. 
- Intercurrent Events: The protocol must specify how to handle events like: 
    a) Patient starts a "rescue" medication.
    b) Patient dies from a non-study cause.
    c) Patient discontinues drug due to side effects.
- Requirement: Relying only on "Treatment Policy" (ITT) without sensitivity analyses for these events is a major "Subtle Failure."'''
    },
    {
        "source": "21 CFR 312.62",
        "type": "Requirement",
        "title": "Data Integrity",
        "content": '''- Requirement: Investigators must maintain accurate, complete, and verifiable case histories.
- Statistical Validity: Data must be "fit for purpose." If Case Report Forms (CRFs) are inconsistent with the electronic database, the statistical validity of the trial is void. '''

    },
    {
        "source": "FDA_Guidance", 
        "type": "Guidance", 
        "title": "Adaptive Designs", 
        "content": '''- Pre-specification: All adaptive design features must be pre-specified in the protocol and statistical analysis plan (SAP).
- Type I Error Control: The overall Type I Error rate must be controlled, accounting for adaptations.
- Simulation: Extensive simulations must be provided to demonstrate the operating characteristics of the design under various scenarios.'''
    },
    {
        "source": "FDA_Guidance",  
        "type": "Guidance",
        "title": "Operational Bias & Data Blinding", 
        "content": '''
- Operational Bias: Even if a trial is "Double Blind," bias can leak if the team making "Protocol Deviation" decisions has seen the aggregate data trends.
- Data Cleaning: If data is cleaned or "queried" more aggressively for the placebo group than the treatment group, the P-value is invalid.
- Requirement: There must be a clear "Firewall" between the Data Monitoring Committee (DMC) and the personnel managing trial operations. '''

    },
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
    }
]


elite_100_sections = [
    {
        "source": "Elite_100_Index",
        "type": "Academic_Rigor",
        "title": "Multiplicity & P-Hacking",
        "content": '''DOMAIN: MULTIPLICITY & P-HACKING (Logic: Controlling False Positives)
- Key Citations: Benjamini & Hochberg (1995) JRSSB; Westfall & Young (1993) JASA; Adda et al. (PNAS 2020).
- Expert Rule: Flag "Spikes" in p-values near 0.05. If multiple secondary endpoints are claimed as "Successes" without FDR control, cite Benjamini & Hochberg.'''
    },
    {
        "source": "Elite_100_Index",
        "type": "Academic_Rigor",
        "title": "Missing Data & Estimands",
        "content": '''DOMAIN: MISSING DATA & ESTIMANDS (Logic: Handling Drop-outs)
- Key Citations: Rubin (1976) Biometrika; Hernán & Robins (2024) Causal Inference; ICH E9(R1).
- Expert Rule: Distinguish between MAR/MNAR. If drop-outs are treatment-related (e.g., side effects), "Missing at Random" assumptions fail. Demand sensitivity analyses per Rubin (1976).'''
    },
    {
        "source": "Elite_100_Index",
        "type": "Academic_Rigor",
        "title": "Adaptive Designs",
        "content": '''DOMAIN: ADAPTIVE DESIGNS (Logic: The "Promising Zone")
- Key Citations: Mehta & Pocock (2011) Stat. Med.; O'Brien & Fleming (1979) Biometrics.
- Expert Rule: Sample size re-estimation (SSR) is only valid without penalty if the interim result is in the "Promising Zone." Ad-hoc doubling of N is a "Statistical Penalty" event.'''
    },
    {
        "source": "Elite_100_Index",
        "type": "Academic_Rigor",
        "title": "External Validity",
        "content": '''DOMAIN: EXTERNAL VALIDITY (Logic: Generalizability)
- Key Citations: Rothwell (2005) Lancet; Pearl (2011) JASA.
- Expert Rule: Flag "Sanitized" populations. If exclusion criteria make the study group unrepresentative of real-world patients, cite Rothwell (2005).'''
    },
    {
        "source": "Elite_100_Index",
        "type": "Academic_Rigor",
        "title": "Causal Inference",
        "content": '''DOMAIN: CAUSAL INFERENCE (Logic: Fairness)
- Key Citations: Rosenbaum & Rubin (1983) Biometrika; Robins et al. (2000) Epidemiology.
- Expert Rule: Ensure "Fair Comparison." Check for Propensity Score usage or G-estimation if treatment switching occurred.'''
    }
]

final_wisdom_sections = [
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
    static_df = pd.DataFrame(wisdom_sections + elite_100_sections + final_wisdom_sections)
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