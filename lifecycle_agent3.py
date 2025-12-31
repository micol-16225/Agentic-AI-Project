import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
from groq import Groq

# This looks for the .env file and loads the key into your system memory
load_dotenv()

# This tells the Groq client to look for that system variable
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

ACADEMIC_MANDATE = """
TONE & STYLE: 
- Professional, precise, and scholarly. Avoid "AI-assistant" filler.
- Use industry-standard terminology (e.g., 'Intercurrent Events', 'MAR/MNAR assumptions', 'Family-wise Error Rate').
- Communicate with the authority of a Senior Methodologist defending a design before a Regulatory Board.

EVIDENTIARY STANDARDS:
- GROUNDING: Your 'Ground Truth' is the provided Knowledge Base (Papers & FDA Guidance). 
- CITATION: Every statistical estimation or inferential procedure must be parenthetically cited, e.g., (Akacha et al., 2017) or (ICH E9 R1).
- HIERARCHY: Prioritize peer-reviewed academic methodology over general LLM knowledge. 
- UNCERTAINTY: If the literature provides multiple valid paths (e.g., MMRM vs. Multiple Imputation), discuss the trade-offs like a scholar.
"""

class BiostatLifecycleAgent3:
    def __init__(self, api_key, library_path):
        self.client = Groq(api_key=api_key)
        self.model_id = "llama-3.3-70b-versatile"
        self.library_path = library_path
        self.knowledge_base = ""
    
    def _load_library(self, search_query=""):
        if not os.path.exists("master_regulatory_library.csv"):
            return "⚠️ Knowledge base missing."
            
        df = pd.read_csv("master_regulatory_library.csv")
        df.columns = [c.strip() for c in df.columns]

        # INTERNAL HELPER: Now with a 'Mandatory' fallback
        def get_track_context(subset_df, query, limit=5, mandatory=False):
            if subset_df.empty: return ""
            
            # If it's a Statutory track, we ALWAYS want the top rules regardless of search
            if mandatory:
                return "\n".join([f"- [{row['title']}]: {row['content']}" for _, row in subset_df.head(limit).iterrows()])

            texts = subset_df['content'].fillna("").astype(str).tolist()
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                boosted_query = f"{query} ICH FDA regulation statistics"
                
                matrix = vectorizer.fit_transform(texts + [boosted_query])
                sims = cosine_similarity(matrix[-1], matrix[:-1])
                
                # --- CONSOLE DEBUGGING START ---
                max_score = sims[0].max()
                print(f"\n🔍 [RAG DEBUG] Track: {subset_df['source_type'].iloc[0]}")
                print(f"   |-- Top Similarity Score: {max_score:.4f}")
                print(f"   |-- Library Size: {len(texts)} documents")
                
                if max_score < 0.1:
                    print(f"   |-- ⚠️ WARNING: Low similarity. Results may be irrelevant.")
                # --- CONSOLE DEBUGGING END ---

                actual_limit = min(len(texts), limit)
                indices = sims[0].argsort()[-actual_limit:][::-1]
            except:
                return "\n".join([f"- [{row['title']}]: {row['content']}" for _, row in subset_df.head(limit).iterrows()])

        # --- THE FIX: MANDATORY LOADING ---
        # We force the Statutory track to load so ICH E9 is NEVER missing
        stat_ctx = get_track_context(df[df['source_type'] == 'Statutory'], search_query, limit=10, mandatory=True)
        acad_ctx = get_track_context(df[df['source_type'] == 'Academic'], search_query, limit=5)
        prec_ctx = get_track_context(df[df['source_type'] == 'FDA_Letter'], search_query, limit=5)

        return f"### MANDATORY RULES ###\n{stat_ctx}\n\n### ACADEMIC/PRECEDENT ###\n{acad_ctx}\n{prec_ctx}"
    
    
    def _generate_response(self, prompt):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_id,
            )
            response = chat_completion.choices[0].message.content
            
            # CLEANING FOR MATH SYMBOLS: 
            # Convert common AI LaTeX wrappers to $ and $$ for Streamlit rendering
            clean_response = response.replace("\\(", "$").replace("\\)", "$")
            clean_response = clean_response.replace("\\[", "$$").replace("\\]", "$$")
            
            return clean_response
        except Exception as e:
            return f"🚨 Connection Error: {str(e)}"

    
    def generate_interview_questions(self, drug_name, indication):
        prompt = f"""
        ROLE: Lead Statistical Methodologist.
        CONTEXT: A biostatistician is designing a Phase 3 trial for {drug_name} targeting {indication}.
        
        TASK: Generate 6-8 SHARP, technical, and probing statistical questions the user MUST answer 
        before you can draft a protocol. 
        
        Focus on:
        - Primary Endpoint distribution and its justification.
        - Intercurrent events and the specific Estimand strategy (ICH E9 R1).
        - Multiplicity adjustment logic for secondary endpoints.
        - Sample size assumptions (Delta, SD, and dropout rate).
        - Handling of longitudinal correlation (MMRM structure).
        
        Format: Return a clean list of questions. No introductory text.
        """
        return self._generate_response(prompt)

    def draft_protocol_from_interview(self, drug_info, interview_data):
        prompt = f"""
        {ACADEMIC_MANDATE}
        ROLE: Principal Biostatistician & Senior Clinical Trial Protocol Architect.
        DRUG INFO: {drug_info}
        USER'S ANSWERS: {interview_data}
        
        TASK: Draft a comprehensive Statistical Analysis Plan (SAP) section for a Phase 3 Protocol.
        Structure the output with these precise sections:
        1. ESTIMAND DEFINITION: Explicitly define the five attributes (Population, Variable, Intercurrent Events, Population-level Summary, and Strategy) per ICH E9 (R1).
        2. SAMPLE SIZE & POWER: Include the mathematical formula in LaTeX, justifying the Delta and Variance assumptions.
        3. ANALYSIS SETS: Define ITT, Per-Protocol, and Safety populations.
        4. STATISTICAL METHODS: Describe the primary model (e.g., MMRM or Cox Proportional Hazards) and the handling of longitudinal correlations.
        5. MULTIPLICITY: Define the specific gatekeeping or fallback procedure for secondary endpoints.
        
        IMPORTANT: For any interview question where the user provided an empty or "I don't know" answer:
        1. APPLY industry "Gold Standard" statistical methods for {drug_info['ind']}.
        2. In the drafted text, add a small 'AI ADVISORY' note explaining why that specific method was chosen.
        
        Ensure the 'Statistical Analysis' section is rigorous and uses LaTeX.
        """
        
        return self._generate_response(prompt)

    def explain_logic(self, protocol_text, query, interview_answers):
        prompt = f"""
        {ACADEMIC_MANDATE}
        ROLE: Principal Biostatistician & Senior Clinical Trial Protocol Architect.
        
        CONTEXT:
        You are defending a protocol draft you just created. A colleague is questioning your statistical rigor.
        
        DRAFTED PROTOCOL: 
        {protocol_text}
        
        USER'S INTERVIEW ANSWERS (The Intent):
        {interview_answers}
        
        THE CHALLENGE:
        "{query}"
        
        INSTRUCTIONS:
        1. Address the challenge directly. (e.g., if asked about applying a Normal distribution to a 1-5 scale, explain the trade-offs between CLT-based normality and Ordinal models).
        2. Check for Hallucinations: If the protocol text contradicts the user's intent or FDA standards, admit it and suggest a correction.
        3. Use technical "Pharma-grade" language (e.g., mention ITT, Type I Error, or Sensitivity Analysis).
        """
        return self._generate_response(prompt)

    def audit_protocol(self, user_protocol, historical_lessons, user_directives=""):
        # DYNAMIC STEP: Update knowledge base using the protocol as the search query
        self.knowledge_base = self._load_library(search_query=user_protocol)

        prompt = f"""
        {ACADEMIC_MANDATE}
        ROLE: FDA Statistical Reviewer (Adversarial Audit).
        
        --- INTEGRATED KNOWLEDGE BASE ---
        {self.knowledge_base}
        
        --- HISTORICAL LESSONS ---
        {historical_lessons}
        
        --- USER DIRECTIVES ---
        {user_directives if user_directives else "Follow standard regulatory guardrails."}

        TASK: Conduct a 'Refusal to File' (RTF) assessment of the following SAP: 
        {user_protocol}

        YOUR OBJECTIVE: Identify any statistical or regulatory risks that would lead to an FDA Refusal to File (RTF).
            CATEGORIES TO ASSESS:   
            - MULTIPLICITY CONTROL
            - MISSING DATA HANDLING
            - ESTIMAND & INTERCURRENT EVENTS
            - NUISANCE PARAMETERS & VARIANCE ASSUMPTIONS
            - COVARIATE PRE-SPECIFICATION

        ADVERSARIAL FOCUS:
        - INTERNAL CONTRADICTIONS: Does the statistical model (Section 4) actually match the endpoint distribution described in Section 1?
        - ESTIMAND ALIGNMENT: Does the handling of 'Rescue Medication' in the Estimand section conflict with the 'Missing Data' assumptions?
        - REGULATORY DEBT: Identify where the protocol uses 'Legacy' methods (like LOCF) instead of modern 'Gold Standards' (like Multiple Imputation or MMRM) required by ICH E9 (R1).

        INSTRUCTIONS:
        1. If interview data exists, you MUST compare the drafted protocol with the interview answers. 
        Does the draft actually follow the user's directives specified in the interview? 
        You must check that ALL answers are correctly implemented in the protocol. If you find any contradictions, or omissions, flag them as CRITICAL.
        2. You MUST compare this draft to the specific FDA Letters in the HISTORICAL LESSONS.
        3. If you find a risk that matches a past failure, you MUST cite it like this: 
        "MATCH FOUND: [Date] | [Recipient Name] | [Quote]"
        Paste the exact, verbatim quote from the letter here. Do not summarize. Do not use ellipses. Write in them in bold.
        4. Follow any USER DIRECTIVES provided.
        5. DEEP SCRUTINY: Do not skim. Read every paragraph of the Statistical Analysis Plan (SAP) section.
        6. LOGICAL PROOF: Before assigning a Risk Level, you must find and cite the specific sentence in the protocol that addresses that risk. 
           - If the protocol says "we will use a gatekeeping strategy," you are REQUIRED to lower the Multiplicity risk to LOW. 
           - Failure to recognize an existing safeguard is a 'Hallucination of Risk' and is strictly forbidden.
        7. ADVERSARIAL CHALLENGE: If a fix is present (like Bonferroni-Holm), evaluate if it is *sufficient* for the trial's complexity. If it is sufficient, the risk is LOW.
        
        
        STRUCTURE YOUR AUDIT REPORT INTO FOUR SECTIONS:
        ---
        ### SECTION 1: FDA HISTORICAL PRECEDENTS (THE CASES) ###
        In this section, you MUST find specific matches in the FDA CRL/Warning Letters from the Knowledge Base.
        If the protocol uses a strategy that was rejected in the past:
        1. Name the Recipient and Date of the letter.
        2. Provide the VERBATIM quote from the FDA letter, in bold characters.
        3. Explain how the current protocol is repeating that specific mistake.
        
        ---
        ### SECTION 2: REGULATORY & ACADEMIC ALIGNMENT (THE RULES) ###
        In this section, you MUST identify conflicts between the protocol and:
        1. The 'Wisdom Sections' (ICH E9, 21 CFR, HITL Gates).
        2. The Academic Literature (e.g., Akacha, Mallinckrodt).
        
        If a method in the protocol is outdated or violates a rule, cite the Rule/Paper and explain the mathematical risk.

        ---
        ### SECTION 3: FINAL RISK ASSESSMENT TABLE ###

        OUTPUT:
        - **Risk Table**: Category | Violation Type | Source | Risk Level (Critical/High/Medium/Low) |
        - **FDA Historical Precedents**: A list of matches between this draft and the FDA Letters provided.
        
        ---
        ### SECTION 4: INTERNAL MONOLOGUE (Mandatory):
        For each category, perform this check:
        - "Does the protocol mention [Category]?" -> Yes/No
        - "What specific method is used?" -> [Exact Quote]
        - "Is there a specific FDA Letter in my Knowledge Base that contradicts or critiques this specific approach?" -> [Exact Letter Quote and Reference]
        - "Does the Academic Literature (e.g., Akacha, Mallinckrodt) suggest a more robust alternative?" -> [Cite all relevant academic papers from the Knowledge Base]

        """
        return self._generate_response(prompt)


    def explain_theory(self, concept_to_explain):
        prompt = f"""
        {ACADEMIC_MANDATE}
        ROLE: Senior Biostatistics Professor & Regulatory Mentor.
        TASK: Explain the following statistical/regulatory concept used in clinical trials:
        
        CONCEPT: {concept_to_explain}
        
        STRUCTURE YOUR RESPONSE AS:
        1. THE "IN PLAIN ENGLISH" DEFINITION: Explain it to a non-expert.
        2. THE MATHEMATICAL LOGIC: Why does this matter for the data? (Use LaTeX).
        3. THE REGULATORY "SO WHAT?": Why does the FDA care about this?
        4. REAL-WORLD EXAMPLE: A scenario where this saved or sank a trial.
        """
        return self._generate_response(prompt)
    
    def optimize_protocol(self, original_protocol, audit_report, user_directives="None", max_iterations=2):
        # 1. SMART CONTEXT LOADING (Instead of raw file read)
        # We use the search query to get the top 15 most relevant rules/letters
        targeted_wisdom = self._load_library(search_query=original_protocol)
        
        current_protocol = original_protocol
        current_audit = audit_report
        
        for iteration in range(max_iterations):
            # 2. THE REFINED PROMPT
            optimization_prompt = f"""
            {ACADEMIC_MANDATE}
            ROLE: Principal Biostatistician & Regulatory Strategist.
            
            STATISTICAL & REGULATORY GROUND TRUTH:
            {targeted_wisdom}

            INPUTS:
            - Draft: {current_protocol}
            - Auditor's Critique: {current_audit}
            - User Directives: {user_directives}

            TASK: Rewrite into a 'Submission-Ready' version.
            1. Resolve EVERY risk flagged by the Auditor. Use SURGICAL precision (numbers, not vague terms).
            2. Incorporate Senior Reviewer Directives as priority.
            3. Use 'Gold Standard' methods from Academic References and CITE THEM.
            4. Formulas in $LaTeX$: $N = \\frac{{(Z_\\alpha + Z_\\beta)^2 \\sigma^2}}{{\\delta^2}}$.

            MANDATORY IMPROVEMENT STANDARDS:
            1. TRACEABILITY: Cite the Wisdom or Precedents used to justify the fix.
            2. MATHEMATICAL RIGOR: Use $LaTeX$ for all statistical models (MMRM, Cox, etc.).
            3. HITL ALIGNMENT: Ensure the fix addresses G-01 through G-04 gates.

            WARNING: If you provide template text without trial-specific parameters, the Auditor will reject you.
            """
            
            candidate_version = self._generate_response(optimization_prompt)

            # 3. INTERNAL SELF-CHECK (The Gatekeeper)
            check_prompt = f"""
            ROLE: Aggressive FDA Reviewer.
            TASK: Compare the FIXES in this new protocol against the PROBLEMS in the Audit.
            NEW PROTOCOL: {candidate_version}
            AUDIT REPORT: {current_audit}
            RESULT: Return 'PASS' if perfect, or list remaining failures specifically.
            """
            check_result = self._generate_response(check_prompt)

            if "PASS" in check_result.upper():
                return candidate_version
            
            # If it didn't pass, update context for the next loop iteration
            current_protocol = candidate_version
            current_audit = check_result
            
        return current_protocol