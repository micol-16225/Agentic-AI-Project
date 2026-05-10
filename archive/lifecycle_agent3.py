import os
import pandas as pd
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

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
    def __init__(self, api_key, library_path, model_id="llama-3.1-8b-instant"):
        self.client = Groq(api_key=api_key)
        self.model_id = model_id
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
        # Statutory rules (ICH E9, 21 CFR, etc.)
        stat_ctx = get_track_context(df[df['type'] == 'Statutory'], search_query, limit=10, mandatory=True)
        
        # Academic/Research standards (Your "Academic_Rigor" tag)
        acad_ctx = get_track_context(df[df['type'] == 'Academic_Rigor'], search_query, limit=3)
        
        # FDA Precedents (Mapped to your TYPE B/C correspondence tags)
        prec_ctx = get_track_context(df[df['type'].str.contains('TYPE', na=False)], search_query, limit=1)

        return f"### MANDATORY RULES ###\n{stat_ctx}\n\n### ACADEMIC/PRECEDENT ###\n{acad_ctx}\n{prec_ctx}"    
    
    def track_usage(self, prompt, response):
        # Rough estimation: 1 word ≈ 1.33 tokens
        in_tokens = len(prompt.split()) * 1.33
        out_tokens = len(response.split()) * 1.33
        
        # ADD flush=True here to bypass the buffer
        print(f"📈 [TOKEN TRACKER] In: {int(in_tokens)} | Out: {int(out_tokens)} | Total: {int(in_tokens + out_tokens)}", flush=True)
        
        return in_tokens + out_tokens

    def _generate_response(self, prompt, max_retries=5):
        for i in range(max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_id,
                )
                response = chat_completion.choices[0].message.content
                
                # Tracking
                self.track_usage(prompt, response)
                
                # Cleaning for LaTeX
                clean_response = response.replace("\\(", "$").replace("\\)", "$")
                clean_response = clean_response.replace("\\[", "$$").replace("\\]", "$$")
                return clean_response if clean_response else "🚨 Error: Empty response from model"
            except Exception as e:
                if "429" in str(e):
                    wait = 2**i
                    print(f"⚠️ Rate limited. Retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"🚨 API Error: {str(e)}", flush=True)
                    return f"🚨 Error: {str(e)}"
        return "🚨 Error: Max retries exceeded"

    
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

    
    def _run_single_precedent_check(self, protocol, precedent_row):
        
        """
        SOURCE 3: THE PARALLEL WORKER (FDA LETTERS ONLY)
        Each worker focuses on ONE historical failure to ensure 100% verbatim accuracy.
        """
        prompt = f"""
        ROLE: FDA Statistical Auditor.
        SOURCE_MATERIAL: "{precedent_row['content']}"
        SOURCE_ID: {precedent_row['title']}

        TASK: 
        1. EXTRACT: Identify the Recipient (Firm) and the Date of the letter from the source above. 
        2. AUDIT: Does the following protocol repeat a failure mentioned in this FDA letter?

        PROTOCOL: {protocol}

        MANDATORY VERBATIM RULE:
        - If a match is found, you MUST copy the quote character-for-character.
        - YOU MUST EXTRACT THE QUOTE VERBATIM. 
        - DO NOT PARAPHRASE. DO NOT SUMMARIZE. DO NOT INSERT ELLIPSES (...).
        - If you change a single word, the audit fails.
        - If no match, return "NO VALIDATED MATCH."

        CRITICAL: You are a character-copying machine. Do NOT use ellipses (...). Do NOT summarize. Start the quote from the very first letter of the sentence and end at the period. If you add even one dot that isn't in the source, the audit is a failure.

        ### OUTPUT FORMAT (MANDATORY):
        MATCH FOUND: [Recipient Name] | [Date] | [EXACT VERBATIM QUOTE]
        IF NO MATCH: Return "NO VALIDATED MATCH."
        """
        return self._generate_response(prompt)

    def audit_protocol(self, user_protocol, historical_lessons="", user_directives=""):
        """
        THE ORCHESTRATOR: Synthesizes Law, Academic Papers, and Parallel Letter Findings.
        """
        # 1. LOAD LIBRARY & SEGREGATE SOURCES
        df = pd.read_csv(self.library_path)
        df.columns = [c.strip() for c in df.columns]

        # SOURCE 1: LAW (Statutory)
        law_df = df[df['type'] == 'Statutory']
        law_context = "\n".join([f"LAW ID: {r['title']}\n{r['content']}" for _, r in law_df.iterrows()])

        # SOURCE 2: ACADEMIC PAPERS (Academic_Rigor)
        acad_df = df[df['type'] == 'Academic_Rigor']
        acad_context = "\n".join([f"PAPER ID: {r['title']}\n{r['content']}" for _, r in acad_df.iterrows()])

        # SOURCE 3: FDA LETTERS (Mapping for Parallel Workers)
        letter_df = df[df['type'].str.contains('TYPE', na=False)].copy()
        vectorizer = TfidfVectorizer(stop_words='english')
        matrix = vectorizer.fit_transform(letter_df['content'].fillna("").tolist() + [user_protocol])
        sims = cosine_similarity(matrix[-1], matrix[:-1])
        top_indices = sims[0].argsort()[-3:][::-1]
        candidates = [letter_df.iloc[idx] for idx in top_indices]

        # 2. EXECUTE PARALLEL WORKERS (FDA Letters Only)
        print(f"🚀 Dispatching {len(candidates)} Parallel Letter Auditors...")
        # --- 4. PARALLEL EXECUTION (Safe Handling) ---
        # Inside audit_protocol:
        with ThreadPoolExecutor(max_workers=1) as executor: # Worker=1 is the only way to stay under 6k TPM
            futures = [executor.submit(self._run_single_precedent_check, user_protocol, c) for c in candidates]
            
            results = []
            for f in as_completed(futures):
                try:
                    res = f.result()
                    if res and isinstance(res, str) and "MATCH FOUND" in res:
                        results.append(res)
                except Exception as exc:
                    print(f"🚨 Worker failed for a candidate: {exc}", flush=True)
                
                # This is the 'Secret Sauce' for Task #3 (200 Tests)
                print("⏳ TPM cooldown... (12 seconds)", flush=True)
                time.sleep(12)

        letter_findings = "\n\n".join(results)

        # 3. FINAL SYNTHESIS (Merging all 3 Sources)
        print("⚖️ Synthesizing Law, Academia, and Precedent Findings...")
        final_prompt = f"""
        {ACADEMIC_MANDATE}
        ROLE: FDA Statistical Reviewer (Adversarial Audit).

        --- SOURCE 1: THE LAW (STATUTORY) ---
        {law_context}
        --- SOURCE 2: ACADEMIC PAPERS ---
        {acad_context}
        --- SOURCE 3: PRECEDENT FINDINGS (VERIFIED) ---
        {letter_findings}

        PROTOCOL: {user_protocol}

        ### MANDATORY VALIDATION RULES ###
        - SECTION 1 & 2 MUST ONLY use exact strings from the Sources above. 
        - DO NOT include your own commentary inside the bold quotes.
        - If a quote is not in Source 1, 2, or 3, DO NOT output it.

        REPORT STRUCTURE:

        ### SECTION 1: FDA HISTORICAL PRECEDENTS ###
        Format: Recipient | Date | **VERBATIM_QUOTE_FROM_SOURCE_3**
        (Repeat only what the Parallel Workers found).

        ### SECTION 2: REGULATORY & ACADEMIC ALIGNMENT ###
        Format: [ID] | **VERBATIM_QUOTE_FROM_SOURCE_1_OR_2**
        (Example: [ICH E9 R1] | **"The use of LOCF is generally not acceptable..."**)

        ### SECTION 3: FINAL RISK ASSESSMENT TABLE ###
        Category | Violation | Source | Risk Level |

        ### SECTION 4: INTERNAL MONOLOGUE ###
        (This is the only section where you are allowed to use your own words to explain the logic.)
        """
        report = self._generate_response(final_prompt)
        return report if report else "🚨 Error: Synthesis Failed"

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
    