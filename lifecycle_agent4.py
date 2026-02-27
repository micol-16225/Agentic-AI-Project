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

from sentence_transformers import SentenceTransformer, util
import torch

import hashlib
import json

# This looks for the .env file and loads the key into your system memory
load_dotenv()

# This tells the Groq client to look for that system variable
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

ACADEMIC_MANDATE = """
TONE & STYLE: 
- Professional, precise, and scholarly. Avoid "AI-assistant" filler and pleasantries.
- Use clinical-trial industry-standard terminology. 
- Communicate with the authority of a Senior Methodologist defending a design before a Regulatory Board.

EVIDENTIARY STANDARDS:
- GROUNDING: Your 'Ground Truth' is the provided Knowledge Base (Papers & FDA Guidance). 
- CITATION: Every statistical estimation or inferential procedure must be parenthetically cited, e.g., (Akacha et al., 2017) or (ICH E9 R1).
- HIERARCHY: Prioritize peer-reviewed academic methodology over general LLM knowledge. 
- UNCERTAINTY: If the literature provides multiple valid paths (e.g., MMRM vs. Multiple Imputation), discuss the trade-offs like a scholar.
"""

class BiostatLifecycleAgent4:
    # Use the S-BioBERT version for semantic similarity
    _model = None 
    _embedding_cache = {} #short term memory for embeddings to avoid redundant computation
    _disk_cache_path = "brain_cache.json" # long-term memory for Q&A pairs to avoid redundant API calls

    def __init__(self, api_key, statutory_path="statutory_truth.csv", intel_path="optimizer_intelligence.csv", model_id="llama-3.1-8b-instant"):
        self.client = Groq(api_key=api_key)
        self.model_id = model_id
        # UPDATED: Dual-path initialization for Task #2 & #4
        self.statutory_path = statutory_path
        self.intel_path = intel_path
        # Lazy load the model so it only occupies RAM when the agent is used
        if BiostatLifecycleAgent4._model is None:
            BiostatLifecycleAgent4._model = SentenceTransformer('pritamdeka/S-BioBert-snli-multinli-stsb')
        self.model = BiostatLifecycleAgent4._model

        # NEW: Pre-embed the libraries for Token/Compute efficiency
        self._precompute_library_embeddings(statutory_path, intel_path)

    def _precompute_library_embeddings(self, stat_path, intel_path):
        print("📊 Pre-computing library embeddings...")
        # Pre-load Dataframes
        self.stat_df = pd.read_csv(stat_path)
        self.intel_df = pd.read_csv(intel_path)
        
        # Pre-embed Academic Rigor (The Intel Path)
        academic_rows = self.intel_df[self.intel_df['type'] == 'Academic_Rigor']['content'].fillna("").tolist()
        self.academic_embeddings = self.model.encode(academic_rows, convert_to_tensor=True)
        self.academic_texts = academic_rows

        # Pre-embed FDA Precedents (The Stat Path)
        precedent_rows = self.stat_df[self.stat_df['type'].str.contains('Precedent|TYPE', na=False)]['content'].fillna("").tolist()
        self.precedent_embeddings = self.model.encode(precedent_rows, convert_to_tensor=True)
        self.precedent_texts = precedent_rows
    
    def _check_memory(self, prompt):
        fingerprint = hashlib.md5(prompt.encode()).hexdigest()
        if os.path.exists(self._disk_cache_path):
            with open(self._disk_cache_path, "r") as f:
                memory = json.load(f)
                return memory.get(fingerprint)
        return None

    def _save_to_memory(self, prompt, response):
        fingerprint = hashlib.md5(prompt.encode()).hexdigest()
        memory = {}
        if os.path.exists(self._disk_cache_path):
            try:
                with open(self._disk_cache_path, "r") as f:
                    memory = json.load(f)
            except json.JSONDecodeError:
                memory = {}
        memory[fingerprint] = response
        with open(self._disk_cache_path, "w") as f:
            json.dump(memory, f)
    

    def _load_library(self, search_query="", mode="audit"):
        """
        UPDATED: Swaps between Statutory Truth and Optimizer Intelligence.
        mode="audit" -> Strictly Statutory (The Law)
        mode="draft/optimize" -> Statutory + Academic + Precedents (The Brain)
        """
        
        # 1. Define File Paths
        STAT_PATH = "statutory_truth.csv"
        INTEL_PATH = "optimizer_intelligence.csv"

        EMBEDDING_CACHE = {}

        def get_track_context(file_path, filter_type, query, limit=5, mandatory=False):
            if not os.path.exists(file_path):
                return ""
            
            df = pd.read_csv(file_path)
            df.columns = [c.strip() for c in df.columns]
            
            # Filter by type (e.g., 'Statutory', 'Academic_Rigor', or 'Precedent')
            # For Precedents, we use string contains to catch 'TYPE B/C'
            if "TYPE" in filter_type or "Precedent" in filter_type:
                subset_df = df[df['type'].str.strip().str.contains('TYPE|Precedent', na=False)]
            else:
                subset_df = df[df['type'] == filter_type]

            if subset_df.empty: return ""
            
            # --- MANDATORY FALLBACK (Your Original Logic) ---
            if mandatory:
                return "\n".join([f"- [{row['title']}]: {row['content']}" for _, row in subset_df.head(limit).iterrows()])

            texts = subset_df['content'].fillna("").astype(str).tolist()

            # USE THE CLASS CACHE
            cache_key = f"{file_path}_{filter_type}"
            if cache_key not in BiostatLifecycleAgent4._embedding_cache:
                print(f"🧬 Encoding {cache_key}...")
                texts = subset_df['content'].fillna("").tolist()
                BiostatLifecycleAgent4._embedding_cache[cache_key] = self.model.encode(texts, convert_to_tensor=True)
            
            doc_embeddings = BiostatLifecycleAgent4._embedding_cache[cache_key]

            # --- NEW BIOBERT RAG SEARCH ---
            try:
                # 1. Skip re-encoding the documents! Use doc_embeddings instead.

                # 2. Turn your search query into a "Bio-vector"
                query_embedding = self.model.encode(query, convert_to_tensor=True)
                
                # 3. Find which papers are "closest" in meaning to your question
                # This is much smarter than just matching words!
                cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
                
                # 4. Pick the best ones
                top_results = torch.topk(cosine_scores, k=min(len(texts), limit))
                indices = top_results.indices.tolist()
                scores = top_results.values.tolist()

                print(f"🔍 [BIOBERT DEBUG] Top Score: {scores[0]:.4f}")

                return "\n".join([f"- [{subset_df.iloc[i]['title']}]: {subset_df.iloc[i]['content']}" for i in indices])

            except Exception as e:
                print(f"❌ Error in BioBERT: {e}")
                # Fallback to just showing the first few rows if the brain breaks
                return "\n".join([f"- [{row['title']}]: {row['content']}" for _, row in subset_df.head(limit).iterrows()])
        
        # --- ORCHESTRATION LOGIC ---
        if mode == "audit":
            # Auditor ONLY sees the Statutory Truth
            stat_ctx = get_track_context(STAT_PATH, 'Statutory', search_query, limit=10, mandatory=True)
            prec_ctx = get_track_context(STAT_PATH, 'Precedent', search_query, limit=3)
            return f"### MANDATORY RULES (STATUTORY) ###\n{stat_ctx}\n\n### FDA PRECEDENTS ###\n{prec_ctx}"
        
        else:
            # Optimizer/Architect sees EVERYTHING
            stat_ctx = get_track_context(STAT_PATH, 'Statutory', search_query, limit=5, mandatory=True)
            acad_ctx = get_track_context(INTEL_PATH, 'Academic_Rigor', search_query, limit=5)
            # Pull Technical/Architecture from Intelligence file
            arch_ctx = get_track_context(INTEL_PATH, 'Architecture', search_query, limit=2)
            
            return f"### STATUTORY CONSTRAINTS ###\n{stat_ctx}\n\n### ACADEMIC RIGOR ###\n{acad_ctx}\n\n### DESIGN PRINCIPLES ###\n{arch_ctx}"


    def track_usage(self, prompt, response):
        # Rough estimation: 1 word ≈ 1.33 tokens
        in_tokens = len(prompt.split()) * 1.33
        out_tokens = len(response.split()) * 1.33
        
        # ADD flush=True here to bypass the buffer
        print(f"📈 [TOKEN TRACKER] In: {int(in_tokens)} | Out: {int(out_tokens)} | Total: {int(in_tokens + out_tokens)}", flush=True)
        
        return in_tokens + out_tokens

    
    def _generate_response(self, prompt, max_retries=5):
        # 1. Check if we already know the answer
        cached_answer = self._check_memory(prompt)
        if cached_answer:
            print("🧠 Using cached memory (0 tokens used!)", flush=True)
            return cached_answer
        
        for i in range(max_retries):
            try:
                # 2. Use the 'system' role for the Academic Mandate
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": ACADEMIC_MANDATE},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_id,
                    temperature=0.0
                )
                response = chat_completion.choices[0].message.content

                # 3. Save the answer so we don't pay for it again
                self._save_to_memory(prompt, response)
                
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
        Refactored to handle the new Statutory Truth schema and 'Precedent' types.
        """
        
        # 1. ENSURE DATA INTEGRITY
        # We extract title, content, and the source/date metadata from our new tiered row
        source_id = precedent_row.get('title', 'Unknown Source')
        source_text = precedent_row.get('content', '')
        source_name = precedent_row.get('source', 'FDA_Historical')

        prompt = f"""
        ROLE: FDA Statistical Auditor.
        SOURCE_MATERIAL: "{source_text}"
        SOURCE_ID: {source_id}
        SOURCE_METADATA: From {source_name}

        TASK: 
        1. EXTRACT: Identify the Recipient (Firm) and the Date of the letter from the source material and metadata provided. 
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
        MATCH FOUND: [Recipient] | [Date] | [EXACT VERBATIM QUOTE]
        IF NO MATCH: Return "NO VALIDATED MATCH."
        """
        return self._generate_response(prompt)

    def audit_protocol(self, user_protocol, historical_lessons="", user_directives=""):
        """
        THE ORCHESTRATOR: Now fully upgraded to BioBERT Semantic Search.
        """
        # 1. LOAD TIERED SOURCES
        stat_df = pd.read_csv("statutory_truth.csv")
        stat_df.columns = [c.strip() for c in stat_df.columns]

        intel_df = pd.read_csv("optimizer_intelligence.csv")
        intel_df.columns = [c.strip() for c in intel_df.columns]

        # --- SOURCE 1: THE LAW (KEEP INTACT) ---
        # We send ALL of these because you want them intact
        law_df = stat_df[stat_df['type'] == 'Statutory']
        law_context = "\n".join([f"LAW: {r['content']}" for _, r in law_df.iterrows()])

        # --- SOURCE 2: ACADEMIC PAPERS (COMPRESS/FILTER) ---
        # Instead of all papers, we use your BioBERT to find the 3 best ones
        paper_texts = intel_df[intel_df['type'] == 'Academic_Rigor']['content'].tolist()
        paper_embeddings = self.model.encode(paper_texts, convert_to_tensor=True)
        proto_embed = self.model.encode(user_protocol, convert_to_tensor=True)
        
        # Pick the top 5 papers only
        scores = util.cos_sim(proto_embed, paper_embeddings)[0]
        top_papers = torch.topk(scores, k=min(len(paper_texts), 5)).indices.tolist()
        acad_context = "\n".join([paper_texts[idx] for idx in top_papers])

        # --- SOURCE 3: FDA LETTERS (Semantic Search Upgrade) ---
        letter_df = stat_df[stat_df['type'].str.contains('Precedent|TYPE', na=False)].copy()

        candidates = []
        if not letter_df.empty:
            try:
                # Vectorize the protocol and the historical letters
                texts = letter_df['content'].fillna("").tolist()
                
                # Use the class-level BioBERT model
                doc_embeddings = self.model.encode(texts, convert_to_tensor=True)
                protocol_embedding = self.model.encode(user_protocol, convert_to_tensor=True)
                
                # Compute similarity
                cos_sims = util.cos_sim(protocol_embedding, doc_embeddings)[0]
                
                # Get top 3 most relevant precedents
                top_results = torch.topk(cos_sims, k=min(len(texts), 3))
                indices = top_results.indices.tolist()
                
                candidates = [letter_df.iloc[idx] for idx in indices]
                print(f"🎯 BioBERT found {len(candidates)} relevant FDA precedents.")
            except Exception as e:
                print(f"🚨 BioBERT Search Failed: {e}. Falling back to top 3 rows.")
                candidates = [letter_df.iloc[i] for i in range(min(len(letter_df), 3))]

        # 2. EXECUTE PARALLEL WORKERS (FDA Letters Only)
        print(f"🚀 Dispatching {len(candidates)} Parallel Letter Auditors...")
        results = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Pass the whole row (c) so the worker can extract Metadata (Source/Date)
            futures = [executor.submit(self._run_single_precedent_check, user_protocol, c) for c in candidates]
            
            for f in as_completed(futures):
                try:
                    res = f.result()
                    if res and "MATCH FOUND" in res:
                        results.append(res)
                except Exception as exc:
                    print(f"🚨 Worker failed: {exc}", flush=True)
                
                # TPM Cooldown for Task #3
                time.sleep(12)

        letter_findings = "\n\n".join(results)

        # 3. FINAL SYNTHESIS (Merging all 3 Sources)
        print("⚖️ Synthesizing Law, Academia, and Precedent Findings...")
        final_prompt = f"""
        {ACADEMIC_MANDATE}
        ROLE: FDA Statistical Reviewer (Audit).
        GOAL: Identify specific violations in the current protocol that risk FDA rejection.
        Conduct a balanced regulatory review. Determine if the protocol is 'Submission-Ready' or requires specific corrections.

        --- SOURCE 1: THE LAW (STATUTORY) ---
        {law_context}

        --- SOURCE 2: VERIFIED FDA PRECEDENTS ---
        {letter_findings}

        --- SOURCE 3: ACADEMIC RIGOR (INTELLIGENCE) ---
        {acad_context}

        PROTOCOL: {user_protocol}

        ### MANDATORY VALIDATION RULES ###
        1. VERBATIM ONLY: All quotes in Section 1 and 2 must be 100% character-accurate.
        2. RELEVANCE: For every citation, you MUST explain the "Violation Logic": Why is this specific rule relevant to this protocol, and how exactly does the draft violate it?
        3. BALANCED JUDGMENT: If the protocol adheres to Source 1 (The Law) and Source 3 (Academic Rigor), you MUST explicitly state that the section is 'Compliant.' Do not invent risks where the standards are met.

        CRITICAL: 
        If you are unsure of the EXACT wording of a quote, omit it entirely. Do not paraphrase. If you output a quote that is not present in Source 1 or in Source 2, the audit will be considered a legal liability. I prefer a report with 3 100% accurate quotes over a report with 20 paraphrased quotes.

        REPORT STRUCTURE:

        ### SECTION 1: REGULATORY ALIGNMENT ###
        Format: 
        [ID] | **VERBATIM_QUOTE_FROM_SOURCE_1**
        - 🚩 **Violation Logic**: Explain why this law applies to the current draft and identify the specific wording in the protocol that risks a regulatory "Refuse to File" (RTF).

        ### SECTION 2: FDA HISTORICAL PRECEDENTS ###
        Format: 
        Recipient | Date | **VERBATIM_QUOTE_FROM_SOURCE_2**
        - 🚩 **Precedent Risk**: Explain how the current draft repeats the same mistake that led to the FDA rejection/deficiency in this historical case.

        ### SECTION 3: ACADEMIC ALIGNMENT ###
        Format: 
        [ID] | **CONCEPT**
        - 🚩 **Methodological Gap**: Why is this academic concept (e.g., MMRM, Estimand attributes) missing or poorly implemented in the current draft?

        ### SECTION 4: FINAL RISK ASSESSMENT TABLE ###
        | Category | Specific Violation | Source of Rule | Risk Level (High/Med) |
        | :--- | :--- | :--- | :--- |

        ### SECTION 5: REGULATORY VERDICT ###
        1. VERDICT: [PROCEED / CAUTION / REJECT]
        2. JUSTIFICATION: 
            If 'PROCEED', explain how the protocol successfully mitigates common industry pitfalls found in the Precedents. Be specific and use the sources as evidence.
            If 'CAUTION', explain what are the risk points that if not mitigated, could lead to FDA pushback, and how to mitigate them. Be specific and use the sources as evidence.
            If 'REJECT', explain what are the major flaws in the protocol, and why they would lead to an FDA rejection. Be specific and use the sources as evidence.
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
        # 1. TIERED CONTEXT LOADING
        # CHANGE: Explicitly set mode to "optimize" to trigger the dual-path RAG (Statutory + Intel)
        targeted_wisdom = self._load_library(search_query=original_protocol, mode="optimize")
        
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

            REQUIREMENT: Return 'PASS' only if all primary statistical risks (Multiplicity, Missing Data, Estimands) are resolved with citations. 
            Otherwise, list the specific failures.
            """
            check_result = self._generate_response(check_prompt)

            if "PASS" in check_result.upper():
                return candidate_version
            
            # If it didn't pass, update context for the next loop iteration
            current_protocol = candidate_version
            current_audit = check_result
            
        return current_protocol
    

