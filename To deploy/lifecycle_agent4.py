import os
import torch
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

print(f"📍 Script is running in: {os.getcwd()}")

ACADEMIC_MANDATE = """
AUTHORITY: Senior Methodologist/FDA Reviewer. Zero filler. Use industry-standard terminology.
EVIDENTIARY RULES:
- PRIORITIZE: Knowledge Base (Papers/FDA Guidance) over general knowledge.
- CITATION: Every statistical claim MUST be cited (e.g., Akacha et al. 2017; ICH E9 R1).
- TRADEOFFS: Discuss methodological paths (e.g., MMRM vs. MI) via technical trade-offs only.
- $LATEX$: Use for all formulas and variables.
"""

class BiostatLifecycleAgent4:
    # Use the S-BioBERT version for semantic similarity
    _model = None 
    _embedding_cache = {} #short term memory for embeddings to avoid redundant computation
    _disk_cache_path = "brain_cache.json" # long-term memory for Q&A pairs to avoid redundant API calls

    def __init__(self, api_key, statutory_path="statutory_truth_with_ids.csv", intel_path="optimizer_intelligence_with_ids.csv", model_id="llama-3.1-8b-instant"):
        self.client = Groq(api_key=api_key)
        self.model_id = model_id
        
        # 1. Load Model (Light version)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Load CSVs
        self.stat_df = pd.read_csv(statutory_path)
        self.intel_df = pd.read_csv(intel_path)

        # 3. FAST LOAD LOGIC
        def get_or_load(file_path, df, type_filter):
            if os.path.exists(file_path):
                print(f"🚀 Loading pre-baked: {file_path}")
                return torch.load(file_path)
            else:
                print(f"⚠️ Not found. Encoding on-the-fly...")
                texts = df[df['type'] == type_filter]['content'].fillna("").tolist()
                return self.model.encode(texts, convert_to_tensor=True)

        # Load the baked brains
        self.acad_embeddings = get_or_load("acad_embeddings.pt", self.intel_df, 'Academic_Rigor')
        self.letter_embeddings = get_or_load("letter_embeddings.pt", self.stat_df, 'Precedent')
        self.stat_law_embeddings = get_or_load("stat_law_embeddings.pt", self.stat_df, 'Statutory')

        # 4. Prepare Reference List
        law_rows = self.stat_df[self.stat_df['type'] == 'Statutory']
        self.law_reference_list = "\n".join([f"ID: {r['doc_id']}" for _, r in law_rows.iterrows()])

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
        REFACTORED: Keeps your BioBERT logic but uses pre-loaded memory.
        """
        
        def get_semantic_context(self, df, doc_embeddings, query, limit=5):
            """Surgically updated to use pre-loaded doc_embeddings"""
            if df.empty or doc_embeddings is None: 
                return ""
            
            # Encode only the query (Fast)
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            
            # Compare against pre-loaded tensors (Instant)
            scores = util.cos_sim(query_embedding, doc_embeddings)[0]
            
            top_results = torch.topk(scores, k=min(len(df), limit))
            indices = top_results.indices.tolist()

            return "\n".join([
                f"- [{df.iloc[i].get('doc_id', 'NO_ID')}]: {df.iloc[i]['content']}" 
                for i in indices
            ])

        # --- ORCHESTRATION LOGIC ---
        if mode == "audit":
            # Uses the Reference List we built in __init__ 
            # (This satisfies your Jan 2nd requirement for Statutory Types)
            return f"### MANDATORY RULES (STATUTORY) ###\n{self.law_reference_list}"
        
        else:
            # Optimizer Mode: Now passing the actual embeddings
            stat_ctx = get_semantic_context(
                self.stat_law_df, 
                self.stat_law_embeddings, # FIX: Replaced None with actual embeddings
                search_query, 
                limit=3
            )
            
            acad_ctx = get_semantic_context(
                self.intel_df[self.intel_df['type'] == 'Academic_Rigor'], 
                self.acad_embeddings, 
                search_query, 
                limit=2
            )
            
            return f"### STATUTORY CONSTRAINTS ###\n{stat_ctx}\n\n### ACADEMIC RIGOR ###\n{acad_ctx}"


    def track_usage(self, prompt, response):
        # Rough estimation: 1 word ≈ 1.33 tokens
        in_tokens = len(prompt.split()) * 1.33
        out_tokens = len(response.split()) * 1.33
        
        # ADD flush=True here to bypass the buffer
        print(f"📈 [TOKEN TRACKER] In: {int(in_tokens)} | Out: {int(out_tokens)} | Total: {int(in_tokens + out_tokens)}", flush=True)
        
        return in_tokens + out_tokens

    
    def _generate_response(self, prompt, max_retries=5):

        # Move the MANDATORY RULES here, out of the user prompt
        SYSTEM_RULES = """
        You are a SKEPTICAL FDA Auditor. 
        1. If Protocol Evidence matches Statutory Text > 80%, you MUST output VIOLATION - LACKS SPECIFICITY.
        2. Do NOT assume intent. If it's not in the text, it's a VIOLATION BY OMISSION.
        3. Use ONLY the provided IDs.
        """

        response = None
        # 1. Check if we already know the answer
        #cached_answer = self._check_memory(prompt)
        #if cached_answer:
        #    print("🧠 Using cached memory (0 tokens used!)", flush=True)
        #    return cached_answer
        
        for i in range(max_retries):
            try:
                combined_system_prompt = f"{ACADEMIC_MANDATE}\n{SYSTEM_RULES}"
                
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": combined_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_id,
                    temperature=0.0, 
                    max_tokens=4096
                )
                
                response = chat_completion.choices[0].message.content
                finish_reason = chat_completion.choices[0].finish_reason
                if finish_reason == "length":
                    print("🚨 WARNING: Response truncated due to token limits!", flush=True)

                # ... after the API call ...
                print(f"DEBUG: Response Length (Chars): {len(response)}")
                print(f"DEBUG: Finish Reason: {chat_completion.choices[0].finish_reason}")
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
        UPDATED: Returns the doc_id to enable the Python Injector.
        """
        # 1. Get the unique ID from the row
        doc_id = precedent_row.get('doc_id', 'UNKNOWN_ID')
        source_text = precedent_row.get('content', '')
        source_name = precedent_row.get('source', 'FDA_Historical')

        prompt = f"""
        ROLE: FDA Statistical Auditor.
        SOURCE_MATERIAL: "{source_text}"
        SOURCE_ID: {doc_id}
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
        MATCH FOUND: ID: {doc_id} | [Recipient] | [Date] | [EXACT VERBATIM QUOTE]
        IF NO MATCH: Return "NO VALIDATED MATCH."
        """
        return self._generate_response(prompt)

    def audit_protocol(self, user_protocol, historical_lessons="", user_directives=""):
        """
        THE ORCHESTRATOR: Fully optimized with ID-based Injection logic.
        """
        # --- 1. PREPARE THE DATA (Wiring the new doc_ids) ---
        
        # SOURCE 1: THE LAW (Full list provided as requested)
        law_df = self.stat_df[self.stat_df['type'] == 'Statutory']
        # Including 'Type' for Jan 2nd requirement and keeping content accessible but distinct
        law_context = "\n".join([
            f"ID: {r['doc_id']} | Type: {r['type']} | Source: {r['source']} | Content: {r['content']}" 
            for _, r in law_df.iterrows()
        ])

        id_registry = "\n".join([f"- VALID_ID: {r['doc_id']} (Refers to: {r['title']})" 
                                 for _, r in law_df.iterrows()])

        # SOURCE 2: ACADEMIC PAPERS (Now with ID mapping)
        proto_embed = self.model.encode(user_protocol, convert_to_tensor=True)
        acad_scores = util.cos_sim(proto_embed, self.acad_embeddings)[0]
        top_acad_idx = torch.topk(acad_scores, k=min(len(self.acad_texts), 2)).indices.tolist()

        # FIX: Include the ID so the LLM doesn't hallucinate one
        acad_context = "\n".join([
            f"ID: {self.acad_df.iloc[idx]['doc_id']} | {self.acad_df.iloc[idx]['content']}" 
            for idx in top_acad_idx
        ])

        # SOURCE 3: FDA LETTERS (Ensuring workers receive the doc_id)
        prec_scores = util.cos_sim(proto_embed, self.letter_embeddings)[0]
        top_prec_idx = torch.topk(prec_scores, k=min(len(self.letter_texts), 3)).indices.tolist()
        candidates = [self.letter_df.iloc[idx] for idx in top_prec_idx]

        # --- 2. EXECUTE PARALLEL WORKERS ---
        results = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(self._run_single_precedent_check, user_protocol, c) for c in candidates]
            for f in as_completed(futures):
                try:
                    res = f.result()
                    if res and "MATCH FOUND" in res:
                        results.append(res)
                except Exception as exc:
                    print(f"🚨 Worker failed: {exc}")
                time.sleep(12) 

        letter_findings = "\n\n".join(results) if results else "No direct historical matches found."

        # --- 3. FINAL SYNTHESIS (Instructions preserved, Format updated for IDs) ---
        final_prompt = f"""
        {ACADEMIC_MANDATE}
        ROLE: FDA Statistical Reviewer (Audit).
        GOAL: Conduct a balanced regulatory review using the provided Knowledge Base IDs.

        ### STATUTORY ID REGISTRY (CRITICAL) ###
        You MUST use these IDs exactly as written. Any variation (adding "_R1_", changing case, etc.) will cause a system failure.
        {id_registry}

        --- SOURCE 1: THE LAW ---
        {law_context}

        --- SOURCE 2: FDA PRECEDENTS ---
        {letter_findings}

        --- SOURCE 3: ACADEMIC RIGOR ---
        {acad_context}

        PROTOCOL: {user_protocol}
        DIRECTIVES: {user_directives}
        LESSONS: {historical_lessons}

        
        ### MANDATORY VALIDATION RULES (VERSION 2.1 - STRICT ADVERSARIAL) ###

        1. ADVERSARIAL MENTALITY: You are a skeptical FDA Lead Reviewer. Your goal is to find reasons the Protocol MIGHT fail. Do not assume intent; if it is not explicitly written with technical detail, it does not exist.

        2. EVIDENCE-BASED AUDIT: Every 'COMPLIANT' status MUST cite a specific section or tag (e.g., [Section 4.2] or [p. 12]). If no text addresses the requirement, it is a 'VIOLATION BY OMISSION'.

        3. THE ANTI-PARROT RULE: If the 'Protocol Evidence' simply restates the 'Statutory Text' or uses circular reasoning (e.g., "The study is adequate because it follows 21 CFR 314.126"), you MUST flag it as 'VIOLATION - LACKS SPECIFICITY'. To be COMPLIANT, the evidence must describe *how* (e.g., "Using a central IRT system for randomization").

        4. ID INTEGRITY & TRUTH: Use the exact [doc_id] from the Registry. I will inject the statutory text; your job is the "Verdict" and "Logic."

        5. CROSS-SECTION COHERENCE: The "Regulatory Verdict" (Section 5) must be the logical sum of all previous sections. You cannot mark a rule as 'COMPLIANT' in Section 1 and then list it as a 'High Risk' in Section 4. 

        6. NO HALLUCINATED PRACTICES: Do not assume "standard medical practice." If the protocol doesn't mention a specific control (like blinding), you cannot assume it is happening.

        REPORT STRUCTURE:

        ### SECTION 1: REGULATORY ALIGNMENT ###
        Format: [doc_id] | Status: [COMPLIANT / VIOLATION]
        - Protocol Evidence: Cite the specific from the protocol that addresses this law.
        - Reviewer Logic: Critically analyze the evidence. Explain whether and why the protocol aligns with or breaches the specific statutory requirement. Do NOT paraphrase NOT quote the law; only apply the law to the protocol's content and logic. If the protocol is compliant, explain how it successfully meets the requirement. If it's a violation, explain what is missing or incorrect in the protocol that leads to non-compliance.
        - IMPORTANT: Use the exact [doc_id] from the Registry above. Do not invent nor modify IDs. Do NOT quote the law; I will inject the full text automatically. Keep your analysis focused and straight to the point. Avoid fluff. 

        ### SECTION 2: FDA HISTORICAL PRECEDENTS ###
        Format: [Precedent ID] | Status: [RISK POTENTIAL / NO RISK]
        - Risk: Explain whether or not the protocol could incur into a rejection similar to the historical rejections. Give reasons for your assessment, comparing the protocol text with the precedent's content. If you identify a risk, specify which part of the protocol is similar to the violation in the precedent and why that could be problematic. If you assess no risk, explain what aspects of the protocol differentiate it from the historical violations and why those differences mitigate the risk.

        ### SECTION 3: ACADEMIC ALIGNMENT ###
        - MANDATE: You may ONLY audit against the IDs provided in 'SOURCE 3: ACADEMIC RIGOR'. 
        - Do NOT use internal knowledge to invent new academic IDs. If only 2 IDs are provided, Section 3 must only have 2 entries.
        Format: [ID] | Status: [ALIGNED / MISALIGNED]
        - Logic: Explain whether or not the academic concept presented in the paper is addressed / applied properly or not in the protocol. If it's aligned, explain how the protocol successfully incorporates the academic principle and why that is important for the trial's success. If it's misaligned, explain what is missing or incorrectly applied in the protocol compared to the academic standard, and what potential consequences that could have on the trial's validity or regulatory acceptance.


        ### SECTION 4: FINAL RISK ASSESSMENT TABLE ###
        | Category | Specific Risk | Source of Rule | Risk Status (Very Low / Medium / High) |
        | :--- | :--- | :--- | :--- |

        ### SECTION 5: REGULATORY VERDICT ###
        1. VERDICT: [PROCEED / CAUTION / REJECT]
        2. JUSTIFICATION: 
            If 'PROCEED', explain how the protocol successfully mitigates common industry pitfalls found in the Precedents. Be specific and use the sources as evidence.
            If 'CAUTION', explain what are the risk points that if not mitigated, could lead to FDA pushback, and how to mitigate them. Be specific and use the sources as evidence.
            If 'REJECT', explain what are the major flaws in the protocol, and why they would lead to an FDA rejection. Be specific and use the sources as evidence.
       
        ### CROSS-SECTION AUDIT (MANDATORY) ###
        Before finalizing, double-check that every 'COMPLIANT' in Section 1 is NOT listed as a 'Medium' or 'High' risk in Section 4. If a risk is identified in Section 4, Section 1 MUST reflect that as a 'VIOLATION' or 'RISK POTENTIAL'.
        """
        
        # Generate the initial "Lean" report
        report = self._generate_response(final_prompt)
        if not report: return "🚨 Error: Synthesis Failed"

        # --- 4. THE PYTHON INJECTOR (Post-Processing) ---
        sorted_indices = self.stat_df['doc_id'].str.len().sort_values(ascending=False).index
        sorted_df = self.stat_df.iloc[sorted_indices]

        enriched_report = report
        for _, row in sorted_df.iterrows():
            did = str(row['doc_id'])
            
            # This only works if the LLM is 100% PERFECT with the ID
            if did in enriched_report:
                if row['type'] == 'Statutory':
                    # Only inject if we haven't injected this specific content block yet
                    if f"Statutory Text ({did})" not in enriched_report:
                        injection = f"\n> **Statutory Text ({did}):** {row['content']}\n"
                        enriched_report = enriched_report.replace(did, f"**{did}**{injection}")
                else:
                    # Bold other IDs
                    enriched_report = enriched_report.replace(did, f"**{did}**")

        return enriched_report


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
    
    def optimize_protocol(self, original_protocol, audit_report, user_directives="None", max_iterations=5):
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
    

