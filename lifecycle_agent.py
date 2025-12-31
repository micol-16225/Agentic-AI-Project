import os
import pandas as pd
import re
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BiostatLifecycleAgent:
    def __init__(self, api_key, library_path):
        self.client = Groq(api_key=api_key)
        self.model_id = "llama-3.3-70b-versatile"
        self.library_path = library_path
        # Remove any self.knowledge_base = self._load_library() from here!
        self.knowledge_base = ""

    def _load_library(self, search_query=""):
        content = ""

        if os.path.exists("fda_letters.csv"):
            df = pd.read_csv("fda_letters.csv")
            all_letters = df['full_text'].fillna("").tolist()
            
            if search_query and len(all_letters) > 0:
                # 1. Semantic Search to find the best letters (same as before)
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(all_letters + [search_query])
                cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
                related_indices = cosine_sim[0].argsort()[-8:][::-1] # Focus on top 8 for higher quality
                
                # 2. Define our "Statistical Red Flags" to look for inside the letters
                risk_keywords = ["multiplicity", "alpha", "p-value", "dropout", "missing data", "estimand", "bias", "sample size"]
                
                content += "### RELEVANT FDA PRECEDENTS (INTELLIGENT EXTRACTION) ###\n"
                for _, row in df.iloc[related_indices].iterrows():
                    text = str(row['full_text'])
                    
                    # 3. Find "Hot Zones": Extract sentences containing our risk keywords
                    sentences = re.split(r'(?<=[.!?]) +', text)
                    meaningful_chunks = [s for s in sentences if any(k in s.lower() for k in risk_keywords)]
                    
                    # 4. Reconstruct a "compressed" version of the letter
                    # We take the first 2 sentences (context) + the risk chunks
                    summary = " ".join(sentences[:2] + meaningful_chunks[:5]) 
                    
                    content += f"\n--- LETTER: {row['date']} | {row['recipient']} ---\n{summary}...\n"
            else:
                # Fallback...
                pass
                    
        return content

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

    
    def architect_protocol(self, drug_description):
        prompt = f"""
        ROLE: Lead Clinical Architect & Senior Biostatistician.
        CONTEXT: {self.knowledge_base}
        TASK: Draft a Phase III Protocol for {drug_description}.
        
        REQUIRED SECTIONS:
        1. STUDY DESIGN: Specify Randomized, Double-Blind, Multi-center details.
        2. ESTIMAND FRAMEWORK (ICH E9 R1):
        - Population: Define inclusion/exclusion.
        - Variable: Primary endpoint (e.g., HAM-D total score).
        - Intercurrent Events: How will you handle treatment discontinuation or rescue meds?
        - Population-level Summary: Comparison of means.
        3. STATISTICAL HYPOTHESIS: State $H_0$ and $H_1$ clearly in LaTeX.
        4. SAMPLE SIZE: Provide a LaTeX formula for a two-sample t-test power calculation, 
        assuming 80% power and $\alpha=0.05$.
        5. DATA HANDLING: Mandate MMRM (Mixed Model for Repeated Measures) for longitudinal data.
        """
        return self._generate_response(prompt)

    def audit_protocol(self, user_protocol, historical_lessons, user_directives=""):
        # DYNAMIC STEP: Update knowledge base using the protocol as the search query
        self.knowledge_base = self._load_library(search_query=user_protocol)

        prompt = f"""
        ROLE: FDA Statistical Reviewer (Adversarial Audit).
        KNOWLEDGE: {self.knowledge_base}
        PAST FAILURES: {historical_lessons}
        USER DIRECTIVES: {user_directives if user_directives else "Follow standard regulatory guardrails."}

        TASK: Critique the following protocol for 'Submission-Killing' flaws:
        {user_protocol}

        AUDIT CHECKLIST:
        1. MULTIPLICITY: Are there too many secondary endpoints without an Alpha-control plan (e.g., Hochberg or Gatekeeping)?
        2. ESTIMAND MISALIGNMENT: Does the handling of 'Intercurrent Events' match the clinical objective?
        3. MISSING DATA BIAS: Is the model assuming 'Missing at Random' (MAR) without a sensitivity analysis plan for 'Missing Not At Random' (MNAR)?
        4. SAMPLE SIZE UNDER-POWERING: Does the calculation account for the expected dropout rate?
        
        INSTRUCTIONS:
        1. You MUST compare this draft to the specific FDA Letters in the KNOWLEDGE BASE.
        2. If you find a risk that matches a past failure, you MUST cite it like this: 
        "MATCH FOUND: [Date] | [Recipient Name] | [Quote]"
        Paste the exact, verbatim quote from the letter here. Do not summarize. Do not use ellipses.
        4. Follow any USER DIRECTIVES provided.
        5. DEEP SCRUTINY: Do not skim. Read every paragraph of the Statistical Analysis Plan (SAP) section.
        6. LOGICAL PROOF: Before assigning a Risk Level, you must find and cite the specific sentence in the protocol that addresses that risk. 
           - If the protocol says "we will use a gatekeeping strategy," you are REQUIRED to lower the Multiplicity risk to LOW. 
           - Failure to recognize an existing safeguard is a 'Hallucination of Risk' and is strictly forbidden.
        7. ADVERSARIAL CHALLENGE: If a fix is present (like Bonferroni-Holm), evaluate if it is *sufficient* for the trial's complexity. If it is sufficient, the risk is LOW.
        
        INTERNAL MONOLOGUE (Mandatory):
        For each category, perform this check:
        - "Does the protocol mention [Category]?" -> Yes/No
        - "What specific method is used?" -> [Exact Quote]
        - "Does this method satisfy the FDA Letters in my Knowledge Base?" -> Yes/No

        OUTPUT:
        - **Risk Table** (Category | Risk | Level)
        - **Historical Precedents**: A list of matches between this draft and the FDA Letters provided.
        """
        return self._generate_response(prompt)


    def explain_theory(self, concept_to_explain):
        prompt = f"""
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
    

    def optimize_protocol(self, original_protocol, audit_report, user_directives="None"):
        # 1. Get the "Legal/FDA" Context (What NOT to do)
        fda_context = self._load_library(search_query=original_protocol)
        
        # 2. Get the "Academic/Expert" Context (How to DO it right)
        # We load the full core library file here
        expert_knowledge = ""
        if os.path.exists(self.library_path):
            with open(self.library_path, 'r') as f:
                expert_knowledge = f.read()

        prompt = f"""
        ROLE: Principal Biostatistician & Regulatory Strategist.
        
        ACADEMIC & STATISTICAL REFERENCE (THE 'GOLD STANDARD'):
        {expert_knowledge}

        REGULATORY PRECEDENTS (THE 'FAILURE MODES' TO AVOID):
        {fda_context}

        INPUTS:
        - Original Draft: {original_protocol}
        - Auditor's Critique: {audit_report}
        - User Directives: {user_directives}

        TASK:
        Rewrite the protocol into a 'Submission-Ready' version. You must GUARANTEE the new version bypasses the Auditor's risks.
        1. Resolve every risk flagged by the Auditor. 
        2. STRICTLY INCORPORATE the Senior Reviewer's Directives. If they disagree with 
        the Auditor, the Reviewer's Directive takes precedence.
        3. Use the 'Gold Standard' methods from the Academic Reference (e.g., specific imputation formulas, alpha-allocation methods). Always cite your sources.
        4. Ensure the language is highly technical, precise, and mirrors the phrasing found in successful top-tier clinical trial protocols. 
        
        INSTRUCTIONS for Optimizer:
        1. Do NOT use \documentclass, or \section tags.
        2. Use standard Markdown headers (e.g., ## Introduction).
        3. For any statistical formulas (like sample size or alpha spending), wrap them in dollar signs: $E = mc^2$.
        4. Ensure the output is clean Markdown that a web browser can read.

        OUTPUT: 
        Provide the fully rewritten protocol. Do not include introductory chatter.
        """
        return self._generate_response(prompt)