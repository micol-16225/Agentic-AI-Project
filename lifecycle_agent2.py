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
        self.knowledge_base = ""
        # Added: Persistent Memory for state awareness
        self.version_history = [] 

    def _load_library(self, search_query=""):
        content = ""
        if os.path.exists("fda_letters.csv"):
            df = pd.read_csv("fda_letters.csv")
            all_letters = df['full_text'].fillna("").tolist()
            
            if search_query and len(all_letters) > 0:
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(all_letters + [search_query])
                cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
                related_indices = cosine_sim[0].argsort()[-8:][::-1]
                
                risk_keywords = ["multiplicity", "alpha", "p-value", "dropout", "missing data", "estimand", "bias", "sample size"]
                
                content += "### RELEVANT FDA PRECEDENTS (INTELLIGENT EXTRACTION) ###\n"
                for _, row in df.iloc[related_indices].iterrows():
                    text = str(row['full_text'])
                    sentences = re.split(r'(?<=[.!?]) +', text)
                    meaningful_chunks = [s for s in sentences if any(k in s.lower() for k in risk_keywords)]
                    summary = " ".join(sentences[:2] + meaningful_chunks[:5]) 
                    content += f"\n--- LETTER: {row['date']} | {row['recipient']} ---\n{summary}...\n"
        return content

    def _generate_response(self, prompt):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_id,
                temperature=0.1 # Enforce precision
            )
            response = chat_completion.choices[0].message.content
            clean_response = response.replace("\\(", "$").replace("\\)", "$")
            clean_response = clean_response.replace("\\[", "$$").replace("\\]", "$$")
            return clean_response
        except Exception as e:
            return f"🚨 Connection Error: {str(e)}"

    def architect_protocol(self, drug_description):
        prompt = f"""
        ROLE: Lead Clinical Architect & Senior Biostatistician.
        TASK: Draft a Phase III Protocol for {drug_description}.
        ... [Existing instructions preserved] ...
        """
        return self._generate_response(prompt)

    # ENHANCEMENT: Added 'history' to ensure State Awareness
    def audit_protocol(self, user_protocol, historical_lessons, user_directives="", history=""):
        self.knowledge_base = self._load_library(search_query=user_protocol)

        prompt = f"""
        ROLE: FDA Statistical Reviewer (Adversarial Audit).
        KNOWLEDGE: {self.knowledge_base}
        PAST FAILURES: {historical_lessons}
        VERSION HISTORY: {history if history else "Initial Version"}
        USER DIRECTIVES: {user_directives if user_directives else "Follow standard regulatory guardrails."}

        TASK: Critique the following protocol for 'Submission-Killing' flaws:
        {user_protocol}

        ENHANCED AUDIT RULES (BILATERAL GROUNDING):
        1. For every 'High' or 'Medium' risk, you MUST cite the exact line from the user's protocol.
        2. You MUST acknowledge if a previous risk mentioned in the HISTORY has been fixed.
        3. If a safeguard is found (e.g. Bonferroni-Holm), you MUST move the risk to LOW.

        OUTPUT FORMAT:
        - **Risk Table** (Category | Risk | Level | Evidence Source)
        - ### 📑 VERIFICATION LOG:
        - [Category]: Protocol Quote: "[Verbatim]" | FDA Precedent: "[Verbatim Match]" | Verdict: [Reasoning]
        """
        return self._generate_response(prompt)

    # ENHANCEMENT: Expert-in-the-Loop Dialogue
    def discuss_audit(self, audit_report, user_question):
        prompt = f"""
        ROLE: FDA Auditor. 
        CONTEXT: You just performed this audit: {audit_report}
        USER CHALLENGE: {user_question}
        
        TASK: Defend or adjust your audit logic. Be highly technical and reference FDA standards.
        """
        return self._generate_response(prompt)

    def optimize_protocol(self, original_protocol, audit_report, user_directives="None"):
        fda_context = self._load_library(search_query=original_protocol)
        expert_knowledge = ""
        if os.path.exists(self.library_path):
            with open(self.library_path, 'r') as f:
                expert_knowledge = f.read()

        prompt = f"""
        ROLE: Principal Biostatistician & Regulatory Strategist.
        ... [Your existing gold-standard logic] ...
        GUARANTEE: Ensure the new version bypasses the Auditor's risks.
        """
        return self._generate_response(prompt)