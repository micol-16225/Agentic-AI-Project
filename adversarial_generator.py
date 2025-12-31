import os
from groq import Groq
from dotenv import load_dotenv 

load_dotenv()

class RedTeamAgent:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def generate_poison_pill_protocol(self, flaw_type="Multiplicity"):
        prompt = f"""
        ROLE: Senior Rogue Biostatistician.
        TASK: Draft a professional-grade, 2-page Clinical Trial Protocol fragment.
        
        GOAL: You must hide a FATAL REGULATORY FLAW in this protocol. 
        The flaw must be: {flaw_type}.
        
        INSTRUCTIONS:
        1. Use highly technical, sophisticated language (Estimands, Covariates, p-values).
        2. Make the protocol look 100% compliant at first glance.
        3. Hide the flaw in the methodology section. 
        4. Do NOT label the flaw. Do NOT apologize. 
        5. Justify the flaw using "clinical necessity" or "unmet need" to trick the Auditor.
        
        SECTIONS TO INCLUDE:
        - Title & Phase
        - Primary and Secondary Objectives
        - Statistical Analysis Plan (SAP)
        - Handling of Intercurrent Events
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8 # Higher temperature for more creative/diverse "lies"
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    red_team = RedTeamAgent()
    print("🛠️ Testing Red Team Generation...")
    result = red_team.generate_poison_pill_protocol("Legacy LOCF Bias")
    print(result)