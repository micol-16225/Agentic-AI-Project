import os
from dotenv import load_dotenv
from lifecycle_agent import BiostatLifecycleAgent

# Load keys
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
library_path = "regulatory_library.txt"

# Initialize
agent = BiostatLifecycleAgent(api_key, library_path)

print("\n" + "="*50)
print("--- PHASE 1: ARCHITECT ---")
print("="*50)
initial_protocol = agent.architect_protocol("Phase III Zenziva Trial for Depression")
print(initial_protocol)

print("\n" + "="*50)
print("--- PHASE 2: AUDIT ---")
print("="*50)
# We feed the protocol from Phase 1 into Phase 2
audit_report = agent.audit_protocol(initial_protocol, "Historical Match: High dropouts in similar trials.")
print(audit_report)

print("\n" + "="*50)
print("--- PHASE 3: OPTIMIZE ---")
print("="*50)
# We feed the audit findings into Phase 3 to get the final version
final_protocol = agent.optimize_protocol(audit_report)
print(final_protocol)

print("\n" + "="*50)
print("--- LEARNING MODE: DEEP DIVE ---")
print("="*50)

# You can ask the agent to explain anything it mentioned in the Audit!
explanation = agent.explain_theory("Alpha-Spending and Multiplicity in Phase III")
print(explanation)

print("\n" + "="*50)
print("--- END OF TEST ---")
print("="*50)