import os
import time
from dotenv import load_dotenv
from lifecycle_agent3 import BiostatLifecycleAgent3

load_dotenv()

# --- THE FIX: Added *args and **kwargs to catch any extra arguments ---
def forced_tiny_load(self, *args, **kwargs):
    return "RULE: The use of the last observation carried forward (LOCF) approach is generally not acceptable."

# This replaces the logic in memory
BiostatLifecycleAgent3._load_library = forced_tiny_load

def quick_test():
    print("🛰️ Connecting to Auditor...")
    
    # We use the 70b model because the prompt is now tiny
    agent = BiostatLifecycleAgent3(
        api_key=os.getenv("GROQ_API_KEY"),
        library_path="master_regulatory_library.csv",
        model_id="llama-3.3-70b-versatile"
    )

    test_input = "The study will use LOCF for the primary endpoint analysis at Week 24."

    print("⚖️ Auditing (Final Attempt)...")
    try:
        # 15s delay to clear previous token counts
        time.sleep(15) 
        
        # We call the audit directly
        report = agent.audit_protocol(test_input, historical_lessons="None")
        
        print("\n" + "!"*30)
        print("IT WORKS! HERE IS THE AUDIT:")
        print("-" * 30)
        print(report)
        print("!"*30)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    quick_test()