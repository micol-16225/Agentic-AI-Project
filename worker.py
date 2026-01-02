import time
import json
from lifecycle_agent3 import BiostatLifecycleAgent3 # Your existing class

# INITIALIZE YOUR AGENT
agent = BiostatLifecycleAgent3(
    api_key="YOUR_GROQ_KEY",
    library_path="master_regulatory_library.csv"
)

def process_audit_job(protocol_data):
    """
    This is the core 'Worker' logic. 
    It doesn't care about loops; it only cares about ONE job.
    """
    try:
        print(f"🚀 Starting audit for: {protocol_data['id']}")
        
        # 1. RUN THE AUDIT
        report = agent.audit_protocol(
            user_protocol=protocol_data['text'],
            historical_lessons=protocol_data.get('lessons', "")
        )
        
        # 2. SAVE RESULTS (This would go to an AWS S3 bucket later)
        with open(f"audit_results_{protocol_data['id']}.txt", "w") as f:
            f.write(report)
            
        return True # Job Success

    except Exception as e:
        if "429" in str(e):
            print(f"🛑 RATE LIMIT REACHED. Strategy: Back-off and Re-queue.")
            return "RETRY" # Signal to the queue to try again later
        else:
            print(f"❌ CRITICAL FAILURE: {e}")
            return False