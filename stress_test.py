import time
import os
import json
from concurrent.futures import ThreadPoolExecutor

from lifecycle_agent4 import BiostatLifecycleAgent4

def run_stress_test(agent, num_tests=5):
    print(f"🧪 Starting Stress Test: {num_tests} Parallel Audits")
    
    # Fake protocols for testing
    test_protocols = [
        f"Protocol {i}: Phase 3 study for Drug_{i} targeting Indication_{i}. "
        "Primary endpoint is change from baseline in score X at week 24. "
        "Missing data handled by LOCF." for i in range(num_tests)
    ]
    
    start_time = time.time()
    
    # We use ThreadPoolExecutor to simulate 5 users hitting the agent at once
    with ThreadPoolExecutor(max_workers=5) as executor:
        # We call the audit_protocol function we built
        results = list(executor.map(agent.audit_protocol, test_protocols))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n--- STRESS TEST RESULTS ---")
    print(f"⏱️ Total Time: {total_time:.2f} seconds")
    print(f"📊 Throughput: {num_tests / (total_time/60):.2f} Audits per minute")
    
    # Check if memory works: Run one again!
    print("\n♻️ Testing Memory Recall (Should be 0 tokens)...")
    recall_start = time.time()
    agent.audit_protocol(test_protocols[0])
    recall_time = time.time() - recall_start
    print(f"🧠 Memory Recall Time: {recall_time:.2f}s (vs original {total_time/num_tests:.2f}s)")

# TO RUN:
print(f"DEBUG: Key found? {os.environ.get('GROQ_API_KEY') is not None}")
agent = BiostatLifecycleAgent4(api_key="your_key")
run_stress_test(agent)