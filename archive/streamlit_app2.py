import streamlit as st
import pandas as pd 
import os
from dotenv import load_dotenv
from lifecycle_agent2 import BiostatLifecycleAgent

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

@st.cache_resource
def get_agent(api_key, library_path):
    return BiostatLifecycleAgent(api_key, library_path)

agent = get_agent(api_key, "regulatory_library.txt")

# --- INITIALIZE SESSION STATE (Updated for State & Dialogue) ---
if 'protocol' not in st.session_state: st.session_state.protocol = None
if 'audit' not in st.session_state: st.session_state.audit = None
if 'final_output' not in st.session_state: st.session_state.final_output = None
if 'user_notes' not in st.session_state: st.session_state.user_notes = ""
if 'history' not in st.session_state: st.session_state.history = [] # State awareness
if 'chat_history' not in st.session_state: st.session_state.chat_history = [] # Dialogue

# ... [show_full_letter function remains exactly as you had it] ...

st.set_page_config(page_title="BioStat Enterprise AI", layout="wide", page_icon="🧬")

st.title("🧬 Biostatistical Lifecycle Orchestrator")
tab1, tab2, tab3 = st.tabs(["🏗️ Phase 1: Input", "🔍 Phase 2: Audit", "⚡ Phase 3: Optimize"])

with tab1:
    st.header("Step 1: Protocol Architecture")
    # ... [Tab 1 code remains identical] ...

with tab2:
    st.header("Step 2: Adversarial Audit")
    if st.session_state.protocol:
        # POINT 2: STATE AWARENESS
        # If we have an optimized version, audit that instead!
        doc_to_audit = st.session_state.final_output if st.session_state.final_output else st.session_state.protocol

        if st.button("Run Regulatory Scan"):
            with st.spinner("Bilateral Grounding in Progress..."):
                st.session_state.audit = agent.audit_protocol(
                    doc_to_audit, 
                    "General Historical Lessons",
                    user_directives=st.session_state.user_notes,
                    history="\n".join(st.session_state.history) # Pass history to AI
                )
                st.session_state.history.append(f"Audit run on {pd.Timestamp.now()}")

        if st.session_state.audit:
            # 📋 Table & Quick Scan
            st.markdown(st.session_state.audit.split("###")[0])
            
            # 🚩 Color Coded Alerts (Restored for visual impact)
            for line in st.session_state.audit.split('\n'):
                if "HIGH" in line.upper(): st.error(line)
                elif "MEDIUM" in line.upper(): st.warning(line)

            # 📁 Evidence Pop-ups
            st.divider()
            st.subheader("📁 Evidence Deep-Dive")
            ev_cols = st.columns(3)
            col_idx = 0
            for line in st.session_state.audit.split('\n'):
                if "MATCH FOUND:" in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        company_name = parts[1].strip()
                        if ev_cols[col_idx % 3].button(f"📖 View {company_name}", key=f"btn_{col_idx}"):
                            show_full_letter(company_name)
                        col_idx += 1

            # POINT 3: EXPERT-IN-THE-LOOP DIALOGUE
            st.divider()
            st.subheader("💬 Challenge the Auditor")
            user_q = st.chat_input("Ask why a certain risk was flagged...")
            if user_q:
                with st.spinner("Consulting FDA Precedents..."):
                    answer = agent.discuss_audit(st.session_state.audit, user_q)
                    st.session_state.chat_history.append((user_q, answer))
            
            for q, a in st.session_state.chat_history:
                with st.chat_message("user"): st.write(q)
                with st.chat_message("assistant"): st.write(a)

            st.session_state.user_notes = st.text_area("✍️ Final Directives for Optimizer", value=st.session_state.user_notes)

with tab3:
    st.header("Step 3: Optimization & Export")
    if st.session_state.audit:
        if st.button("🚀 Finalize Optimized Protocol"):
            with st.spinner("Heavy Lifting: Cross-referencing Expert Directives..."):
                # POINT 1: HEAVY LIFTING
                # We feed the ACTUAL protocol + the audit + the notes
                st.session_state.final_output = agent.optimize_protocol(
                    st.session_state.protocol, 
                    st.session_state.audit, 
                    st.session_state.user_notes
                )
                st.session_state.history.append("Optimization v1 applied.")

        if st.session_state.final_output:
            st.markdown(st.session_state.final_output)
            st.download_button("Download Markdown (.md)", st.session_state.final_output, "protocol.md")