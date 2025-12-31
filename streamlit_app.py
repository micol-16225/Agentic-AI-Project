import streamlit as st
import pandas as pd 
import os
from dotenv import load_dotenv
from lifecycle_agent import BiostatLifecycleAgent

# Load Environment
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Cache the agent to avoid re-initialization on every interaction
@st.cache_resource
def get_agent(api_key, library_path):
    return BiostatLifecycleAgent(api_key, library_path)

# Initialize Agent
agent = get_agent(api_key, "regulatory_library.txt")

# --- POP-UP DIALOG FOR FULL LETTER VIEWING ---
@st.dialog("Original FDA Correspondence", width="large")
def show_full_letter(name_to_find):
    df = pd.read_csv("fda_letters.csv")
    
    # --- SAFE FIX START ---
    # 1. Force the 'recipient' column to be text (strings)
    # 2. Fill any empty boxes with an empty space so it doesn't crash
    df['recipient'] = df['recipient'].astype(str).fillna('')
    # --- SAFE FIX END ---
    
    # Now this line won't crash!
    match = df[df['recipient'].str.contains(name_to_find, case=False, na=False)]
    
    if not match.empty:
        st.subheader(f"Full Letter to {match.iloc[0]['recipient']}")
        st.caption(f"Date: {match.iloc[0]['date']}")
        st.divider()
        st.text_area("Official Text", value=match.iloc[0]['full_text'], height=600)
    else:
        st.error(f"Could not find a letter for '{name_to_find}' in the database.")


# --- INITIALIZE SESSION STATE ---
# We use this to ensure data persists across tab switches and button clicks
if 'protocol' not in st.session_state: st.session_state.protocol = None
if 'audit' not in st.session_state: st.session_state.audit = None
if 'final_output' not in st.session_state: st.session_state.final_output = None
if 'user_notes' not in st.session_state: st.session_state.user_notes = ""

# --- UI CONFIG ---
st.set_page_config(page_title="BioStat Enterprise AI", layout="wide", page_icon="🧬")
agent = BiostatLifecycleAgent(api_key, "regulatory_library.txt")

# --- SIDEBAR (MLOps Explanation) ---
with st.sidebar:
    st.title("🛡️ System Control")
    st.status("Infrastructure: Active", state="complete")
    st.caption("v1.2.0-Production-ICH")
    st.divider()
    
    with st.expander("🏗️ MLOps Architecture"):
        st.write("""
        - **Decoupled Logic:** UI and AI are separate.
        - **RAG-Ready:** Grounded in regulatory text.
        - **Audit Trail:** Step-by-step history.
        """)
    
    with st.expander("ℹ️ Why 3-Steps?"):
        st.write("Ensures **Transparency** and **Human-in-the-loop** oversight.")

# --- MAIN APP ---
st.title("🧬 Biostatistical Lifecycle Orchestrator")

tab1, tab2, tab3 = st.tabs(["🏗️ Phase 1: Input", "🔍 Phase 2: Audit", "⚡ Phase 3: Optimize"])

# --- TAB 1: ARCHITECT ---
with tab1:
    st.header("Step 1: Protocol Architecture")
    mode = st.radio("Source:", ["AI Architect", "Upload .txt Draft"], horizontal=True)
    
    if mode == "AI Architect":
        drug_input = st.text_input("Drug & Indication", "Zenziva for Treatment-Resistant Depression")
        if st.button("Generate Draft"):
            with st.spinner("Drafting..."):
                st.session_state.protocol = agent.architect_protocol(drug_input)
    else:
        file = st.file_uploader("Upload Protocol", type=['txt'])
        if file:
            st.session_state.protocol = file.getvalue().decode("utf-8")

    if st.session_state.protocol:
        st.markdown("---")
        st.markdown(st.session_state.protocol)

with tab2:
    st.header("Step 2: Adversarial Audit")
    
    if st.session_state.protocol:
        if st.button("Run Regulatory Scan"):
            with st.spinner("Performing Semantic Search & Regulatory Audit..."):
                st.session_state.audit = agent.audit_protocol(
                    st.session_state.protocol, 
                    "General Historical Lessons"
                )
        
        if st.session_state.audit:
            # --- 1. THE TABLE ---
            st.subheader("📋 Regulatory Analysis")
            # Reformat the markdown table for better display
            formatted_audit = st.session_state.audit.replace("Risk Level |", "Risk Level |\n| --- | --- | --- |")
            st.markdown(formatted_audit)
            
            st.divider()

            # --- 3. THE EVIDENCE POP-UPS (The new part you wanted) ---
            st.subheader("📁 Evidence Deep-Dive")
            st.write("Click a button below to see the full original FDA letter:")
            
            ev_cols = st.columns(3)
            col_idx = 0
            for line in st.session_state.audit.split('\n'):
                if "MATCH FOUND:" in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        company_name = parts[1].strip()
                        with ev_cols[col_idx % 3]:
                            if st.button(f"📖 View {company_name}", key=f"old_btn_{col_idx}"):
                                show_full_letter(company_name)
                        col_idx += 1

            st.divider()

            # --- 4. THE NOTES SECTION ---
            st.session_state.user_notes = st.text_area(
                "✍️ Senior Reviewer Directives (Your Expert Input)", 
                placeholder="e.g., 'Change alpha-spending to O'Brien-Fleming'"
            )
    else:
        st.info("Awaiting protocol from Phase 1...")

# --- TAB 3: OPTIMIZER (With Export) ---
with tab3:
    st.header("Step 3: Optimization & Export")
    if st.session_state.audit:
        if st.button("🚀 Finalize Optimized Protocol"):
            with st.spinner("Integrating Expert Feedback..."):
                st.session_state.final_output = agent.optimize_protocol(
                    st.session_state.audit, st.session_state.user_notes
                )
        
        if st.session_state.final_output:
            st.markdown(st.session_state.final_output)
            
            # --- OPTIMIZED DOWNLOAD BUTTONS ---
            st.divider()
            st.subheader("📥 Export Final Draft")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Download Text (.txt)", st.session_state.final_output, "protocol.txt")
            with col2:
                st.download_button("Download Markdown (.md)", st.session_state.final_output, "protocol.md")
    else:
        st.info("Run Audit in Phase 2 first.")