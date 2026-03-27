import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from lifecycle_agent4 import BiostatLifecycleAgent4, ACADEMIC_MANDATE

# --- 1. CONFIGURATION & STATE ---
st.set_page_config(page_title="Clinical Trial Protocol Auditor v1", layout="wide", page_icon="🧬")

# Load Environment
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize Session State for Agent 4's specific outputs
# This ensures these variables exist even before the user clicks any buttons
if 'answers' not in st.session_state:
    st.session_state.answers = {}

if 'protocol' not in st.session_state:
    st.session_state.protocol = None

if 'questions' not in st.session_state:
    st.session_state.questions = None

if 'audit_report' not in st.session_state:
    st.session_state.audit_report = None

if 'final_protocol' not in st.session_state:
    st.session_state.final_protocol = None

# --- 2. AGENT 4 INITIALIZATION ---
@st.cache_resource
def load_production_agent():
    return BiostatLifecycleAgent4(
        api_key=api_key,
        statutory_path="statutory_truth_with_ids.csv",
        intel_path="optimizer_intelligence_with_ids.csv"
    )

agent = load_production_agent()

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


# --- 4. MAIN INTERFACE ---
tab1, tab2, tab3 = st.tabs(["🏗️ Architect", "🔍 Tiered Audit", "🚀 Optimize"])

# --- TAB 1: ARCHITECT (The Interview Path) ---
with tab1:
    st.header("🏗️ Phase 1: Clinical Design Interview")

    # Let the user choose their path
    entry_mode = st.radio(
        "How would you like to start?",
        ["New Design (AI Interview)", "Upload Existing Draft (.txt)"],
        horizontal=True
    )

    st.divider()

    # --- PATH A: AI INTERVIEW ---
    if entry_mode == "New Design (AI Interview)":
    
        # 1. Input basic drug info to start the process
        col_a, col_b = st.columns(2)
        drug_name = col_a.text_input("Target Drug Name", "Zenziva")
        indication = col_b.text_input("Primary Indication", "Treatment-Resistant Depression")

        # STEP 1: TRIGGER THE INTERROGATOR
        if st.button("🎤 Start Statistical Interview"):
            with st.spinner("Lead Methodologist is preparing sharp questions..."):
                # Call Agent 1
                st.session_state.questions = agent.generate_interview_questions(drug_name, indication)
                # Reset final protocol until interview is finished
                st.session_state.protocol = None 

        # STEP 2: DISPLAY THE INTERVIEW FORM
        if 'questions' in st.session_state and st.session_state.questions:
            st.divider()
            st.subheader("📋 Lead Methodologist's Interview")
            st.caption("Tip: If you aren't sure about a specific parameter, leave it blank. The Architect will suggest a regulatory 'Gold Standard' for you.")
            
            q_list = [q.strip() for q in st.session_state.questions.split('\n') if q.strip()]
            
            with st.form("interview_submission"):
                user_responses = {}
                for i, question in enumerate(q_list):
                    # We display the question, and provide a text area
                    user_responses[f"q_{i}"] = st.text_area(
                        label=f"Q{i+1}: {question}", 
                        placeholder="Enter your intent here, or leave blank for AI suggestion...",
                        key=f"ans_{i}"
                    )
                
                if st.form_submit_button("🏗️ Finalize Design & Draft Protocol"):
                    # Check if all are empty (optional, just to warn the user)
                    all_empty = all(v.strip() == "" for v in user_responses.values())
                    
                    with st.spinner("Architect is synthesizing your input with industry standards..."):
                        st.session_state.answers = user_responses
                        st.session_state.protocol = agent.draft_protocol_from_interview(
                            {"drug": drug_name, "ind": indication}, 
                            user_responses
                        )
        # DISPLAY THE RESULTING PROTOCOL
        if st.session_state.protocol:
            st.divider()
            st.success("SAP Protocol Drafted Successfully!")
            st.markdown(st.session_state.protocol)

        # --- LOGIC DEFENSE SECTION: PROBING THE ARCHITECT ---
        if st.session_state.protocol:
            st.divider()
            st.subheader("💬 Statistical Logic Defense")
            st.write("Does a specific section seem off? Challenge the Architect's reasoning here.")

            # We use a unique key for the chat input to prevent refreshing the whole page unnecessarily
            with st.expander("Ask the Architect 'Why?'", expanded=True):
                col1, col2 = st.columns([4, 1])
                user_query = col1.text_input("Enter your question:", placeholder="e.g., Why use a Normal distribution for a discrete scale?", key="logic_query")
                
                if col2.button("Ask") and user_query:
                    with st.spinner("Analyzing statistical trade-offs..."):
                        defense = agent.explain_logic(
                            protocol_text=st.session_state.protocol,
                            query=user_query,
                            interview_answers=st.session_state.answers
                        )
                        st.info(f"**Architect's Reasoning:**\n\n{defense}")

    # --- PATH B: UPLOAD DRAFT ---
    else:
        st.subheader("📁 Upload Your Protocol Draft")
        uploaded_file = st.file_uploader("Choose a .txt file", type=['txt'])
        if uploaded_file is not None:
            # We read the file and save it directly to session state
            st.session_state.protocol = uploaded_file.getvalue().decode("utf-8")
            st.success("Draft Uploaded Successfully! Proceed to Phase 2 for the Audit.")
        # DISPLAY THE RESULTING PROTOCOL
        if st.session_state.protocol:
            st.divider()
            st.success("SAP Protocol Drafted Successfully!")
            st.markdown(st.session_state.protocol)


# --- TAB 2: TIERED AUDIT (The Hard Gate) ---
with tab2:
    st.header("Phase 2: Adversarial Regulatory Audit")
    if st.session_state.protocol:
        if st.button("Run Tiered Scan (Law, FDA Precedents, Academic Papers)"):
            with st.spinner("Dispatching Parallel Letter Auditors..."):
                # Agent 4 uses Statutory Truth for Section 1/2
                st.session_state.audit_report = agent.audit_protocol(st.session_state.protocol)
        
        if st.session_state.audit_report:
            st.info("Agent 4 Synthesis Complete. View Internal Monologue below.")
            st.markdown(st.session_state.audit_report)
    else:
        st.warning("Please draft a protocol in Tab 1 first.")

# --- TAB 3: OPTIMIZE (The Brain) ---
with tab3:
    st.header("Phase 3: Final Optimization Loop")
    if st.session_state.audit_report:
        directives = st.text_input("Senior Reviewer Directives", "None")
        if st.button("Apply Optimization Iterations"):
            with st.spinner("Resolving audit risks via Optimizer Intelligence..."):
                # Agent 4 uses Statutory + Academic + Architecture context here
                st.session_state.final_protocol = agent.optimize_protocol(
                    st.session_state.protocol,
                    st.session_state.audit_report,
                    user_directives=directives
                )

        if st.session_state.final_protocol:
            st.success("Optimization Passed Security Checks")
            # --- OPTIMIZED DOWNLOAD BUTTONS ---
            st.divider()
            st.subheader("📥 Export Final Draft")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Download SAP Text (.txt)", st.session_state.final_protocol, "protocol.txt")
            with col2:
                st.download_button("Download SAP Markdown (.md)", st.session_state.final_protocol, "protocol.md")
    else:
        st.info("Run Audit in Phase 2 first.")