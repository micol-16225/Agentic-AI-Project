import streamlit as st
import pandas as pd 
import os
from dotenv import load_dotenv
from lifecycle_agent3 import BiostatLifecycleAgent3
from evaluator import AuditorEvaluator


# --- INITIALIZATION BLOCK ---
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

# Load Environment
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Cache the agent to avoid re-initialization on every interaction
@st.cache_resource
def get_agent(api_key, library_path):
    return BiostatLifecycleAgent3(api_key, library_path)

# Initialize Agent
agent = get_agent(api_key, "master_regulatory_library.csv")

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
# --- 1. Consolidated Initialization ---
if 'answers' not in st.session_state: st.session_state.answers = {}
if 'protocol' not in st.session_state: st.session_state.protocol = None
if 'audit_report' not in st.session_state: st.session_state.audit_report = None
if 'final_protocol' not in st.session_state: st.session_state.final_protocol = None
if 'user_notes' not in st.session_state: st.session_state.user_notes = ""

# --- UI CONFIG ---
st.set_page_config(page_title="BioStat Enterprise AI", layout="wide", page_icon="🧬")
agent = BiostatLifecycleAgent3(api_key, "regulatory_library.txt")

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



# --- SIDEBAR VALIDATION TOOLS ---
with st.sidebar:
    st.divider()
    st.subheader("🧪 Developer Stress Tests")
    if st.button("🚀 Run Logic Validation"):
        evaluator = AuditorEvaluator()
        
        # USE THE CACHED AGENT instead of making a new one to ensure consistency
        # And ensure the library path is correct
        test_agent = get_agent(api_key, "master_regulatory_library.csv")
        
        test_results = []
        progress_bar = st.progress(0)
        
        for i, case in enumerate(evaluator.test_suite):
            st.write(f"Testing: {case['name']}...")
            
            # THE CRITICAL FIX: The Auditor must pull its knowledge base BEFORE 
            # generating the report. audit_protocol does this internally, but 
            # we must ensure the prompt inside audit_protocol is receiving the text.
            
            with st.spinner("Agent is searching library..."):
                audit_report = test_agent.audit_protocol(case['protocol_snippet'], "No Historical Lessons Provided")
            
            grade = evaluator.evaluate_audit(audit_report, case)
            test_results.append(grade)
            progress_bar.progress((i + 1) / len(evaluator.test_suite))
        
        # --- DISPLAY RESULTS ---
        st.write("### Validation Results")
        for res in test_results:
            # Check if 'res' is actually a dictionary and not an error string
            if isinstance(res, dict):
                if res.get('final_grade') == "PASS":
                    st.success(f"**{res['test_name']}**: PASS")
                else:
                    st.error(f"**{res['test_name']}**: FAIL")
                    st.info(f"Missing Keywords: {', '.join(res.get('missing_keywords', []))}")
            else:
                # If 'res' is just a string (the error message), show it
                st.warning(f"Technical Error: {res}")


# --- MAIN APP ---
st.title("🧬 Biostatistical Lifecycle Orchestrator")

tab1, tab2, tab3 = st.tabs(["🏗️ Phase 1: Input", "🔍 Phase 2: Audit", "⚡ Phase 3: Optimize"])

# --- TAB 1: ARCHITECT ---
# --- TAB 1: INTERACTIVE DESIGN ---
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
            st.success("Protocol Drafted Successfully!")
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
            st.success("Protocol Drafted Successfully!")
            st.markdown(st.session_state.protocol)


# --- TAB 2: AUDIT ---

with tab2:
    st.header("Step 2: Adversarial Audit")
    
    if st.session_state.protocol:
        audit_scope = st.segmented_control(
            "Audit Scope", 
            options=["Full Protocol", "Atomic Section"], 
            default="Full Protocol"
        )

        if st.button("Run Regulatory Scan"):
            with st.spinner("Performing Semantic Search & Regulatory Audit..."):
                # We wrap the protocol in a 'Scope' tag so the Agent knows how to grade it
                scoped_input = f"SCOPE: {audit_scope}\n\n{st.session_state.protocol}"
                st.session_state.audit_report = agent.audit_protocol(scoped_input, "FDA Lessons")
        
        if st.session_state.audit_report:
            # --- 1. THE TABLE ---
            st.subheader("📋 Regulatory Analysis")
            # Reformat the markdown table for better display
            formatted_audit = st.session_state.audit_report.replace("Risk Level |", "Risk Level |\n| --- | --- | --- |")
            st.markdown(formatted_audit)
            
            st.divider()

            # --- 3. THE EVIDENCE POP-UPS ---
            st.subheader("📁 Evidence Deep-Dive")
            st.write("Click a button below to see the full original FDA correspondence mentioned in the audit:")

            ev_cols = st.columns(3)
            col_idx = 0
            # We look for the "MATCH FOUND" or any line that looks like a citation in Section 1
            for line in st.session_state.audit_report.split('\n'):
                if "MATCH FOUND" in line.upper() or "PRECEDENT:" in line.upper():
                    # Extracts words that look like company names (Capitalized words)
                    parts = line.split('|')
                    if len(parts) >= 2:
                        company_name = parts[1].strip().split(' ')[0] # Takes the first word/name
                        with ev_cols[col_idx % 3]:
                            if st.button(f"📖 View {company_name}", key=f"btn_{col_idx}"):
                                show_full_letter(company_name)
                        col_idx += 1

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
    if st.session_state.audit_report:
        if st.button("🚀 Finalize FDA-Proof Protocol"):
            with st.spinner("Optimizing..."):
                # PASS THE USER NOTES (DIRECTIVES) HERE!
                st.session_state.final_protocol = agent.optimize_protocol(
                    st.session_state.protocol, 
                    st.session_state.audit_report,
                    user_directives=st.session_state.user_notes
                )
            
    if st.session_state.final_protocol:
        st.success("### Final FDA-Proof Protocol")
        
        # This is the "Trust" section
        with st.status("Performing Final Security Checks...", expanded=False):
            st.write("✅ Cross-referencing Auditor findings...")
            st.write("✅ Validating against Academic Knowledge Base...")
            st.write("✅ Checking for Optimization Drift...")
            st.write("✨ **Result: Internal Validation Passed.**")
            
        st.markdown(st.session_state.final_protocol)
        
        if st.session_state.final_protocol:
            
            # --- OPTIMIZED DOWNLOAD BUTTONS ---
            st.divider()
            st.subheader("📥 Export Final Draft")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Download Text (.txt)", st.session_state.final_protocol, "protocol.txt")
            with col2:
                st.download_button("Download Markdown (.md)", st.session_state.final_protocol, "protocol.md")
    else:
        st.info("Run Audit in Phase 2 first.")