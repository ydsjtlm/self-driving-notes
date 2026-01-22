import streamlit as st
import paper_filter
import paper_utils
import os
import json

# Set page configuration
st.set_page_config(
    page_title="AI Paper Recommender",
    page_icon="📄",
    layout="wide"
)

# Initialize session state for recommendations
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = []

st.title("📄 AI Paper Recommender")
st.markdown("""
This tool helps you discover relevant papers by analyzing a webpage and filtering them 
based on your personal reading history (markdown notes).
""")

# Sidebar for Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Settings
    st.subheader("LLM Configuration")
    api_key = st.text_input(
        "API Key", 
        value=os.environ.get("LLM_API_KEY", "sk-863ce5e99f0c41059e1c1bbcb69bd340"), 
        type="password",
        help="Your LLM Provider API Key"
    )
    base_url = st.text_input(
        "Base URL", 
        value=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
        help="API Base URL (e.g., https://api.deepseek.com)"
    )
    model = st.text_input(
        "Model Name", 
        value="deepseek-chat",
        help="Model to use for filtering (e.g., deepseek-chat, gpt-4)"
    )
    
    st.divider()
    
    # Application Settings
    st.subheader("Preferences")
    notes_dir = st.text_input(
        "Notes Directory", 
        value=".", 
        help="Root directory where your markdown notes are stored"
    )
    top_k = st.slider(
        "Max Recommendations", 
        min_value=1, 
        max_value=50, 
        value=10,
        help="Number of top papers to show"
    )

# Main Content Area
col1, col2 = st.columns([3, 1])
with col1:
    url = st.text_input(
        "Target Website URL", 
        placeholder="https://arxiv.org/list/cs/recent",
        help="Enter the URL of the website listing papers (e.g., ArXiv, Conference page, Blog)"
    )
with col2:
    # Add some spacing to align button
    st.write("") 
    st.write("")
    analyze_btn = st.button("🚀 Analyze & Recommend", type="primary", use_container_width=True)

if analyze_btn:
    if not api_key:
        st.error("❌ Please provide an API Key in the sidebar.")
        st.stop()
        
    if not url:
        st.warning("⚠️ Please enter a URL to analyze.")
        st.stop()

    # Step 1: Load History
    with st.spinner("Processing..."):
        st.info("📚 Reading your history notes...")
        try:
            history_titles = paper_filter.get_all_history_titles(notes_dir)
            st.success(f"✅ Found {len(history_titles)} papers in your reading history.")
        except Exception as e:
            st.error(f"Error reading history: {e}")
            st.stop()

        # Step 2: Fetch and Extract Candidates
        st.info(f"🌐 Fetching content from {url}...")
        html = paper_filter.fetch_url_content(url)
        if not html:
            st.error("Failed to fetch content from the URL. Please check the link.")
            st.stop()
            
        candidates = paper_filter.extract_papers_from_html(html)
        st.success(f"✅ Found {len(candidates)} candidate papers on the page.")
        
        if not candidates:
            st.warning("No papers found on the page. Try a different URL.")
            st.stop()

        # Step 3: LLM Filtering
        st.info("🤖 Asking LLM to filter and rank papers...")
        recommendations = paper_filter.call_llm_filter(
            history_titles, candidates, api_key, base_url, model, top_k=top_k
        )
        st.session_state['recommendations'] = recommendations

# Display Results
if st.session_state['recommendations']:
    st.subheader(f"Top {len(st.session_state['recommendations'])} Recommendations")
    
    for i, rec in enumerate(st.session_state['recommendations']):
        score = rec.get('score', 0)
        
        # Determine color based on score
        if score >= 80:
            border_color = "green"
        elif score >= 50:
            border_color = "orange"
        else:
            border_color = "grey"
            
        with st.container():
            st.markdown(f"""
            <div style="border-left: 5px solid {border_color}; padding-left: 10px; margin-bottom: 10px;">
                <h3>{i+1}. <a href="{rec.get('link', '#')}" target="_blank">{rec.get('title', 'Unknown Title')}</a></h3>
                <p><strong>Relevance Score:</strong> {score}/100</p>
                <p><strong>Why:</strong> {rec.get('reason', 'No reason provided')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Summary Section
            summary_key = f"summary_{i}"
            col_a, col_b = st.columns([1, 5])
            with col_a:
                if st.button(f"📝 Summarize", key=f"btn_sum_{i}"):
                    with st.spinner("Downloading and analyzing paper..."):
                        pdf_path = paper_utils.download_pdf(rec.get('link'))
                        if pdf_path:
                            text = paper_utils.extract_text_from_pdf(pdf_path)
                            summary = paper_utils.summarize_paper(text, api_key, base_url, model)
                            st.session_state[summary_key] = summary
                        else:
                            st.error("Could not download PDF. Please check the link.")
            
            if summary_key in st.session_state:
                with st.expander("📄 Paper Summary", expanded=True):
                    st.markdown(st.session_state[summary_key])
            
            st.divider()
