import streamlit as st
from utils import *
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Medical RAG - CBOW vs Skip-Gram",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Metric cards - Fixed spacing */
    .metric-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid;
        margin-bottom: 0.5rem;
        min-height: 80px;
    }
    
    .metric-card.cbow {
        border-left-color: #667eea;
    }
    
    .metric-card.skipgram {
        border-left-color: #10b981;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .metric-value.cbow {
        color: #667eea;
    }
    
    .metric-value.skipgram {
        color: #10b981;
    }
    
    /* Answer boxes - Clean design */
    .answer-box {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #334155;
    }
    
    .answer-box.cbow {
        border-top: 3px solid #667eea;
    }
    
    .answer-box.skipgram {
        border-top: 3px solid #10b981;
    }
    
    .answer-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .answer-title.cbow {
        color: #667eea;
    }
    
    .answer-title.skipgram {
        color: #10b981;
    }
    
    .answer-content {
        color: #e2e8f0;
        line-height: 1.7;
        font-size: 0.95rem;
    }
    
    /* Question box */
    .question-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #818cf8;
    }
    
    .question-box p {
        color: #e0e7ff;
        margin: 0;
        font-size: 1rem;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: #60a5fa;
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
    }
    
    .info-box p {
        color: #cbd5e1;
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Streamlit component overrides */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Expander fix */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #e2e8f0 !important;
        background: transparent !important;
    }
    
    div[data-testid="stExpander"] {
        background: #1e293b;
        border-radius: 8px;
        border: 1px solid #334155;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ================== SESSION STATE ==================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

# ================== LLM SETUP ==================
@st.cache_resource
def get_llm_client():
    """Initialize HuggingFace Inference Client."""
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        st.error("⚠️ HuggingFace API token not found. Please set HUGGINGFACEHUB_API_TOKEN in .env file")
        return None
    return InferenceClient(token=token)

def generate_answer(client, context, query):
    """Generate answer using HuggingFace Inference API with chat completion."""
    if client is None:
        return "Error: LLM client not initialized. Check your API token."
    
    try:
        # Use chat completion with proper message format
        messages = [
            {
                "role": "system",
                "content": "You are a helpful medical assistant. Answer questions based ONLY on the provided context. If the context doesn't contain enough information, say so clearly. Be concise and accurate."
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a clear, accurate answer based on the context above."
            }
        ]
        
        response = client.chat_completion(
            messages=messages,
            model="mistralai/Mistral-7B-Instruct-v0.3",
            max_tokens=400,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = str(e)
        # Try fallback model if Mistral fails
        try:
            response = client.chat_completion(
                messages=messages,
                model="HuggingFaceH4/zephyr-7b-beta",
                max_tokens=400,
                temperature=0.3
            )
            return response.choices[0].message.content
        except:
            return f"Error generating response: {error_msg}"

# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown("## 🧬 Medical RAG")
    st.markdown("---")
    
    st.markdown("### 📄 Upload Document")
    uploaded_pdf = st.file_uploader(
        "Upload medical PDF",
        type="pdf",
        help="Upload a PDF for Q&A"
    )
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Chunks to retrieve", 1, 5, 3)
    
    st.markdown("---")
    
    with st.expander("🔵 About CBOW"):
        st.markdown("""
        Predicts target word from context.
        - ✅ Faster training
        - ✅ Better for frequent words
        """)
    
    with st.expander("🟢 About Skip-Gram"):
        st.markdown("""
        Predicts context from target word.
        - ✅ Better for rare words
        - ✅ Captures nuances
        """)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.rerun()

# ================== MAIN CONTENT ==================
st.markdown("""
<div class="main-header">
    <h1>🧬 Medical RAG: CBOW vs Skip-Gram</h1>
    <p>Compare word embedding methods for medical document Q&A</p>
</div>
""", unsafe_allow_html=True)

# Initialize LLM client
llm_client = get_llm_client()

# Welcome message
if not st.session_state.pdf_processed:
    st.markdown("""
    <div class="info-box">
        <h4>👋 Welcome!</h4>
        <p>Upload a medical PDF in the sidebar, then ask questions to compare CBOW and Skip-Gram embeddings.</p>
    </div>
    """, unsafe_allow_html=True)

# ================== PDF PROCESSING ==================
if uploaded_pdf:
    if uploaded_pdf.name != st.session_state.pdf_name:
        with st.spinner("📄 Processing document..."):
            try:
                text = read_pdf(uploaded_pdf)
                chunks = chunk_text(text)
                
                st.session_state.chunks = chunks
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_pdf.name
                st.session_state.chat_history = []
                clear_cache()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.pdf_processed = False
    
    if st.session_state.pdf_processed:
        # Document info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Document", st.session_state.pdf_name[:15] + "..." if len(st.session_state.pdf_name) > 15 else st.session_state.pdf_name)
        with col2:
            st.metric("📝 Chunks", len(st.session_state.chunks))
        with col3:
            st.metric("💬 Questions", len(st.session_state.chat_history))
        
        st.markdown("---")
        
        # Question input
        with st.form(key="qa_form", clear_on_submit=True):
            query = st.text_input(
                "🔍 Ask a medical question:",
                placeholder="e.g., What is human reproduction?"
            )
            submitted = st.form_submit_button("🚀 Get Answers")
        
        if submitted and query:
            chunks = st.session_state.chunks
            
            with st.spinner("🔄 Retrieving and generating answers..."):
                # Retrieve chunks
                cbow_chunks, cbow_metrics = retrieve_chunks(query, chunks, "cbow", top_k=top_k)
                sg_chunks, sg_metrics = retrieve_chunks(query, chunks, "skipgram", top_k=top_k)
                
                # Create context
                cbow_context = "\n\n".join(cbow_chunks)
                sg_context = "\n\n".join(sg_chunks)
                
                # Generate answers
                cbow_answer = generate_answer(llm_client, cbow_context, query)
                sg_answer = generate_answer(llm_client, sg_context, query)
            
            # Store result
            st.session_state.chat_history.append({
                "query": query,
                "cbow": {"answer": cbow_answer, "chunks": cbow_chunks, "metrics": cbow_metrics},
                "skipgram": {"answer": sg_answer, "chunks": sg_chunks, "metrics": sg_metrics}
            })
            
            st.rerun()
        
        # ================== DISPLAY RESULTS ==================
        if st.session_state.chat_history:
            for i, entry in enumerate(reversed(st.session_state.chat_history)):
                idx = len(st.session_state.chat_history) - i
                
                # Question
                st.markdown(f"""
                <div class="question-box">
                    <p><strong>🙋 Q{idx}:</strong> {entry['query']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics row
                st.markdown("#### 📊 Performance Metrics")
                m1, m2, m3, m4 = st.columns(4)
                
                cbow_m = entry['cbow']['metrics']
                sg_m = entry['skipgram']['metrics']
                cbow_sim = np.mean(cbow_m['similarity_scores']) if cbow_m['similarity_scores'] else 0
                sg_sim = np.mean(sg_m['similarity_scores']) if sg_m['similarity_scores'] else 0
                
                with m1:
                    st.markdown(f"""
                    <div class="metric-card cbow">
                        <div class="metric-label">CBOW Time</div>
                        <div class="metric-value cbow">{cbow_m['retrieval_time']:.3f}s</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m2:
                    st.markdown(f"""
                    <div class="metric-card cbow">
                        <div class="metric-label">CBOW Similarity</div>
                        <div class="metric-value cbow">{cbow_sim:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m3:
                    st.markdown(f"""
                    <div class="metric-card skipgram">
                        <div class="metric-label">Skip-Gram Time</div>
                        <div class="metric-value skipgram">{sg_m['retrieval_time']:.3f}s</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m4:
                    st.markdown(f"""
                    <div class="metric-card skipgram">
                        <div class="metric-label">Skip-Gram Similarity</div>
                        <div class="metric-value skipgram">{sg_sim:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Answers row
                st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="answer-box cbow">
                        <div class="answer-title cbow">🔵 CBOW Answer</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(entry['cbow']['answer'])
                    
                    with st.expander("📌 CBOW Chunks"):
                        for j, chunk in enumerate(entry['cbow']['chunks']):
                            score = cbow_m['similarity_scores'][j] if j < len(cbow_m['similarity_scores']) else 0
                            st.caption(f"Chunk {j+1} | Similarity: {score:.3f}")
                            st.info(chunk[:400] + "..." if len(chunk) > 400 else chunk)
                
                with col2:
                    st.markdown("""
                    <div class="answer-box skipgram">
                        <div class="answer-title skipgram">🟢 Skip-Gram Answer</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(entry['skipgram']['answer'])
                    
                    with st.expander("📌 Skip-Gram Chunks"):
                        for j, chunk in enumerate(entry['skipgram']['chunks']):
                            score = sg_m['similarity_scores'][j] if j < len(sg_m['similarity_scores']) else 0
                            st.caption(f"Chunk {j+1} | Similarity: {score:.3f}")
                            st.success(chunk[:400] + "..." if len(chunk) > 400 else chunk)
                
                st.markdown("---")

else:
    # How it works
    st.markdown("## 🎯 How It Works")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### 1️⃣ Upload")
        st.markdown("Upload a medical PDF document")
    
    with c2:
        st.markdown("### 2️⃣ Ask")
        st.markdown("Type your medical question")
    
    with c3:
        st.markdown("### 3️⃣ Compare")
        st.markdown("See CBOW vs Skip-Gram results")

# Footer
st.markdown("---")
st.caption("🧬 Medical RAG - Comparing CBOW & Skip-Gram for Medical Text Analytics")
