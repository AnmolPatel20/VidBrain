"""
VidBrain — YouTube Video Q&A Chatbot
=====================================
Paste a YouTube URL, ask questions, and get answers grounded in the video transcript.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from utils.transcript_loader import (
    extract_video_id,
    fetch_transcript,
    clean_transcript,
    clean_whisper_segments,
)
from utils.rag_pipeline import (
    split_transcript,
    create_embeddings,
    build_vectorstore,
    get_retriever,
    create_llm,
    build_qa_chain,
    ask_with_history,
    ask_question_simple,
)
from utils.helpers import (
    format_source_reference,
    get_video_thumbnail,
    format_duration,
    format_views,
)
from utils.audio_downloader import (
    get_video_metadata,
    download_audio,
    split_audio_if_needed,
    cleanup_audio,
)
from utils.transcriber import (
    get_groq_client,
    groq_transcribe_audio,
    clean_whisper_transcript,
)
from utils.cache_manager import CacheManager

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="VidBrain — YouTube Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Premium Teal/Cyan Dark Theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ---------- Google Fonts ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, div.stMarkdown, div.stButton button {
    font-family: 'Inter', sans-serif;
}

/* ---------- Animated Background ---------- */
.stApp {
    background: #0a0f1a;
    background-image:
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(20, 184, 166, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 85% 70%, rgba(6, 182, 212, 0.07) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 50% 90%, rgba(34, 211, 238, 0.05) 0%, transparent 50%);
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 0;
    background-image:
        radial-gradient(circle 320px at 15% 25%, rgba(20, 184, 166, 0.09) 0%, transparent 100%),
        radial-gradient(circle 250px at 80% 15%, rgba(6, 182, 212, 0.07) 0%, transparent 100%),
        radial-gradient(circle 300px at 70% 75%, rgba(34, 211, 238, 0.06) 0%, transparent 100%),
        radial-gradient(circle 200px at 30% 80%, rgba(20, 184, 166, 0.05) 0%, transparent 100%);
    animation: bg-drift 20s ease-in-out infinite alternate;
}

@keyframes bg-drift {
    0%   { transform: translate(0, 0) scale(1); }
    25%  { transform: translate(15px, -10px) scale(1.02); }
    50%  { transform: translate(-10px, 15px) scale(0.98); }
    75%  { transform: translate(10px, 10px) scale(1.01); }
    100% { transform: translate(-15px, -5px) scale(1); }
}

/* Subtle dot grid overlay */
.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 0;
    background-image: radial-gradient(circle 1px at center, rgba(148, 163, 184, 0.06) 0%, transparent 100%);
    background-size: 28px 28px;
}

/* Ensure main content sits above the background */
.stApp > * {
    position: relative;
    z-index: 1;
}

/* ---------- Root Palette ---------- */
:root {
    --bg-primary: #0a0f1a;
    --bg-secondary: #111827;
    --bg-card: #131c2e;
    --border-subtle: #1e2d45;
    --border-glow: #0d9488;
    --accent-primary: #14b8a6;
    --accent-secondary: #06b6d4;
    --accent-tertiary: #22d3ee;
    --accent-warm: #f59e0b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --gradient-brand: linear-gradient(135deg, #14b8a6 0%, #06b6d4 40%, #22d3ee 100%);
    --gradient-card: linear-gradient(145deg, #131c2e 0%, #0f172a 100%);
    --gradient-sidebar: linear-gradient(180deg, #0a0f1a 0%, #0f172a 40%, #111827 100%);
    --shadow-glow: 0 0 20px rgba(20, 184, 166, 0.15);
    --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.3);
}

/* ---------- Animated Gradient Header ---------- */
@keyframes gradient-shift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes float-up {
    0%   { transform: translateY(0); opacity: 0.6; }
    50%  { opacity: 1; }
    100% { transform: translateY(-8px); opacity: 0.6; }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 8px rgba(20, 184, 166, 0.2); }
    50%      { box-shadow: 0 0 20px rgba(20, 184, 166, 0.45); }
}

@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}

.main-header {
    background: linear-gradient(135deg, #14b8a6, #06b6d4, #22d3ee, #0d9488, #14b8a6);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-shift 6s ease infinite;
    font-size: 2.6rem;
    font-weight: 800;
    margin-bottom: 0;
    line-height: 1.15;
    letter-spacing: -0.03em;
}
.main-header .brain-icon {
    -webkit-text-fill-color: initial;
    animation: float-up 2.5s ease-in-out infinite;
    display: inline-block;
    font-size: 2.2rem;
    margin-right: 0.35rem;
}

.sub-header {
    color: var(--text-secondary);
    font-size: 1.05rem;
    font-weight: 400;
    margin-top: 0.2rem;
    margin-bottom: 1.5rem;
    letter-spacing: 0.01em;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--gradient-sidebar);
    border-right: 1px solid var(--border-subtle);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary);
}
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.3rem;
}
.sidebar-brand .brand-icon {
    font-size: 1.6rem;
    animation: float-up 3s ease-in-out infinite;
}
.sidebar-brand .brand-text {
    font-size: 1.4rem;
    font-weight: 700;
    background: var(--gradient-brand);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sidebar-tagline {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-bottom: 1rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ---------- Video card ---------- */
.video-card {
    background: var(--gradient-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 1rem;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.video-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow);
}
.video-card img {
    border-radius: 10px;
    width: 100%;
}

/* ---------- Source block ---------- */
.source-block {
    background: var(--bg-card);
    border-left: 3px solid var(--accent-primary);
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    color: #cbd5e1;
    transition: border-color 0.2s ease;
}
.source-block:hover {
    border-left-color: var(--accent-tertiary);
}

/* ---------- Stat pill ---------- */
.stat-pill {
    background: var(--gradient-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 0.65rem 1rem;
    text-align: center;
    transition: border-color 0.3s ease, transform 0.2s ease;
}
.stat-pill:hover {
    border-color: var(--border-glow);
    transform: translateY(-2px);
}
.stat-pill .label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}
.stat-pill .value {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.25s ease;
    border: 1px solid var(--border-subtle) !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(20, 184, 166, 0.25);
    border-color: var(--accent-primary) !important;
}

/* ---------- Chat messages ---------- */
.stChatMessage {
    border-radius: 14px;
}

/* ---------- Welcome Feature Cards ---------- */
.feature-card {
    background: var(--gradient-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    height: 100%;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gradient-brand);
    opacity: 0;
    transition: opacity 0.35s ease;
}
.feature-card:hover {
    border-color: var(--border-glow);
    transform: translateY(-4px);
    box-shadow: var(--shadow-glow);
}
.feature-card:hover::before {
    opacity: 1;
}
.feature-card .card-icon {
    font-size: 2.2rem;
    margin-bottom: 0.3rem;
    display: inline-block;
}
.feature-card h4 {
    color: var(--text-primary);
    margin: 0.5rem 0;
    font-weight: 600;
    font-size: 1rem;
}
.feature-card p {
    color: var(--text-secondary);
    font-size: 0.88rem;
    line-height: 1.5;
}

/* ---------- Welcome Hero ---------- */
.welcome-hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.welcome-hero .hero-icon {
    font-size: 4.5rem;
    margin-bottom: 0.6rem;
    display: inline-block;
    animation: float-up 3s ease-in-out infinite;
}
.welcome-hero h3 {
    color: var(--text-primary);
    font-weight: 700;
    font-size: 1.45rem;
    margin-bottom: 0.6rem;
}
.welcome-hero p {
    color: var(--text-secondary);
    max-width: 520px;
    margin: auto;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* ---------- Badge ---------- */
.powered-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(20, 184, 166, 0.08);
    border: 1px solid rgba(20, 184, 166, 0.2);
    border-radius: 100px;
    padding: 0.35rem 0.9rem;
    font-size: 0.75rem;
    color: var(--accent-primary);
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-top: 1rem;
    animation: pulse-glow 3s ease-in-out infinite;
}

/* ---------- Quick Actions Section ---------- */
.quick-actions-title {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--text-primary);
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
}
.quick-actions-title .bolt {
    color: var(--accent-warm);
}

/* ---------- Divider accent ---------- */
hr {
    border-color: var(--border-subtle) !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------

DEFAULTS = {
    "chat_history": [],       # list of {"role": ..., "content": ...}
    "vectorstore": None,
    "retriever": None,
    "qa_chain": None,
    "llm": None,
    "transcript_text": None,
    "raw_transcript": None,
    "video_id": None,
    "chunks": None,
    "video_loaded": False,
    "transcript_source": None,
    "processing_method": None,
    "video_metadata": None,
    "speaker_count": None,
    "cache_used": False,
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<span class="brand-icon">🧠</span>'
        '<span class="brand-text">VidBrain</span>'
        '</div>'
        '<div class="sidebar-tagline">AI-Powered Video Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # --- API Key ---
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        api_key = groq_key
    else:
        api_key = st.text_input(
            "🔑 Groq API Key",
            value="",
            type="password",
            help="Get a free key at https://console.groq.com/keys",
        )
        if api_key:
            st.success("API key loaded", icon="✅")
        else:
            st.warning("Enter your Groq API key to continue", icon="⚠️")
        st.markdown("---")

    # --- Video Info (shown after loading) ---
    if st.session_state.video_loaded and st.session_state.video_id:
        vid = st.session_state.video_id
        meta = st.session_state.video_metadata or {}
        
        st.markdown("### 📹 Video Info")

        st.markdown(
            f'<div class="video-card">'
            f'<img src="{get_video_thumbnail(vid)}" alt="thumbnail" />'
            f'<div style="margin-top: 0.8rem;">'
            f'<div style="color: var(--text-primary); font-weight: 600; font-size: 0.95rem; line-height: 1.3; margin-bottom: 0.3rem;">{meta.get("title", "Unknown Title")}</div>'
            f'<div style="color: var(--text-secondary); font-size: 0.85rem;">{meta.get("channel", "Unknown Channel")}</div>'
            f'</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Stats Grid 1
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="stat-pill">'
                f'<div class="label">Duration</div>'
                f'<div class="value">{format_duration(meta.get("duration", 0))}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="stat-pill">'
                f'<div class="label">Views</div>'
                f'<div class="value">{format_views(meta.get("view_count", 0))}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            
        st.write("") # spacer
        
        # Stats Grid 2
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(
                f'<div class="stat-pill">'
                f'<div class="label">Source</div>'
                f'<div style="font-size: 1rem; font-weight: 600; color: var(--text-primary); margin-top: 0.3rem;">{st.session_state.transcript_source or "N/A"}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="stat-pill">'
                f'<div class="label">Chunks</div>'
                f'<div class="value">{len(st.session_state.chunks) if st.session_state.chunks else 0}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

    # --- Actions ---
    if st.session_state.video_loaded:
        st.markdown(
            '<div class="quick-actions-title">'
            '<span class="bolt">⚡</span> Quick Actions'
            '</div>',
            unsafe_allow_html=True,
        )

        if st.button("📝 Summarize Video", use_container_width=True):
            with st.spinner("Generating summary…"):
                result = ask_question_simple(
                    st.session_state.llm,
                    st.session_state.retriever,
                    "Provide a comprehensive summary of this entire video. "
                    "Cover all major topics discussed.",
                )
                st.session_state.chat_history.append(
                    {"role": "user", "content": "📝 Summarize this video"}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result["answer"]}
                )
                st.rerun()

        if st.button("🎯 Key Takeaways", use_container_width=True):
            with st.spinner("Extracting takeaways…"):
                result = ask_question_simple(
                    st.session_state.llm,
                    st.session_state.retriever,
                    "List the key takeaways and main points from this video as bullet points.",
                )
                st.session_state.chat_history.append(
                    {"role": "user", "content": "🎯 Key takeaways from this video"}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result["answer"]}
                )
                st.rerun()

        if st.button("📚 Important Topics", use_container_width=True):
            with st.spinner("Identifying topics…"):
                result = ask_question_simple(
                    st.session_state.llm,
                    st.session_state.retriever,
                    "What are the important topics and themes discussed in this video? "
                    "List each topic with a brief description.",
                )
                st.session_state.chat_history.append(
                    {"role": "user", "content": "📚 Important topics in this video"}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result["answer"]}
                )
                st.rerun()



    # --- Clear / Reset ---
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Reset Everything", use_container_width=True):
        for key in DEFAULTS:
            st.session_state[key] = DEFAULTS[key]
        st.rerun()

    # --- Footer credit ---
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; padding: 0.5rem 0 0.2rem;">'
        '<span style="color: #64748b; font-size: 0.78rem; font-weight: 400;">'
        '<span style="color: #14b8a6;">~</span> Made by <span style="font-weight: 600; color: #94a3b8;">Anmol</span>'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Area — Header
# ---------------------------------------------------------------------------

st.markdown(
    '<h1 class="main-header">'
    '<span class="brain-icon">🧠</span> VidBrain'
    '</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Paste a YouTube URL and ask anything about the video</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# URL Input & Video Loading
# ---------------------------------------------------------------------------

col_url, col_btn = st.columns([4, 1])

with col_url:
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
    

with col_btn:
    load_clicked = st.button("🚀 Load Video", use_container_width=True)


# --- Load pipeline ---
if load_clicked:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar first.")
        st.stop()

    if not youtube_url.strip():
        st.error("Please paste a YouTube URL above.")
        st.stop()

    video_id = extract_video_id(youtube_url.strip())
    if not video_id:
        st.error("Invalid YouTube URL. Please check and try again.")
        st.stop()

    # Skip reprocessing if the same video is already loaded
    if st.session_state.video_id == video_id and st.session_state.video_loaded:
        st.info("This video is already loaded! Ask away 👇")
    else:
        try:
            with st.status("🔄 Processing video…", expanded=True) as status:
                st.write("📥 Checking cache…")
                
                # Check Metadata Cache
                video_metadata = CacheManager.get_metadata(video_id)
                if not video_metadata:
                    st.write("📥 Extracting metadata…")
                    video_metadata = get_video_metadata(video_id)
                    CacheManager.save_metadata(video_id, video_metadata)
                st.session_state.video_metadata = video_metadata
                
                # Check Transcript Cache
                cached_transcript_data = CacheManager.get_transcript(video_id)
                raw_transcript = None
                transcript_text = None
                
                if cached_transcript_data:
                    st.write("✅ Transcript loaded from cache!")
                    raw_transcript = cached_transcript_data.get("raw", [])
                    st.session_state.transcript_source = cached_transcript_data.get("source", "Cache")
                    st.session_state.processing_method = "Cache"
                    st.session_state.cache_used = True
                    transcript_text = clean_transcript(raw_transcript)
                else:
                    # 1. Fetch transcript via API
                    st.write("📥 Fetching transcript…")
                    raw_transcript = fetch_transcript(video_id)
                    
                    if raw_transcript:
                        st.write("✅ Found YouTube captions!")
                        st.session_state.transcript_source = "YouTube Captions"
                        st.session_state.processing_method = "API"
                        st.write("🧹 Cleaning transcript…")
                        transcript_text = clean_transcript(raw_transcript)
                        CacheManager.save_transcript(video_id, raw_transcript, "YouTube Captions")
                    else:
                        st.write("⚠️ No captions found. Using Groq Whisper API...")
                        st.write("🎧 Downloading audio (yt-dlp)…")
                        audio_path = download_audio(video_id)
                        if not audio_path:
                            raise ValueError("Failed to download audio for transcription.")
                            
                        # Try Groq Whisper API
                        st.write("🚀 Initiating Groq Whisper pipeline…")
                        groq_client = get_groq_client(api_key)
                        whisper_segments = []
                        
                        if groq_client:
                            # Chunk audio if it exceeds Groq's 25MB limit
                            st.write("✂️ Checking audio file size and splitting if needed...")
                            audio_chunks = split_audio_if_needed(audio_path, max_size_mb=24)
                            
                            whisper_segments = groq_transcribe_audio(audio_chunks, groq_client)
                            if whisper_segments:
                                st.session_state.transcript_source = "Groq Whisper API"
                                st.session_state.processing_method = "Cloud Compute"
                                
                            cleanup_audio(audio_chunks)
                            if audio_path not in audio_chunks:
                                cleanup_audio([audio_path])
                        
                        if not whisper_segments:
                            raise ValueError("Groq transcription failed. Ensure your API key is valid.")
                        
                        raw_transcript = clean_whisper_transcript(whisper_segments)
                        st.write("🧹 Formatting transcript…")
                        transcript_text = clean_transcript(raw_transcript)
                        
                        CacheManager.save_transcript(video_id, raw_transcript, st.session_state.transcript_source)
                
                st.session_state.raw_transcript = raw_transcript
                st.session_state.transcript_text = transcript_text

                # 3. Chunks
                chunks = CacheManager.get_chunks(video_id)
                if chunks:
                    st.write("✅ Chunks loaded from cache!")
                else:
                    st.write("✂️ Splitting into chunks…")
                    chunks = split_transcript(transcript_text)
                    CacheManager.save_chunks(video_id, chunks)
                st.session_state.chunks = chunks

                # 4. Embeddings & vector store
                st.write("🧠 Loading embedding model…")
                embeddings = create_embeddings()
                
                vectorstore = CacheManager.get_vectorstore(video_id, embeddings)
                if vectorstore:
                    st.write("⚡ Loading vectorstore from cache…")
                else:
                    vectorstore = build_vectorstore(chunks, embeddings)
                    CacheManager.save_vectorstore(video_id, vectorstore)
                    
                st.session_state.vectorstore = vectorstore

                # 5. Retriever
                st.write("🔍 Configuring retriever…")
                first_chunk = chunks[0] if chunks else None
                retriever = get_retriever(vectorstore, first_chunk)
                st.session_state.retriever = retriever

                # 6. LLM + chain
                st.write("⚡ Setting up LLM…")
                llm = create_llm(api_key)
                st.session_state.llm = llm
                chain = build_qa_chain(llm, retriever)
                st.session_state.qa_chain = chain

                # Done
                st.session_state.video_id = video_id
                st.session_state.video_loaded = True
                st.session_state.chat_history = []  # fresh chat per video
                status.update(label="✅ Video loaded!", state="complete", expanded=False)

            st.rerun()  # refresh sidebar

        except ValueError as e:
            st.error(f"⚠️ {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            st.stop()


# ---------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------

if st.session_state.video_loaded:
    st.markdown("---")

    # Display chat history
    for msg in st.session_state.chat_history:
        avatar = "🧑" if msg["role"] == "user" else "🧠"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            # Show sources if attached
            if msg.get("sources"):
                with st.expander("📚 View Transcript Sources", expanded=False):
                    for i, src in enumerate(msg["sources"]):
                        st.markdown(
                            f'<div class="source-block">{format_source_reference(src, i)}</div>',
                            unsafe_allow_html=True,
                        )

    # Chat input
    user_question = st.chat_input("Ask anything about the video…")

    if user_question:
        if not user_question.strip():
            st.warning("Please enter a question.")
            st.stop()

        # Show user message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_question)

        # Generate answer
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Thinking…"):
                try:
                    result = ask_with_history(
                        st.session_state.qa_chain,
                        user_question,
                        st.session_state.chat_history,
                        st.session_state.llm,
                    )
                    answer = result["answer"]
                    sources = result.get("sources", [])

                    st.markdown(answer)

                    # Show sources
                    if sources:
                        with st.expander("📚 View Transcript Sources", expanded=False):
                            for i, src in enumerate(sources):
                                st.markdown(
                                    f'<div class="source-block">'
                                    f"{format_source_reference(src, i)}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    # Save to history
                    st.session_state.chat_history.append(
                        {"role": "user", "content": user_question}
                    )
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")

else:
    # --- Welcome state ---
    st.markdown("---")
    st.markdown(
        """
        <div class="welcome-hero">
            <div class="hero-icon">🧠</div>
            <h3>Paste a YouTube URL above to get started</h3>
            <p>
                VidBrain extracts the video transcript, builds a knowledge base,
                and lets you chat with any YouTube video using AI.
            </p>
            <div class="powered-badge">
                ⚡ Powered by Groq &amp; LangChain
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards
    cols = st.columns(3)
    features = [
        ("🔍", "Smart Q&A", "Ask any question and get accurate, transcript-grounded answers."),
        ("📝", "Video Summary", "Get a full summary, key takeaways, and important topics."),
        ("📚", "Source References", "Every answer includes the relevant transcript sections."),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="card-icon">{icon}</div>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
