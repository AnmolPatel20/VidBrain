# 🧠 VidBrain

A production-ready YouTube Video Q&A chatbot built entirely on **Groq APIs**.

VidBrain extracts video metadata, retrieves YouTube captions, or uses **Groq Whisper Large V3 Turbo** to transcribe videos that lack captions. It then builds a lightweight local FAISS vector store using **FastEmbed** and answers your questions using **Groq's Llama 3.3 70B** model.

## Features

- **Blazing Fast**: Powered exclusively by Groq's APIs. No heavy local models.
- **Smart Transcription Fallback**: Always uses YouTube Captions if available; seamlessly falls back to Whisper API if not.
- **Auto Audio Chunking**: Automatically compresses and chunks audio to bypass Whisper API file size limits.
- **Zero Redundant Processing**: Aggressively caches metadata, transcripts, chunks, and vector stores per video.
- **Conversational Context**: Chat history is preserved, and follow-up questions are automatically rewritten into standalone search queries.
- **Precise Citations**: Every answer provides the exact transcript segments used for grounding.
- **Quick Insights**: One-click buttons to generate Summaries, Key Takeaways, and Important Topics.

## Tech Stack

- **UI**: Streamlit
- **Transcription**: Groq API (`whisper-large-v3-turbo`)
- **LLM Engine**: Groq API (`llama-3.3-70b-versatile`)
- **Vector Database**: FAISS (Local)
- **Embeddings**: FastEmbed (Lightweight ONNX models)
- **RAG Framework**: LangChain
- **Audio Extraction**: `yt-dlp` and `pydub`

## Setup Instructions

### 1. Requirements

Ensure you have `ffmpeg` installed on your system.

**Mac:**
```bash
brew install ffmpeg
```
**Windows:**
Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add to PATH.

### 2. Clone and Install

```bash
git clone <your-repo-url>
cd VidBrain

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. API Keys

```bash
cp .env.example .env
```
Edit `.env` and add your Groq API key:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Run the Application

```bash
streamlit run app.py
```
