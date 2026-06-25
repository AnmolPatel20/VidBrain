import re
import streamlit as st
from typing import Optional
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

@st.cache_data(show_spinner=False)
def fetch_transcript(video_id: str) -> Optional[list[dict]]:
    """
    Fetch the transcript for a YouTube video.
    Returns a list of dicts:  [{"text": ..., "start": ..., "duration": ...}, ...]
    Returns None if transcripts are disabled or not found.
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        return transcript.to_raw_data()
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        print(f"Failed to fetch transcript: {e}")
        return None

def clean_transcript(raw_data: list[dict]) -> str:
    """
    Convert raw transcript data into a clean, readable text block,
    embedding timestamps roughly every minute.
    """
    if not raw_data:
        return ""

    cleaned_text = []
    current_chunk = []
    last_timestamp = 0.0
    
    for i, item in enumerate(raw_data):
        start = float(item.get("start", 0))
        text = item.get("text", "").strip()
        
        current_chunk.append(text)
        
        if start - last_timestamp >= 60.0 or i == len(raw_data) - 1:
            minutes = int(start // 60)
            seconds = int(start % 60)
            time_str = f"[{minutes:02d}:{seconds:02d}]"
            
            paragraph = " ".join(current_chunk)
            paragraph = re.sub(r'\s+', ' ', paragraph).strip()
            
            cleaned_text.append(f"{time_str} {paragraph}\n")
            
            current_chunk = []
            last_timestamp = start

    return "\n".join(cleaned_text)

def clean_whisper_segments(segments: list[dict]) -> list[dict]:
    """Convert whisper segments into a format similar to youtube-transcript-api."""
    formatted = []
    for seg in segments:
        formatted.append({
            "text": seg["text"],
            "start": seg["start"],
            "duration": seg["end"] - seg["start"]
        })
    return formatted
