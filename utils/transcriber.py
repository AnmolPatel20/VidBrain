import os
import streamlit as st
from typing import Optional
from groq import Groq

@st.cache_resource(show_spinner="Loading Groq Client...")
def get_groq_client(api_key: str) -> Optional[Groq]:
    if not api_key:
        return None
    # Use a generous 10-minute (600 seconds) timeout to allow uploading
    # large audio files up to 24MB even on slower network connections.
    return Groq(api_key=api_key, timeout=600.0)

def groq_transcribe_audio(audio_paths: list[str], client: Groq) -> list[dict]:
    """
    Transcribe one or more audio files using Groq Whisper API (whisper-large-v3-turbo).
    Returns list of segment dicts: {"text", "start", "end"}
    """
    all_segments = []
    time_offset = 0.0
    
    total_chunks = len(audio_paths)
    for idx, audio_path in enumerate(audio_paths):
        if total_chunks > 1:
            st.write(f"  🎙️ Transcribing chunk {idx + 1}/{total_chunks}...")
        else:
            st.write("  🎙️ Uploading and transcribing audio...")
        try:
            with open(audio_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                )
                
            # Parse segments
            segments = getattr(transcription, 'segments', None)
            if segments is None and isinstance(transcription, dict):
                segments = transcription.get("segments")
                
            last_end_time = 0.0
            
            if segments:
                for segment in segments:
                    text = segment.get("text", "") if isinstance(segment, dict) else getattr(segment, "text", "")
                    start = segment.get("start", 0) if isinstance(segment, dict) else getattr(segment, "start", 0)
                    end = segment.get("end", 0) if isinstance(segment, dict) else getattr(segment, "end", 0)
                    
                    start_float = float(start)
                    end_float = float(end)
                    
                    all_segments.append({
                        "text": text.strip(),
                        "start": start_float + time_offset,
                        "end": end_float + time_offset
                    })
                    
                    if end_float > last_end_time:
                        last_end_time = end_float
                        
            time_offset += last_end_time
            
        except Exception as e:
            print(f"Groq API Transcription failed for {audio_path}: {e}")
            
    return all_segments

def clean_whisper_transcript(segments: list[dict]) -> list[dict]:
    """
    Clean Whisper transcription segments.
    Removes fillers, deduplicates text, merges tiny segments.
    """
    if not segments:
        return []
        
    fillers = {"um", "uh", "you know", "like"}
    
    cleaned_segments = []
    prev_text = ""
    
    for seg in segments:
        text = " ".join(seg["text"].split())
        if text.lower().strip(".,!?") in fillers:
            continue
            
        if text.lower() == prev_text.lower():
            if cleaned_segments:
                cleaned_segments[-1]["end"] = seg["end"]
            continue
            
        if text:
            cleaned_segments.append({
                "text": text,
                "start": seg["start"],
                "end": seg["end"]
            })
            prev_text = text
            
    merged = []
    for seg in cleaned_segments:
        if not merged:
            merged.append(seg)
            continue
            
        prev_seg = merged[-1]
        duration = seg["end"] - seg["start"]
        gap = seg["start"] - prev_seg["end"]
        
        if duration < 1.5 and gap < 1.0:
            prev_seg["text"] += f" {seg['text']}"
            prev_seg["end"] = seg["end"]
        else:
            merged.append(seg)
            
    return merged
