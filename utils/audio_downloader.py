import os
import yt_dlp
from typing import Optional
from pydub import AudioSegment

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")
AUDIO_DIR = os.path.join(TEMP_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

def get_video_metadata(video_id: str) -> dict:
    ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            metadata = {
                "title": info.get("title", "Unknown Title"),
                "channel": info.get("uploader", "Unknown Channel"),
                "duration": info.get("duration", 0),
                "view_count": info.get("view_count", 0),
            }
            return metadata
        except Exception:
            return {"title": "Unknown Title", "channel": "Unknown Channel", "duration": 0, "view_count": 0}

def download_audio(video_id: str) -> Optional[str]:
    output_template = os.path.join(AUDIO_DIR, f"{video_id}.%(ext)s")
    
    # Compress audio to 64kbps to save space and minimize Groq limit issues
    ydl_opts = {
        'format': 'worstaudio[ext=m4a]/worstaudio/worst',
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '64',
        }]
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
            file_path = os.path.join(AUDIO_DIR, f"{video_id}.mp3")
            if os.path.exists(file_path):
                return file_path
            for file in os.listdir(AUDIO_DIR):
                if file.startswith(video_id):
                    return os.path.join(AUDIO_DIR, file)
            return None
        except Exception as e:
            print(f"Error downloading audio: {e}")
            return None

def split_audio_if_needed(file_path: str, max_size_mb: int = 24) -> list[str]:
    """Split audio file if it exceeds the max size (default 24MB for Groq API)."""
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb <= max_size_mb:
        return [file_path]
    
    print(f"Audio size {file_size_mb:.2f}MB exceeds {max_size_mb}MB limit. Splitting...")
    try:
        audio = AudioSegment.from_file(file_path)
        # 30 minutes in milliseconds
        chunk_length_ms = 30 * 60 * 1000
        chunks = []
        for i, chunk_start in enumerate(range(0, len(audio), chunk_length_ms)):
            chunk = audio[chunk_start:chunk_start + chunk_length_ms]
            chunk_path = f"{file_path}_chunk{i}.mp3"
            chunk.export(chunk_path, format="mp3", bitrate="64k")
            chunks.append(chunk_path)
        return chunks
    except Exception as e:
        print(f"Failed to split audio: {e}")
        return [file_path]

def cleanup_audio(file_paths: list[str]):
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
