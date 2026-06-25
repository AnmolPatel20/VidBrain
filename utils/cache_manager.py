import os
import json
import pickle
import streamlit as st
from typing import Optional
from langchain_community.vectorstores import FAISS

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
METADATA_CACHE = os.path.join(CACHE_DIR, "metadata")
TRANSCRIPTS_CACHE = os.path.join(CACHE_DIR, "transcripts")
CHUNKS_CACHE = os.path.join(CACHE_DIR, "chunks")
VECTORSTORES_CACHE = os.path.join(CACHE_DIR, "vectorstores")

# Ensure all cache directories exist
for directory in [METADATA_CACHE, TRANSCRIPTS_CACHE, CHUNKS_CACHE, VECTORSTORES_CACHE]:
    os.makedirs(directory, exist_ok=True)


class CacheManager:
    @staticmethod
    def _read_json(path: str) -> Optional[dict]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading {path}: {e}")
        return None

    @staticmethod
    def _write_json(path: str, data: dict):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error writing {path}: {e}")

    # --- Metadata Cache ---
    @staticmethod
    def get_metadata(video_id: str) -> Optional[dict]:
        path = os.path.join(METADATA_CACHE, f"{video_id}.json")
        return CacheManager._read_json(path)

    @staticmethod
    def save_metadata(video_id: str, metadata: dict):
        path = os.path.join(METADATA_CACHE, f"{video_id}.json")
        CacheManager._write_json(path, metadata)

    # --- Transcript Cache ---
    @staticmethod
    def get_transcript(video_id: str) -> Optional[dict]:
        """
        Returns dict with "raw" (list of dicts) and "source" (str)
        """
        path = os.path.join(TRANSCRIPTS_CACHE, f"{video_id}.json")
        return CacheManager._read_json(path)

    @staticmethod
    def save_transcript(video_id: str, raw_transcript: list[dict], source: str):
        path = os.path.join(TRANSCRIPTS_CACHE, f"{video_id}.json")
        CacheManager._write_json(path, {"raw": raw_transcript, "source": source})

    # --- Chunks Cache ---
    @staticmethod
    def get_chunks(video_id: str) -> Optional[list]:
        path = os.path.join(CHUNKS_CACHE, f"{video_id}.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error reading {path}: {e}")
        return None

    @staticmethod
    def save_chunks(video_id: str, chunks: list):
        path = os.path.join(CHUNKS_CACHE, f"{video_id}.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump(chunks, f)
        except Exception as e:
            print(f"Error writing {path}: {e}")

    # --- Vectorstore Cache ---
    @staticmethod
    def get_vectorstore(video_id: str, embeddings) -> Optional[FAISS]:
        path = os.path.join(VECTORSTORES_CACHE, video_id)
        # Check if the FAISS index file exists within the directory
        if os.path.exists(path) and os.path.exists(os.path.join(path, "index.faiss")):
            try:
                return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Error loading FAISS vectorstore from {path}: {e}")
        return None

    @staticmethod
    def save_vectorstore(video_id: str, vectorstore: FAISS):
        path = os.path.join(VECTORSTORES_CACHE, video_id)
        try:
            vectorstore.save_local(path)
        except Exception as e:
            print(f"Error saving FAISS vectorstore to {path}: {e}")
