"""
Transcript Indexer

Processes transcripts to build a searchable text index using FAISS and SentenceTransformers.
"""
import os
import sys
import time
import traceback
import argparse
import json
import numpy as np
from tqdm import tqdm
import logging
import glob
from sentence_transformers import SentenceTransformer
import re
import torch

# NOTE: faiss is imported locally within functions that use it to prevent
# a C++ library conflict with torch/sentence_transformers on some systems.

# Define output directories
DEFAULT_INPUT_DIR = "video_index_output"
TRANSCRIPT_INDEX_LOG_DIR = "transcript_indexer_logs"
# NOTE: The Qwen3-Embedding-8B model is a large 8B parameter model that requires a significant
# amount of VRAM (~16GB) and compute for efficient operation. 
# Using a powerful GPU is highly recommended.
DEFAULT_TEXT_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
DEFAULT_BATCH_SIZE = 32
DEFAULT_SAVE_INTERVAL = 0  # Segments

# ====== Setup Logging ======
def setup_transcript_indexer_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "transcript_indexer.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Transcript indexer logging started.")

# ====== Function to Save Transcript Index Data ======
def save_transcript_index_data(index, metadata, config, output_dir, suffix=""):
    """Saves the transcript FAISS index, metadata, and config."""
    import faiss  # Local import to avoid conflicts
    index_path = os.path.join(output_dir, f"transcript_index{suffix}.faiss")
    metadata_path = os.path.join(output_dir, f"transcript_metadata{suffix}.json")
    config_path = os.path.join(output_dir, "transcript_index_config.json")

    logging.info(f"Saving transcript FAISS index ({index.ntotal} vectors) to {index_path}...")
    try:
        faiss.write_index(index, index_path)
    except Exception as e:
        logging.error(f"Error saving transcript FAISS index: {e}", exc_info=True)

    logging.info(f"Saving transcript metadata ({len(metadata)} entries) to {metadata_path}...")
    try:
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=None, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving transcript metadata JSON: {e}", exc_info=True)

    # Save config
    if suffix == "" or not os.path.exists(config_path):
        logging.info(f"Saving transcript index configuration to {config_path}...")
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving transcript index config JSON: {e}", exc_info=True)

    logging.info(f"Saved transcript data with suffix '{suffix}' to {output_dir}")

# ====== Transcript Processing Functions ======
def clean_text(text):
    """Clean and normalize text for better embeddings."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common transcription artifacts
    text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
    text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical content
    
    # Normalize punctuation
    text = re.sub(r'[.]{2,}', '.', text)  # Multiple periods to single
    text = re.sub(r'[,]{2,}', ',', text)  # Multiple commas to single
    
    return text.strip()

def combine_short_segments(segments, min_length=20, max_combine_duration=30.0):
    """Combine short segments with nearby segments for better context."""
    if not segments:
        return segments
    
    combined_segments = []
    i = 0
    
    while i < len(segments):
        current_segment = segments[i]
        current_text = current_segment["text"]
        
        # If segment is long enough, keep it as is
        if len(current_text) >= min_length:
            combined_segments.append(current_segment)
            i += 1
            continue
        
        # Try to combine with next segments
        combined_text = current_text
        end_time = current_segment["end_time"]
        segments_to_combine = [current_segment]
        j = i + 1
        
        while (j < len(segments) and 
               len(combined_text) < 100 and  # Don't make segments too long
               (segments[j]["start_time"] - current_segment["start_time"]) < max_combine_duration):
            
            next_segment = segments[j]
            combined_text += " " + next_segment["text"]
            end_time = next_segment["end_time"]
            segments_to_combine.append(next_segment)
            j += 1
            
            # Stop if we have enough text
            if len(combined_text) >= min_length:
                break
        
        # Only keep the combined segment if it meets minimum length
        if len(combined_text.strip()) >= min_length:
            combined_segment = {
                "video_path": current_segment["video_path"],
                "transcript_path": current_segment["transcript_path"],
                "segment_id": current_segment["segment_id"],
                "start_time": current_segment["start_time"],
                "end_time": end_time,
                "text": combined_text.strip(),
                "language": current_segment["language"]
            }
            combined_segments.append(combined_segment)
        
        i = j
    
    return combined_segments

def extract_segments_from_transcripts(transcript_manifest_path):
    """Extract all transcript segments from the manifest."""
    with open(transcript_manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    all_segments = []
    
    for transcript_info in manifest.get("transcripts", []):
        video_path = transcript_info["video_path"]
        transcript_path = transcript_info["transcript_path"]
        
        # Load the full transcript file
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load transcript {transcript_path}: {e}")
            continue
        
        # Process segments for this video
        video_segments = []
        for segment in transcript_data.get("segments", []):
            segment_text = clean_text(segment.get("text", ""))
            if not segment_text:
                continue
                
            segment_metadata = {
                "video_path": video_path,
                "transcript_path": transcript_path,
                "segment_id": segment.get("id"),
                "start_time": segment.get("start", 0),
                "end_time": segment.get("end", 0),
                "text": segment_text,
                "language": transcript_data.get("language", "unknown")
            }
            
            video_segments.append(segment_metadata)
        
        # Combine short segments for better context
        combined_segments = combine_short_segments(video_segments)
        all_segments.extend(combined_segments)
        
        logging.info(f"Video {os.path.basename(video_path)}: {len(video_segments)} -> {len(combined_segments)} segments after combining")
    
    logging.info(f"Extracted {len(all_segments)} transcript segments from manifest")
    return all_segments

def find_latest_transcript_snapshot(output_dir, manifest_path):
    """Find the latest valid transcript snapshot to resume from."""
    if not os.path.exists(manifest_path):
        return None
        
    manifest_mtime = os.path.getmtime(manifest_path)
    
    potential_indices = glob.glob(os.path.join(output_dir, "transcript_index_*.faiss"))
    valid_snapshots = []
    
    for idx_path in potential_indices:
        suffix = idx_path.replace(os.path.join(output_dir, "transcript_index"), "").replace(".faiss", "")
        if suffix.startswith("_") and suffix[1:].isdigit():
            meta_path = os.path.join(output_dir, f"transcript_metadata{suffix}.json")
            config_path = os.path.join(output_dir, "transcript_index_config.json")
            
            if (os.path.exists(meta_path) and 
                os.path.exists(config_path) and
                os.path.getmtime(idx_path) >= manifest_mtime):
                try:
                    segment_count = int(suffix[1:])
                    valid_snapshots.append((segment_count, suffix))
                except ValueError:
                    continue
    
    if valid_snapshots:
        valid_snapshots.sort(key=lambda x: x[0], reverse=True)
        return valid_snapshots[0][1]
    
    return None

# ====== Main Indexing Function ======
def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    text_model_name = args.text_model
    batch_size = args.batch_size
    save_interval = args.save_interval if args.save_interval >= 0 else 0
    resume_suffix = args.resume_from_suffix

    # Setup logging
    log_dir = os.path.join(input_dir, TRANSCRIPT_INDEX_LOG_DIR)
    setup_transcript_indexer_logging(log_dir)

    logging.info("--- Transcript Indexer ---")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Text model: {text_model_name}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Save interval: {'Disabled' if save_interval == 0 else f'{save_interval} segments'}")

    # Load model first to avoid memory fragmentation.
    logging.info(f"Loading text embedding model: {text_model_name}")
    model_kwargs = {}
    if args.fp16:
        logging.info("Using fp16 for model loading.")
        model_kwargs = {'torch_dtype': torch.float16}

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            # MPS doesn't fully support fp16 with all operations, but we can still try.
            # The performance gain might vary.
            if args.fp16:
                logging.info("fp16 is enabled on MPS. This is experimental.")
            device = "mps"
        
        logging.info(f"Use pytorch device_name: {device}")
        text_model = SentenceTransformer(text_model_name, device=device, model_kwargs=model_kwargs)
        embedding_dim = text_model.get_sentence_embedding_dimension()
        logging.info(f"Text model loaded successfully, embedding dimension: {embedding_dim}")
    except Exception as e:
        logging.error(f"Failed to load text model {text_model_name}: {e}", exc_info=True)
        return

    transcript_manifest_path = os.path.join(input_dir, "transcript_manifest.json")
    if not os.path.exists(transcript_manifest_path):
        logging.error(f"Transcript manifest not found at {transcript_manifest_path}")
        return

    # Defer FAISS import and initialization until after the model is loaded.
    import faiss
    index = None
    metadata = []
    last_processed_segment_count = 0

    suffix_to_load = f"_{resume_suffix}" if resume_suffix else find_latest_transcript_snapshot(input_dir, transcript_manifest_path)

    if suffix_to_load:
        logging.info(f"Attempting to resume from snapshot: '{suffix_to_load}'")
        try:
            index_path = os.path.join(input_dir, f"transcript_index{suffix_to_load}.faiss")
            metadata_path = os.path.join(input_dir, f"transcript_metadata{suffix_to_load}.json")
            index = faiss.read_index(index_path)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if index.d != embedding_dim:
                logging.error(f"FATAL: Dimension mismatch between loaded index ({index.d}) and model ({embedding_dim}).")
                return
            last_processed_segment_count = index.ntotal
            logging.info(f"Resumed successfully. Index has {last_processed_segment_count} vectors.")
        except Exception as e:
            logging.error(f"Could not resume from snapshot {suffix_to_load}. Starting fresh. Error: {e}", exc_info=True)
            index = None # Force re-creation

    if index is None:
        logging.info("Initializing new FAISS index.")
        # For normalized embeddings, IP (Inner Product) is equivalent to Cosine Similarity.
        index = faiss.IndexFlatIP(embedding_dim)
        metadata = []
        last_processed_segment_count = 0

    all_segments = extract_segments_from_transcripts(transcript_manifest_path)
    segments_to_process = all_segments[last_processed_segment_count:]

    if not segments_to_process:
        logging.info("No new segments to process. Index is up to date.")
        return

    logging.info(f"Processing {len(segments_to_process)} segments starting from overall index {last_processed_segment_count}")

    for i in tqdm(range(0, len(segments_to_process), batch_size), desc="Indexing Transcripts", unit="batch"):
        batch_segments = segments_to_process[i:i + batch_size]
        if not batch_segments: continue
        
        batch_texts = [segment["text"] for segment in batch_segments]
        try:
            # Normalize embeddings to unit vectors for cosine similarity
            embeddings = text_model.encode(
                batch_texts, 
                convert_to_numpy=True, 
                show_progress_bar=False,
                normalize_embeddings=True
            )
        except Exception as e:
            logging.error(f"Error encoding batch: {e}", exc_info=True)
            continue
            
        # FAISS requires float32, so we ensure the conversion.
        index.add(embeddings.astype(np.float32))
        metadata.extend(batch_segments)

        current_total_segments = index.ntotal
        if save_interval > 0 and (i // batch_size + 1) % (save_interval // batch_size) == 0 and current_total_segments > 0:
            save_transcript_index_data(index, metadata, {"text_model": text_model_name, "embedding_dim": embedding_dim}, input_dir, suffix=f"_{current_total_segments}")
    
    save_transcript_index_data(index, metadata, {"text_model": text_model_name, "embedding_dim": embedding_dim}, input_dir)
    logging.info("=== Transcript Indexing Complete ===")
    logging.info(f"Total segments indexed: {index.ntotal}")
    logging.info(f"Index saved to: {input_dir}")

# ====== Argument Parsing ======
def parse_args():
    parser = argparse.ArgumentParser(description="Index transcript segments for semantic search.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help=f"Directory containing transcripts and manifest (default: {DEFAULT_INPUT_DIR}).")
    parser.add_argument("--text-model", type=str, default=DEFAULT_TEXT_MODEL_NAME, help=f"Sentence transformer model name (default: {DEFAULT_TEXT_MODEL_NAME}).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size for text embedding (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL, help="Save intermediate index every N segments (0 to disable, default: 0).")
    parser.add_argument("--resume-from-suffix", type=str, default=None, help="Explicitly resume from snapshot with this suffix (e.g., _1000).")
    parser.add_argument("--fp16", action='store_true', help="Use fp16 half-precision for loading the model to save memory.")
    return parser.parse_args()

if __name__ == "__main__":
    main() 