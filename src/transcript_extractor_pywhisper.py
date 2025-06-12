"""
transcript_extractor_pywhisper.py

High-performance transcript extractor using pywhispercpp with Apple Metal GPU acceleration.
Designed to be a drop-in replacement for the original transcript_extractor.py.

Usage:
    pip install pywhispercpp  # or with GPU: WHISPER_COREML=1 pip install git+https://github.com/absadiki/pywhispercpp
    python transcript_extractor_pywhisper.py "/path/to/videos"
"""

import os
import sys
import time
import traceback
import argparse
import json
import hashlib
import logging
import multiprocessing as mp
from tqdm import tqdm

try:
    from pywhispercpp.model import Model as PyWhisperModel
    PYWHISPERCPP_AVAILABLE = True
except ImportError:
    PYWHISPERCPP_AVAILABLE = False
    print("pywhispercpp not installed. Install with: pip install pywhispercpp")

# Same constants as original
HOME_DIR = os.path.expanduser("~")
DEFAULT_EXCLUDE_PATTERNS = [
    os.path.join(HOME_DIR, "Library"),
    os.path.join(HOME_DIR, ".Trash"),
    "node_modules",
    ".git",
    "__pycache__"
]

SUPPORTED_VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', 
    '.mpg', '.mpeg', '.3gp', '.asf', '.rm', '.rmvb', '.vob', '.ogv',
    '.dv', '.ts', '.mts', '.m2ts'
}

SUPPORTED_AUDIO_EXTENSIONS = {
    '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus'
}

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def find_media_files(directories, exclude_patterns=None):
    """Find all video and audio files in given directories"""
    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUDE_PATTERNS
    
    media_files = []
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        print(f"Scanning {directory} for video files...")
        
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            if any(exclude in root for exclude in exclude_patterns):
                continue
                
            # Remove excluded directories from dirs to prevent walking into them
            dirs[:] = [d for d in dirs if not any(exclude in os.path.join(root, d) for exclude in exclude_patterns)]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in SUPPORTED_VIDEO_EXTENSIONS or file_ext in SUPPORTED_AUDIO_EXTENSIONS:
                    media_files.append(file_path)
    
    return sorted(media_files)

def get_file_hash(file_path):
    """Generate a hash for the file based on path and modification time"""
    stat = os.stat(file_path)
    content = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(content.encode()).hexdigest()

def transcribe_single_file(args):
    """Transcribe a single media file using pywhispercpp"""
    media_file, output_dir, whisper_model_name, logger_name = args
    
    # Setup logging for this process
    logger = logging.getLogger(logger_name)
    
    try:
        # Generate output filename
        file_hash = get_file_hash(media_file)
        basename = os.path.splitext(os.path.basename(media_file))[0]
        transcript_filename = f"{basename}_{file_hash[:8]}.json"
        transcript_path = os.path.join(output_dir, "transcripts", transcript_filename)
        
        # Check if transcript already exists
        if os.path.exists(transcript_path):
            logger.info(f"Transcript already exists for {os.path.basename(media_file)}, skipping")
            return {"file": media_file, "success": True, "transcript_path": transcript_path, "status": "existed"}
        
        # Create transcripts directory if it doesn't exist
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
        
        # Load pywhispercpp model
        logger.info(f"Loading pywhispercpp model '{whisper_model_name}' for {os.path.basename(media_file)}")
        
        start_time = time.time()
        
        # Create model - pywhispercpp handles GPU automatically if available
        model = PyWhisperModel(whisper_model_name)
        
        # Transcribe the file
        logger.info(f"Transcribing {os.path.basename(media_file)} with pywhispercpp...")
        segments = model.transcribe(media_file)
        
        # Convert segments to list and extract data
        segments_list = list(segments)
        
        # Build result structure compatible with original format
        result = {
            "text": "",
            "segments": [],
            "language": "unknown"  # pywhispercpp may not expose language detection
        }
        
        full_text_parts = []
        
        for segment in segments_list:
            # Extract segment data (pywhispercpp format may vary)
            if hasattr(segment, 'text'):
                text = segment.text.strip()
            elif hasattr(segment, '__str__'):
                text = str(segment).strip()
            else:
                text = ""
            
            if text:
                full_text_parts.append(text)
                
                # Extract timing from pywhispercpp format (t0, t1 are in centiseconds)
                start_time_seg = getattr(segment, 't0', 0) / 100.0  # Convert centiseconds to seconds
                end_time_seg = getattr(segment, 't1', 0) / 100.0    # Convert centiseconds to seconds
                
                segment_data = {
                    "id": len(result["segments"]),
                    "seek": int(start_time_seg * 100),  # Convert to centiseconds
                    "start": start_time_seg,
                    "end": end_time_seg,
                    "text": text,
                    "tokens": [],  # pywhispercpp may not provide token details
                    "temperature": 0.0,
                    "avg_logprob": 0.0,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.0
                }
                
                # Add word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    segment_data["words"] = []
                    for word in segment.words:
                        word_data = {
                            "word": getattr(word, 'word', str(word)),
                            "start": getattr(word, 'start', start_time_seg),
                            "end": getattr(word, 'end', end_time_seg),
                            "probability": getattr(word, 'probability', 1.0)
                        }
                        segment_data["words"].append(word_data)
                
                result["segments"].append(segment_data)
        
        result["text"] = " ".join(full_text_parts)
        
        # Add metadata
        result["metadata"] = {
            "transcriber": "pywhispercpp",
            "whisper_model": whisper_model_name,
            "source_file": media_file,
            "file_hash": file_hash,
            "transcription_time": time.time() - start_time,
            "timestamp": time.time()
        }
        
        # Save transcript
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Completed {os.path.basename(media_file)} in {duration:.1f}s ({len(result['segments'])} segments)")
        
        return {
            "file": media_file,
            "success": True,
            "transcript_path": transcript_path,
            "segments": len(result["segments"]),
            "duration": duration,
            "status": "transcribed"
        }
        
    except Exception as e:
        logger.error(f"Error transcribing {os.path.basename(media_file)}: {e}")
        logger.error(traceback.format_exc())
        return {
            "file": media_file,
            "success": False,
            "error": str(e),
            "status": "failed"
        }

def create_manifest(output_dir, results):
    """Create a manifest file with all transcript information"""
    manifest = {
        "version": "1.0",
        "transcriber": "pywhispercpp",
        "created": time.time(),
        "transcripts": []
    }
    
    for result in results:
        if result["success"] and "transcript_path" in result:
            transcript_info = {
                "video_path": result["file"],
                "transcript_path": result["transcript_path"],
                "segments": result.get("segments", 0),
                "status": result.get("status", "unknown")
            }
            if "duration" in result:
                transcript_info["transcription_duration"] = result["duration"]
            
            manifest["transcripts"].append(transcript_info)
    
    manifest_path = os.path.join(output_dir, "transcript_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    return manifest_path

def main():
    parser = argparse.ArgumentParser(description="Extract transcripts from media files using pywhispercpp")
    parser.add_argument("directories", nargs="+", help="Directories containing media files")
    parser.add_argument("--output-dir", default="video_index_output", help="Output directory for transcripts")
    parser.add_argument("--whisper-model", default="base", 
                        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3"],
                        help="Whisper model to use")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (pywhispercpp works best with 1)")
    parser.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE_PATTERNS, help="Patterns to exclude from search")
    
    args = parser.parse_args()
    
    # Check if pywhispercpp is available
    if not PYWHISPERCPP_AVAILABLE:
        print("ERROR: pywhispercpp is not installed.")
        print("Install with: pip install pywhispercpp")
        print("Or with GPU support: WHISPER_COREML=1 pip install git+https://github.com/absadiki/pywhispercpp --no-cache --force-reinstall")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=== PyWhisperCPP Transcript Extractor Started ===")
    logger.info(f"Video directories: {args.directories}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Whisper model: {args.whisper_model}")
    logger.info(f"Workers: {args.workers}")
    
    # Find all media files
    media_files = find_media_files(args.directories, args.exclude)
    
    if not media_files:
        logger.warning("No media files found!")
        return
    
    print(f"Found {len(media_files)} video files")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "transcripts"), exist_ok=True)
    
    # Process files
    print(f"Processing {len(media_files)} videos with {args.workers} workers...")
    
    # Prepare arguments for multiprocessing
    process_args = [(media_file, args.output_dir, args.whisper_model, __name__) 
                   for media_file in media_files]
    
    results = []
    
    if args.workers > 1:
        # Use multiprocessing
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap(transcribe_single_file, process_args),
                total=len(process_args),
                desc="Transcribing"
            ))
    else:
        # Single threaded for better GPU utilization
        for process_arg in tqdm(process_args, desc="Transcribing"):
            result = transcribe_single_file(process_arg)
            results.append(result)
    
    # Create manifest
    manifest_path = create_manifest(args.output_dir, results)
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    logger.info("=== Transcription Complete ===")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Manifest saved to: {manifest_path}")
    
    print(f"\nTranscription Summary:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Manifest: {manifest_path}")
    
    return successful == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 