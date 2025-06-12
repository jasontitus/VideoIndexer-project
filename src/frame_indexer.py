"""
Frame Indexer

Processes pre-extracted frames to build a searchable FAISS index.
Works with manifest files created by the Frame Extractor.

Features:
- Automatic resume from snapshots
- Configurable model selection
- Progress tracking and reporting
- Snapshot saving for long-running jobs
"""
import os
import sys
import time
import traceback
import argparse
import json
import torch
import faiss
import numpy as np
from PIL import Image
import glob # Added for snapshot detection
from tqdm import tqdm
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
import logging

# Define output directories relative to the input base (expected structure)
# Assumes input_dir points to the 'video_index_output' created by the extractor
DEFAULT_INPUT_DIR = "video_index_output"
INDEX_LOG_DIR = "indexer_logs" # Specific log dir for this script
DEFAULT_MODEL_NAME = "PE-Core-L14-336"
DEFAULT_BATCH_SIZE = 64
DEFAULT_SAVE_INTERVAL = 0 # Frames

# ====== Setup Logging ======
def setup_indexer_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "frame_indexer.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler(sys.stdout) # Also print logs to console
        ]
    )
    logging.info("Indexer logging started.")

# ====== Function to Save Index Data ======
def save_index_data(index, metadata, config, output_dir, suffix=""):
    """Saves the FAISS index, metadata list, and config to the specified directory."""
    index_path = os.path.join(output_dir, f"video_index{suffix}.faiss")
    metadata_path = os.path.join(output_dir, f"video_frame_metadata{suffix}.json")
    config_path = os.path.join(output_dir, "index_config.json") # Config always saved without suffix

    logging.info(f"Saving FAISS index ({index.ntotal} vectors) to {index_path}...")
    try:
        faiss.write_index(index, index_path)
    except Exception as e:
        logging.error(f"Error saving FAISS index: {e}", exc_info=True)

    logging.info(f"Saving frame metadata list ({len(metadata)} entries) to {metadata_path}...")
    try:
        # Save the metadata directly as a list: [[video_path, frame_num], ...]
        with open(metadata_path, "w") as f:
            # Use json.dump to save the list
            json.dump(metadata, f, indent=None) # No indentation for potentially large lists
    except Exception as e:
        logging.error(f"Error saving metadata JSON list: {e}", exc_info=True)

    # Save config only if it's the final save (or first time)
    if suffix == "" or not os.path.exists(config_path): # Only write config on FINAL save or if missing
        logging.info(f"Saving index configuration to {config_path}...")
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving index config JSON: {e}", exc_info=True)

    logging.info(f"Saved data with suffix '{suffix}' to {output_dir}")

# Add proper configuration saving
def save_index_config(config, output_dir, suffix=""):
    """Save standardized index configuration for use by search server"""
    config_path = os.path.join(output_dir, f"index_config{suffix}.json")
    config_data = {
        "model_name": config["model_name"],
        "embedding_dim": config["embedding_dim"],
        "frame_count": config["frame_count"],
        "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "index_path": f"video_index{suffix}.faiss",
        "metadata_path": f"video_frame_metadata{suffix}.json",
        "thumbnail_dir": "thumbnails" if config.get("save_thumbnails") else None
    }
    
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"Saved index configuration to {config_path}")

# Add proper snapshot management
def find_latest_snapshot(output_dir, manifest_path):
    """Find the latest valid snapshot to resume from"""
    # Get manifest modification time as reference
    manifest_mtime = os.path.getmtime(manifest_path)
    
    # Find all potential snapshots
    potential_indices = glob.glob(os.path.join(output_dir, "video_index_*.faiss"))
    valid_snapshots = []
    
    for idx_path in potential_indices:
        suffix = idx_path.replace(os.path.join(output_dir, "video_index"), "").replace(".faiss", "")
        if suffix.startswith("_") and suffix[1:].isdigit():
            meta_path = os.path.join(output_dir, f"video_frame_metadata{suffix}.json")
            config_path = os.path.join(output_dir, f"index_config{suffix}.json")
            
            # Check if all required files exist and are newer than manifest
            if (os.path.exists(meta_path) and 
                os.path.exists(config_path) and
                os.path.getmtime(idx_path) >= manifest_mtime):
                try:
                    frame_count = int(suffix[1:])
                    valid_snapshots.append((frame_count, suffix))
                except ValueError:
                    continue
    
    if valid_snapshots:
        valid_snapshots.sort(key=lambda x: x[0], reverse=True)
        return valid_snapshots[0][1]  # Return the suffix with highest frame count
    
    return None

# ====== Argument Parsing ======
def parse_args():
    parser = argparse.ArgumentParser(description="Index extracted video frames using a vision model.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing extracted frames and manifest (default: {DEFAULT_INPUT_DIR}).")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                        choices=['PE-Core-B16-224', 'PE-Core-L14-336', 'PE-Core-G14-448'],
                        help=f"Name of the PE Core model to use (default: {DEFAULT_MODEL_NAME}).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for model inference (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL,
                        help="Save intermediate index every N frames (0 to disable, default: 0).")
    parser.add_argument("--resume-from-suffix", type=str, default=None,
                        help="Explicitly resume from snapshot with this suffix (e.g., _10000). Overrides auto-detection.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16 precision for model inference to reduce memory usage (default: fp32).")
    return parser.parse_args()

# --- Main Execution Guard ---
if __name__ == "__main__":
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    # Get command-line model arg, but don't assume it's the final one yet
    cmd_line_model_name = args.model_name
    batch_size = args.batch_size
    save_interval = args.save_interval if args.save_interval >= 0 else 0
    resume_suffix = args.resume_from_suffix
    use_fp16 = args.fp16

    # Setup logging
    log_dir = os.path.join(input_dir, INDEX_LOG_DIR)
    setup_indexer_logging(log_dir)

    logging.info("--- Frame Indexer ---")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Save interval: {'Disabled' if save_interval == 0 else f'{save_interval} frames'}")
    logging.info(f"Precision: {'fp16' if use_fp16 else 'fp32'}")

    # --- Configuration Loading and Model Determination --- #
    initial_config = {}
    config_model_name = None
    config_embedding_dim = None
    model_to_load = None
    expected_embedding_dim = None # Dimension expected based on config or determined by model
    suffix_to_load = None # Suffix for loading index/metadata

    # 1. Load existing config if present
    config_path = os.path.join(input_dir, "index_config.json")
    if os.path.exists(config_path):
        logging.info(f"Loading existing configuration from {config_path}...")
        try:
            with open(config_path, "r") as f:
                initial_config = json.load(f)
            config_model_name = initial_config.get("model_name")
            config_embedding_dim = initial_config.get("embedding_dim")
            if not config_model_name or not isinstance(config_embedding_dim, int):
                logging.warning("Existing index_config.json is missing model_name or embedding_dim. It will be ignored for validation/resumption.")
                initial_config = {} # Treat as invalid
                config_model_name = None
                config_embedding_dim = None
        except Exception as e:
            logging.error(f"Error loading existing index config: {e}. It will be ignored.", exc_info=True)
            initial_config = {}
            config_model_name = None
            config_embedding_dim = None
    else:
        logging.info("No existing index_config.json found.")

    # 2. Determine Model and Suffix based on Resume flag
    if resume_suffix:
        # --- Explicit Resume: Config is king --- #
        logging.info(f"Explicit resume requested from suffix: {resume_suffix}")
        if not initial_config: # Check if config was loaded successfully
            logging.error(f"FATAL: Cannot resume from suffix '{resume_suffix}' because a valid index_config.json was not found or loaded in the input directory.")
            sys.exit(1)

        model_to_load = config_model_name
        expected_embedding_dim = config_embedding_dim # Expect dimension from config
        suffix_to_load = resume_suffix
        if not suffix_to_load.startswith("_"): suffix_to_load = f"_{suffix_to_load}" # Ensure format

        if cmd_line_model_name != model_to_load:
            logging.warning(f"Warning: Command line model '{cmd_line_model_name}' differs from config model '{model_to_load}' required for resumption. Using config model '{model_to_load}'.")
        logging.info(f"Resuming with model specified in config: {model_to_load}")

    else:
        # --- No Explicit Resume: Use command line args, check for conflicts, auto-detect --- #
        model_to_load = cmd_line_model_name # Use command line model
        logging.info(f"Using specified model: {model_to_load}")
        # Embedding dim will be determined after loading model
        # We still need to check for conflicts if a config exists

        # Auto-detect latest valid suffix for potential auto-resume
        logging.info("Attempting to auto-detect latest valid snapshot...")
        potential_indices = glob.glob(os.path.join(input_dir, "video_index_*.faiss"))
        valid_suffixes = []
        for idx_path in potential_indices:
            suffix = idx_path.replace(os.path.join(input_dir, "video_index"), "").replace(".faiss", "")
            if suffix.startswith("_") and suffix[1:].isdigit():
                 meta_path = os.path.join(input_dir, f"video_frame_metadata{suffix}.json")
                 if os.path.exists(meta_path):
                     try:
                        frame_count = int(suffix[1:])
                        valid_suffixes.append((frame_count, suffix))
                     except ValueError: continue
        if valid_suffixes:
            valid_suffixes.sort(key=lambda item: item[0], reverse=True)
            suffix_to_load = valid_suffixes[0][1]
            logging.info(f"Auto-detected latest valid snapshot suffix: {suffix_to_load}")
        else:
            logging.info("No valid snapshots found for auto-detection.")
            suffix_to_load = None # Will initialize fresh

    # ====== Load Manifest (Can happen anytime before processing loop) ======
    manifest_path = os.path.join(input_dir, "extraction_manifest.json") # Define manifest path
    if not os.path.exists(manifest_path):
        logging.error(f"Error: Manifest file '{manifest_path}' not found.")
        sys.exit(1)
    logging.info(f"Loading frame info from {manifest_path}...") # Use manifest_path
    frame_info_list = []
    try:
        with open(manifest_path, "r") as f: # Use manifest_path
            manifest_data = json.load(f)
            # Expecting a list of objects, each detailing a frame
            extracted_frames_data = manifest_data.get("extracted_frames", [])
            if not extracted_frames_data:
                 logging.warning("Manifest file contains no frame data in 'extracted_frames'.")
            else:
                # Validate and store frame info
                required_keys = ["frame_path", "original_video_path", "frame_number"]
                for i, frame_data in enumerate(extracted_frames_data):
                    if isinstance(frame_data, dict) and all(key in frame_data for key in required_keys):
                        frame_info_list.append(frame_data)
                    else:
                        logging.warning(f"Manifest entry {i} is missing required keys ({required_keys}) or is not an object. Skipping: {frame_data}")

                logging.info(f"Loaded info for {len(frame_info_list)} frames from manifest.")
    except Exception as e:
        logging.error(f"Error reading or parsing manifest file: {e}", exc_info=True)
        sys.exit(1)

    total_frames_to_index = len(frame_info_list)
    if total_frames_to_index == 0:
        logging.info("No valid frame information loaded from manifest. Exiting.")
        sys.exit(0)

    # ====== Setup Device (Can happen anytime before model load) ======
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # ====== Load the Determined Model ======
    logging.info(f"Loading model {model_to_load}...")
    try:
        model = pe.CLIP.from_config(model_to_load, pretrained=True)
        model = model.to(device)
        
        # Convert to fp16 if requested
        if use_fp16:
            if device.type == "cpu":
                logging.warning("fp16 requested but using CPU device. fp16 is not well supported on CPU, continuing with fp32.")
                use_fp16 = False  # Disable fp16 for CPU
            else:
                logging.info("Converting model to fp16 precision...")
                model = model.half()
        
        model.eval()
        image_size_tuple = model.image_size
        local_preprocess = transforms.get_image_transform(image_size_tuple)
        actual_embedding_dim = model.visual.proj.shape[1] # Get actual dim from loaded model
        logging.info(f"Model loaded. Image size: {image_size_tuple}, Embedding dim: {actual_embedding_dim}")
    except Exception as e:
        logging.error(f"Error loading model {model_to_load}: {e}", exc_info=True)
        sys.exit(1)

    # ====== Validate Dimensions ======
    # Case 1: Explicit resume - loaded model dim must match config dim
    if resume_suffix:
        if actual_embedding_dim != expected_embedding_dim:
            logging.error(f"FATAL: Loaded model '{model_to_load}' dimension ({actual_embedding_dim}) does not match dimension expected from config ({expected_embedding_dim}). Cannot resume.")
            sys.exit(1)
    # Case 2: No explicit resume, but config exists - loaded model dim must match config dim
    elif config_embedding_dim is not None: # Check if config was loaded and valid
         if actual_embedding_dim != config_embedding_dim:
            logging.error(f"FATAL: Loaded model '{model_to_load}' dimension ({actual_embedding_dim}) does not match dimension found in existing index_config.json ({config_embedding_dim}). Delete config or fix model mismatch.")
            sys.exit(1)
         else:
            expected_embedding_dim = actual_embedding_dim # Config matches model, use actual dim
    # Case 3: No explicit resume, no config - use actual_embedding_dim
    else:
        expected_embedding_dim = actual_embedding_dim # Set expected dim for index creation
    
    # Ensure expected_embedding_dim is always set
    if expected_embedding_dim is None:
        expected_embedding_dim = actual_embedding_dim
        logging.info(f"Fallback: Setting expected_embedding_dim to actual model dimension: {actual_embedding_dim}")

    # ====== Setup FAISS index & Metadata Storage (Load or Initialize) ======
    index = None
    indexed_frame_metadata = []
    processed_frame_count = 0 # How many frames are already IN the loaded index

    if suffix_to_load:
        snapshot_index_path = os.path.join(input_dir, f"video_index{suffix_to_load}.faiss")
        snapshot_metadata_path = os.path.join(input_dir, f"video_frame_metadata{suffix_to_load}.json")
        logging.info(f"Attempting to load snapshot from suffix '{suffix_to_load}'...")
        logging.info(f"Index file: {snapshot_index_path}")
        logging.info(f"Metadata file: {snapshot_metadata_path}")

        if os.path.exists(snapshot_index_path) and os.path.exists(snapshot_metadata_path):
            try:
                logging.info("Loading FAISS index...")
                index = faiss.read_index(snapshot_index_path)
                # Validate against the *final* expected dimension (from config or loaded model)
                if index.d != expected_embedding_dim:
                    raise ValueError(f"Loaded index dimension ({index.d}) does not match expected dimension ({expected_embedding_dim})")
                logging.info(f"Loaded index with {index.ntotal} vectors.")

                logging.info("Loading metadata list...")
                with open(snapshot_metadata_path, "r") as f:
                    indexed_frame_metadata = json.load(f)
                logging.info(f"Loaded metadata for {len(indexed_frame_metadata)} frames.")

                if index.ntotal != len(indexed_frame_metadata):
                     raise ValueError(f"Index vector count ({index.ntotal}) does not match metadata length ({len(indexed_frame_metadata)})")

                processed_frame_count = index.ntotal
                logging.info(f"Successfully loaded snapshot. Resuming after frame {processed_frame_count}.")

            except Exception as e:
                logging.error(f"Error loading snapshot '{suffix_to_load}': {e}. Starting fresh.", exc_info=True)
                index = None
                indexed_frame_metadata = []
                processed_frame_count = 0
        else:
            logging.warning(f"Snapshot files not found for suffix '{suffix_to_load}'. Starting fresh.")
            index = None

    # Initialize fresh if no snapshot loaded or loading failed
    if index is None:
        logging.info(f"Initializing fresh FAISS index (Dim: {expected_embedding_dim})...")
        index = faiss.IndexFlatL2(expected_embedding_dim) # Use the final expected dimension
        indexed_frame_metadata = []
        processed_frame_count = 0

    # ====== Process Frames ======
    # Determine frames to process (skip already processed ones)
    remaining_frames_info = frame_info_list[processed_frame_count:]
    num_remaining_frames = len(remaining_frames_info)

    if num_remaining_frames == 0:
        logging.info("No remaining frames to process based on loaded snapshot/manifest. Exiting.")
        sys.exit(0)
    else:
         logging.info(f"Starting indexing for {num_remaining_frames} remaining frames (Total in manifest: {total_frames_to_index}, Already indexed: {processed_frame_count})...")

    processing_start_time = time.time()
    accumulated_tensors = []
    accumulated_metadata_for_batch = [] # Stores the [video_path, frame_num] for the current batch

    # Counters for THIS run
    newly_processed_frame_count = 0
    failed_frame_count = 0
    total_inference_time = 0.0
    total_faiss_add_time = 0.0 # Reset timers for this run
    frames_since_last_save = 0

    pbar = tqdm(remaining_frames_info, total=num_remaining_frames, desc="Indexing Frames", unit="frame") # Iterate over remaining

    for frame_info in pbar: # Now iterates over remaining frames
        frame_path = frame_info["frame_path"]
        original_video_path = frame_info["original_video_path"]
        frame_number = frame_info["frame_number"]

        # Update progress bar description to show total progress
        pbar.set_description(f"Indexing Frames (Total: {processed_frame_count + newly_processed_frame_count}/{total_frames_to_index})")

        try:
            # Check if frame path is absolute, if not, assume relative to input_dir
            if not os.path.isabs(frame_path):
                # Important: Resolve path relative to the manifest's directory (input_dir)
                # Assuming frame_path in manifest is relative to input_dir
                absolute_frame_path = os.path.abspath(os.path.join(input_dir, frame_path))
            else:
                absolute_frame_path = frame_path

            if not os.path.exists(absolute_frame_path):
                 logging.warning(f"Skipping frame: File not found at {absolute_frame_path} (original path in manifest: {frame_path})")
                 failed_frame_count += 1
                 continue

            # Load and preprocess image
            img = Image.open(absolute_frame_path).convert("RGB")
            preprocessed_tensor = local_preprocess(img).cpu() # Keep on CPU until batching

            accumulated_tensors.append(preprocessed_tensor)
            accumulated_metadata_for_batch.append([original_video_path, frame_number])

            # --- Process Batch When Ready ---
            if len(accumulated_tensors) >= batch_size:
                batch_processed_ok = False
                try:
                    imgs_tensor = torch.stack(accumulated_tensors).to(device)
                    
                    # Convert to fp16 if model is in fp16
                    if use_fp16:
                        imgs_tensor = imgs_tensor.half()

                    # Model Inference
                    inference_start_time = time.time()
                    with torch.no_grad():
                        image_features, _, _ = model(imgs_tensor, None)
                    inference_end_time = time.time()
                    total_inference_time += (inference_end_time - inference_start_time)

                    # Add to FAISS Index
                    faiss_start_time = time.time()
                    index.add(image_features.cpu().numpy())
                    faiss_end_time = time.time()
                    total_faiss_add_time += (faiss_end_time - faiss_start_time)

                    # Update overall metadata and counts
                    batch_size_processed = len(accumulated_tensors)
                    indexed_frame_metadata.extend(accumulated_metadata_for_batch) # Append metadata for the new batch
                    processed_frame_count += batch_size_processed # Increment total frames including snapshot
                    newly_processed_frame_count += batch_size_processed # Increment frames processed *this run*
                    frames_since_last_save += batch_size_processed
                    batch_processed_ok = True

                except Exception as batch_e:
                    logging.error(f"Error processing batch ending with frame {frame_path}: {batch_e}", exc_info=True)
                    failed_frame_count += len(accumulated_tensors) # Count batch as failed

                finally:
                     # Clear accumulation lists regardless of success/failure
                     accumulated_tensors = []
                     accumulated_metadata_for_batch = []


                # --- Check for Intermediate Save --- # Note: processed_frame_count is the *total* count including snapshot
                if batch_processed_ok and save_interval > 0 and frames_since_last_save >= save_interval:
                    save_suffix = f"_{processed_frame_count}"
                    # Use initial_config if loaded, otherwise create new. Don't overwrite if just intermediate.
                    current_config = initial_config if initial_config else {"model_name": model_to_load, "embedding_dim": expected_embedding_dim}
                    logging.info(f"--- SAVING INTERMEDIATE INDEX (Total Frames: {processed_frame_count}) ---")
                    save_index_data(index, indexed_frame_metadata, current_config, input_dir, suffix=save_suffix) # Pass current_config
                    frames_since_last_save = 0 # Reset counter

        except Exception as frame_e:
            logging.warning(f"Skipping frame {frame_path} due to error: {frame_e}", exc_info=True)
            failed_frame_count += 1
            # Ensure this frame isn't left in accumulation lists if error occurred before batching
            # Check using the metadata tuple/list, not just frame_path
            metadata_tuple = [original_video_path, frame_number]
            if metadata_tuple in accumulated_metadata_for_batch:
                 try:
                     idx = accumulated_metadata_for_batch.index(metadata_tuple)
                     accumulated_metadata_for_batch.pop(idx)
                     accumulated_tensors.pop(idx)
                 except (ValueError, IndexError):
                     pass # Should not happen if logic is correct

    pbar.close()

    # --- Process Final Remaining Batch ---
    if accumulated_tensors:
        logging.info(f"Processing final remaining batch of {len(accumulated_tensors)} frames...")
        try:
            imgs_tensor = torch.stack(accumulated_tensors).to(device)
            
            # Convert to fp16 if model is in fp16
            if use_fp16:
                imgs_tensor = imgs_tensor.half()
                
            inference_start_time = time.time()
            with torch.no_grad():
                image_features, _, _ = model(imgs_tensor, None)
            inference_end_time = time.time()
            total_inference_time += (inference_end_time - inference_start_time)

            faiss_start_time = time.time()
            index.add(image_features.cpu().numpy())
            faiss_end_time = time.time()
            total_faiss_add_time += (faiss_end_time - faiss_start_time)

            batch_size_processed = len(accumulated_tensors)
            indexed_frame_metadata.extend(accumulated_metadata_for_batch)
            processed_frame_count += batch_size_processed # Increment total count
            newly_processed_frame_count += batch_size_processed # Increment this run's count
            # No need to check save interval here, final save follows

        except Exception as batch_e:
            logging.error(f"Error processing final batch: {batch_e}", exc_info=True)
            failed_frame_count += len(accumulated_tensors)
        finally:
            # Clear lists
            accumulated_tensors = []
            accumulated_metadata_for_batch = []

    processing_end_time = time.time()
    total_script_time = processing_end_time - processing_start_time # Only time indexing phase for *this run*

    # ====== 6. Save FINAL index and metadata ======
    logging.info(f"===== Indexing Run Complete =====")
    logging.info(f"Frames submitted via manifest: {total_frames_to_index}")
    logging.info(f"Frames processed in this run: {newly_processed_frame_count}")
    logging.info(f"Total frames indexed (incl. resumed): {processed_frame_count}")
    logging.info(f"Failed/skipped frames in this run: {failed_frame_count}")

    if processed_frame_count > 0:
        # Perform final save
        # Final config should always reflect the model used in *this* run
        # (or the consistent model if resuming)
        index_config = {"model_name": model_to_load, "embedding_dim": expected_embedding_dim}
        logging.info("Performing final save...")
        save_index_data(index, indexed_frame_metadata, index_config, input_dir, suffix="") # Empty suffix for final
    else:
        logging.info("No index generated as no frames were successfully processed.")

    # ====== 7. Print Summary ======
    logging.info(f"Total time for this indexing run: {total_script_time:.2f} seconds")
    if newly_processed_frame_count > 0: # Calculate stats based on *this* run's work
        avg_fps = newly_processed_frame_count / max(1, total_script_time)
        logging.info(f"Average indexing rate (this run): {avg_fps:.2f} frames/sec")
        logging.info(f"  - Time in Model Inference (this run): {total_inference_time:.2f} seconds")
        logging.info(f"  - Time adding to FAISS index (this run): {total_faiss_add_time:.2f} seconds")
        other_time = total_script_time - total_inference_time - total_faiss_add_time
        logging.info(f"  - Est. Image Loading + Preprocessing + Overhead (this run): {other_time:.2f} seconds")

    logging.info("Frame indexer finished.") 