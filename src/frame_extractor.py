"""
frame_extractor.py

Scans directories for videos, uses multiprocessing to extract frames at
a specified interval, and saves the frames as image files.
Creates a manifest file listing all successfully extracted frames.
"""

import os
import sys
import time
import traceback
import argparse
import json
# import torch # Keep for type hinting if needed, but model not used here
# import faiss # Keep for type hinting if needed, but index not used here
# import numpy as np # Not needed
from PIL import Image # Not needed directly, handled by cv2 save
from tqdm import tqdm
import cv2
# import core.vision_encoder.transforms as transforms # Needed for image size info? No, remove.
import multiprocessing as mp
import hashlib # For hashing paths
import logging # For worker logging

# Define default directory names or paths relative to HOME to exclude
HOME_DIR = os.path.expanduser("~")
DEFAULT_EXCLUDE_PATTERNS = [
    os.path.join(HOME_DIR, "Library"),
    os.path.join(HOME_DIR, "Documents"),
    os.path.join(HOME_DIR, ".Trash"),
    os.path.join(HOME_DIR, ".cache"),
    "Photos Library.photoslibrary",
    "node_modules",
    ".git",
    "__pycache__",
    ".AppleDouble"
]

# Output structure
DEFAULT_OUTPUT_DIR = "video_index_output"
FRAME_IMAGE_DIR = "frame_images"
THUMBNAIL_DIR = "thumbnails"
EXTRACTOR_LOG_DIR = "extractor_logs"
WORKER_LOG_SUBDIR = "worker_logs"
MANIFEST_FILENAME = "extraction_manifest.json"

DEFAULT_FRAME_INTERVAL = 1.0
DEFAULT_WORKERS = max(1, os.cpu_count() // 2)
THUMBNAIL_HEIGHT = 150
FRAME_IMAGE_QUALITY = 90 # JPEG Quality

# ====== Worker Setup for Logging ======
def setup_worker_logging(pid, base_log_dir):
    # Create a unique log file for each worker inside the worker log subdir
    log_dir = os.path.join(base_log_dir, WORKER_LOG_SUBDIR)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"worker_{pid}.log")

    logger = logging.getLogger(f'worker_{pid}')
    logger.setLevel(logging.INFO) # Log info and above
    logger.propagate = False

    if not logger.hasHandlers():
        # File handler
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Redirect the worker's stderr to the same log file
        try:
            log_file_path = fh.baseFilename
            stderr_log_file = open(log_file_path, 'a', buffering=1)
            sys.stderr = stderr_log_file
        except Exception as e:
            # Use print here as logger might not be fully set up if fh failed
            print(f"[Worker {pid}] Failed to redirect stderr to {log_file_path}: {e}")

    return logger

# ====== Worker Function for Video Frame Extraction ======
def process_video_extract_task(task_data):
    """
    Processes a single video: extracts frames and saves them as images.
    task_data: (video_path_abs, [frame_time_ms, ...], output_dir_base, backend_preference)
    Returns: Tuple(total_frames_planned, List of successfully saved frame info dictionaries).
    """
    # Imports needed for the worker
    import os, time, hashlib, cv2, logging, sys

    video_path_abs, frame_times, output_dir_base, backend_preference = task_data
    pid = os.getpid()
    base_log_dir = os.path.join(output_dir_base, EXTRACTOR_LOG_DIR)
    logger = setup_worker_logging(pid, base_log_dir)

    saved_frame_info_list = []
    total_frames_in_task = len(frame_times)
    processed_count = 0
    failed_count = 0

    # Define output paths relative to the base output directory
    path_hash = hashlib.sha256(video_path_abs.encode()).hexdigest()
    frame_image_video_dir = os.path.join(output_dir_base, FRAME_IMAGE_DIR, path_hash)
    # --- Define SEPARATE output path for THUMBNAILS ---
    # Use Application Support directory if available, otherwise fallback to local
    try:
        from app_paths import get_video_thumbnails_dir
        thumbnail_output_base = get_video_thumbnails_dir()
    except ImportError:
        thumbnail_output_base = "video_thumbnails_output" # Fallback for development
    
    thumbnail_video_dir = os.path.join(thumbnail_output_base, path_hash)
    os.makedirs(thumbnail_video_dir, exist_ok=True) # Ensure thumbnail dir exists
    # --- End Define SEPARATE ---
    os.makedirs(frame_image_video_dir, exist_ok=True)
    # os.makedirs(thumbnail_video_dir, exist_ok=True) # Redundant, created above

    video = None
    t_video_start = time.time()
    t_open = 0
    video_opened_ok = False
    backend_name = "Unknown" # Variable to store backend info

    try:
        t0 = time.time()
        # Use the backend determined during the scan phase
        if backend_preference == 'avfoundation':
            video = cv2.VideoCapture(video_path_abs, cv2.CAP_AVFOUNDATION)
            backend_name = "AVFOUNDATION (Requested)"
        else: # 'fallback' or any other case
            video = cv2.VideoCapture(video_path_abs)
            backend_name = "Default/Fallback (Requested)"
        t_open = time.time() - t0

        # Single check if opening succeeded (no internal fallback here)
        if not video.isOpened():
            logger.error(f"VideoNotOpened | Backend: {backend_preference} | {video_path_abs}")
            # Return planned count and empty list on failure
            return total_frames_in_task, saved_frame_info_list

        video_opened_ok = True

        # Get actual backend information if possible (after successful open)
        try:
             backend_id = video.get(cv2.CAP_PROP_BACKEND)
             backend_map = {cv2.CAP_FFMPEG: "FFMPEG", cv2.CAP_AVFOUNDATION: "AVFOUNDATION",
                           cv2.CAP_GSTREAMER: "GSTREAMER", cv2.CAP_V4L2: "V4L2",
                           cv2.CAP_OPENNI: "OPENNI", cv2.CAP_IMAGES: "IMAGES",
                           cv2.CAP_MSMF: "MSMF", cv2.CAP_OPENCV_MJPEG: "OPENCV_MJPEG" }
             actual_backend_name = backend_map.get(backend_id, f"ID_{backend_id}")
             backend_name = f"{backend_name.split(' ')[0]} (Actual: {actual_backend_name})" # Combine requested and actual
        except AttributeError:
             backend_name += " (Actual: N/A - Old CV2?)"
        except Exception as be:
             backend_name += f" (Actual: Error {be})"

        for frame_time_ms in frame_times:
            t_seek = t_read = t_thumb_save = t_frame_save = 0
            status = "FAIL"
            error_msg = ""
            frame_rel_path = None

            try:
                # --- Seek --- #
                t0 = time.time()
                seek_success = video.set(cv2.CAP_PROP_POS_MSEC, frame_time_ms)
                t_seek = time.time() - t0
                if not seek_success:
                    error_msg = "SeekFailed"
                    raise ValueError(error_msg)

                # --- Read --- #
                t0 = time.time()
                ret, frame = video.read()
                t_read = time.time() - t0
                if not ret or frame is None:
                    error_msg = "ReadFailed"
                    raise ValueError(error_msg)

                frame_id_str = f"{int(frame_time_ms)}"

                # --- Save Thumbnail --- #
                t0 = time.time()
                try:
                    h, w = frame.shape[:2]
                    if h == 0 or w == 0: raise ValueError("Invalid frame dimensions for thumbnail")
                    aspect_ratio = w / h
                    new_width = int(THUMBNAIL_HEIGHT * aspect_ratio)
                    thumbnail_frame = cv2.resize(frame, (new_width, THUMBNAIL_HEIGHT), interpolation=cv2.INTER_AREA)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_IMAGE_QUALITY]
                    thumbnail_filename = f"{frame_id_str}.jpg"
                    # Path uses the separate thumbnail directory
                    thumbnail_path_abs = os.path.join(thumbnail_video_dir, thumbnail_filename)
                    save_ok = cv2.imwrite(thumbnail_path_abs, thumbnail_frame, encode_param)
                    if not save_ok:
                        logger.warning(f"ThumbSaveFail | {video_path_abs} | {frame_time_ms}ms")
                except Exception as thumb_err:
                    logger.warning(f"ThumbErr | {video_path_abs} | {frame_time_ms}ms | Err: {thumb_err}")
                t_thumb_save = time.time() - t0

                # --- Save Full Frame Image --- #
                t0 = time.time()
                try:
                    frame_filename = f"{frame_id_str}.jpg"
                    frame_path_abs = os.path.join(frame_image_video_dir, frame_filename)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_IMAGE_QUALITY]
                    save_ok = cv2.imwrite(frame_path_abs, frame, encode_param)
                    if not save_ok:
                        error_msg = "FrameSaveFailed"
                        raise ValueError(error_msg)
                    # Store relative path for manifest
                    frame_rel_path = os.path.relpath(frame_path_abs, output_dir_base)
                    status = "OK"
                except Exception as frame_save_err:
                    error_msg = f"FrameSaveErr: {frame_save_err}"
                    raise ValueError(error_msg)
                t_frame_save = time.time() - t0

                # Add info dict to list only on full success
                saved_frame_info_list.append({
                    "frame_path": frame_rel_path,
                    "original_video_path": video_path_abs,
                    "frame_number": int(frame_time_ms)
                })
                processed_count += 1

            except Exception as e:
                if not error_msg:
                    error_msg = f"OtherError: {type(e).__name__}: {e}"
                status = "FAIL"
                failed_count += 1

            finally:
                # Log per-frame status
                log_line = f"{status} | {video_path_abs} | {frame_time_ms} | {t_seek:.4f} | {t_read:.4f} | {t_thumb_save:.4f} | {t_frame_save:.4f} | {error_msg}"
                logger.info(log_line)

    except Exception as video_e:
        logger.error(f"VideoProcessingError | Backend: {backend_preference} | {video_path_abs} | Err: {video_e}", exc_info=True)
        failed_count = total_frames_in_task - processed_count

    finally:
        if video is not None and video.isOpened():
            video.release()

        # Log video summary
        t_video_end = time.time()
        total_video_time = t_video_end - t_video_start
        summary_status = "OK" if failed_count == 0 else "PARTIAL" if processed_count > 0 else "FAIL"
        logger.info(f"VideoSummary | {summary_status} | {video_path_abs} | Backend: {backend_name} | Processed: {processed_count}/{total_frames_in_task} | Failed: {failed_count} | TotalTime: {total_video_time:.2f} | OpenTime: {t_open:.4f}")

    # Return planned count AND the list of successful frame info dicts
    return total_frames_in_task, saved_frame_info_list

# ====== Argument Parsing ======
def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from videos and save as images.")
    parser.add_argument("start_dir", help="The root directory to start scanning from.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Base directory to save extracted frames, thumbnails, logs, and manifest (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--exclude", nargs='+', default=[],
                        help="Additional directories or paths to exclude from scanning.")
    parser.add_argument("--stay-on-device", action="store_true",
                        help="Prevent scanning from crossing filesystem boundaries.")
    parser.add_argument("--frame-interval", type=float, default=DEFAULT_FRAME_INTERVAL,
                        help=f"Interval in seconds between frames to extract (default: {DEFAULT_FRAME_INTERVAL}).")
    # --- Worker arguments ---
    default_av_workers = 2 # Sensible default for limited hardware decoders
    default_fallback_workers = max(1, (os.cpu_count() // 2) - default_av_workers) # Use remaining cores
    parser.add_argument("--av-workers", type=int, default=default_av_workers,
                        help=f"Number of workers optimized for AVFoundation (hardware) (default: {default_av_workers}).")
    parser.add_argument("--fallback-workers", type=int, default=default_fallback_workers,
                        help=f"Number of workers for fallback (CPU) processing (default: {default_fallback_workers}).")
    return parser.parse_args()

# --- Main Execution Guard ---
if __name__ == "__main__":
    # Set start method (consider spawn for macOS/CUDA consistency)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Info: Could not set start method to 'spawn', using default.")

    args = parse_args()
    start_dir = os.path.abspath(args.start_dir)
    output_dir_base = os.path.abspath(args.output_dir)
    frame_interval = args.frame_interval
    # num_workers = args.workers # Replaced by specific worker counts
    num_av_workers = args.av_workers
    num_fallback_workers = args.fallback_workers
    total_workers = num_av_workers + num_fallback_workers

    # Setup base logging directory
    base_log_dir = os.path.join(output_dir_base, EXTRACTOR_LOG_DIR)
    os.makedirs(base_log_dir, exist_ok=True)
    # Basic configuration for the main process log file
    main_log_file = os.path.join(base_log_dir, "frame_extractor.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(main_log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("--- Frame Extractor ---")
    logging.info(f"Scanning from: {start_dir}")
    logging.info(f"Outputting to: {output_dir_base}")
    logging.info(f"Frame interval: {frame_interval:.2f} seconds")
    logging.info(f"Worker processes: AVFoundation={num_av_workers}, Fallback={num_fallback_workers} (Total={total_workers})")

    # Validate args
    if frame_interval <= 0:
        logging.error("Error: Frame interval must be positive.")
        sys.exit(1)
    if num_av_workers < 0 or num_fallback_workers < 0:
        logging.error("Error: Number of workers cannot be negative.")
        sys.exit(1)
    if total_workers <= 0:
        logging.error("Error: Total number of workers must be positive.")
        sys.exit(1)

    # Prepare exclusion list (absolute paths)
    exclude_paths_abs = set()
    for pattern in DEFAULT_EXCLUDE_PATTERNS:
        if os.path.isabs(pattern):
            exclude_paths_abs.add(os.path.normpath(pattern))
        else:
            # Add pattern relative to home AND as a basename pattern
            potential_abs_path = os.path.normpath(os.path.join(HOME_DIR, pattern))
            exclude_paths_abs.add(potential_abs_path)
            exclude_paths_abs.add(pattern) # Keep basename pattern
    for user_path in args.exclude:
        abs_user_path = os.path.normpath(os.path.abspath(os.path.expanduser(user_path)))
        exclude_paths_abs.add(abs_user_path)

    exclude_list_sorted = sorted(list(exclude_paths_abs))
    logging.info(f"Excluding paths/patterns matching: {exclude_list_sorted}")
    if args.stay_on_device:
        logging.info("Option --stay-on-device enabled.")

    if not os.path.isdir(start_dir):
        logging.error(f"Error: Start directory '{start_dir}' not found or is not a directory.")
        sys.exit(1)

    # Get starting device ID if needed
    start_dev = -1
    if args.stay_on_device:
        try:
            start_dev = os.stat(start_dir).st_dev
            logging.info(f"Starting device ID: {start_dev}")
        except OSError as e:
            logging.error(f"Error getting device ID for {start_dir}: {e}. Cannot use --stay-on-device.")
            sys.exit(1)

    # ====== Scan Directories and Generate Video Tasks ======
    logging.info("Scanning directories to identify videos and generate tasks...")
    scan_start_time = time.time()
    # Separate task lists
    av_tasks = []       # List of (video_path_abs, [frame_times], output_dir_base, 'avfoundation')
    fallback_tasks = [] # List of (video_path_abs, [frame_times], output_dir_base, 'fallback')
    videos_found_paths = [] # Just for counting/logging

    found_video_count = 0
    skipped_permission_count = 0
    skipped_excluded_dir_count = 0
    skipped_device_count = 0
    initial_scan_errors = 0
    categorized_av = 0
    categorized_fallback = 0

    # Use a context manager for tqdm if possible, or ensure close
    walker = os.walk(start_dir, topdown=True, onerror=lambda err: logging.warning(f"os.walk error: {err} - Skipping directory"))
    try:
        pbar_scan = tqdm(walker, desc="Scanning Dirs", unit="dir", leave=False)
        for root, dirs, files in pbar_scan:
            abs_root = os.path.abspath(root)

            # --- Device Check ---
            if args.stay_on_device:
                try:
                    current_dev = os.stat(abs_root).st_dev
                    if current_dev != start_dev:
                        skipped_device_count += (len(dirs) + 1)
                        dirs[:] = [] # Don't descend further
                        continue
                except OSError as e:
                    initial_scan_errors += 1
                    skipped_permission_count += 1
                    dirs[:] = []
                    continue

            # --- Exclusion Check for Current Directory ---
            is_excluded_root = False
            for excluded_path_or_pattern in exclude_paths_abs:
                try:
                    path_basename = os.path.basename(abs_root)
                    # Check absolute path prefix match
                    if os.path.isabs(excluded_path_or_pattern) and abs_root.startswith(excluded_path_or_pattern):
                        is_excluded_root = True; break
                    # Check basename match for patterns like ".git"
                    if not os.path.isabs(excluded_path_or_pattern) and path_basename == excluded_path_or_pattern:
                         is_excluded_root = True; break
                except Exception as path_err:
                     initial_scan_errors += 1
                     continue # Be cautious
            if is_excluded_root:
                skipped_excluded_dir_count += (len(dirs) + 1)
                dirs[:] = [] # Don't descend further
                continue

            # --- Prune Subdirectories ---
            original_dirs = list(dirs)
            dirs[:] = []
            for d in original_dirs:
                abs_d_path = os.path.abspath(os.path.join(root, d))
                # Device check
                if args.stay_on_device:
                    try:
                        sub_dev = os.stat(abs_d_path).st_dev
                        if sub_dev != start_dev: skipped_device_count += 1; continue
                    except OSError: initial_scan_errors += 1; skipped_permission_count += 1; continue
                # Exclusion check
                is_excluded_sub = False
                dir_basename = os.path.basename(abs_d_path)
                for excluded_path_or_pattern in exclude_paths_abs:
                     try:
                        if os.path.isabs(excluded_path_or_pattern) and abs_d_path.startswith(excluded_path_or_pattern): is_excluded_sub = True; break
                        if not os.path.isabs(excluded_path_or_pattern) and dir_basename == excluded_path_or_pattern: is_excluded_sub = True; break
                     except Exception: initial_scan_errors += 1; continue
                if not is_excluded_sub: dirs.append(d)
                else: skipped_excluded_dir_count += 1


            # --- Identify Video Files and Generate Frame Times --- #
            supported_extensions = tuple(ext.lower() for ext in (".mp4", ".mov", ".avi", ".mkv", ".wmv"))
            for filename in files:
                if filename.startswith((".DS_Store")) or filename.startswith("._"): # Updated to check both
                    continue

                if filename.lower().endswith(supported_extensions):
                    found_video_count += 1
                    video_path_abs = os.path.abspath(os.path.join(root, filename))
                    videos_found_paths.append(video_path_abs)

                    # --- Determine Frame Timestamps and Categorize Task --- #
                    temp_video = None
                    frame_times_ms = []
                    task_category = None # 'avfoundation' or 'fallback' or None

                    try:
                        # 1. Try AVFoundation
                        temp_video = cv2.VideoCapture(video_path_abs, cv2.CAP_AVFOUNDATION)
                        if temp_video.isOpened():
                            task_category = 'avfoundation'
                            # logging.info(f"AVFoundation OK (Scan) | {video_path_abs}") # Optional: Debug log
                        else:
                            # 2. AVFoundation failed, try default backend
                            temp_video.release() # Release failed attempt
                            logging.info(f"AVFoundationFailed (Scan) | {video_path_abs} | Retrying with default.") # Log once on failure
                            temp_video = cv2.VideoCapture(video_path_abs)
                            if temp_video.isOpened():
                                task_category = 'fallback'
                                # logging.info(f"Fallback OK (Scan) | {video_path_abs}") # Optional: Debug log
                            else:
                                # 3. Both failed
                                logging.warning(f"Could not open video (Scan - Both Failed): {video_path_abs}")
                                initial_scan_errors += 1
                                task_category = None

                        # 4. If opened (either way), get metadata
                        if task_category:
                            fps = temp_video.get(cv2.CAP_PROP_FPS)
                            total_frames = int(temp_video.get(cv2.CAP_PROP_FRAME_COUNT))
                            if fps > 0 and total_frames > 0:
                                duration_sec = total_frames / fps
                                duration_ms = duration_sec * 1000
                                frame_interval_ms = frame_interval * 1000.0
                                current_frame_time_ms = 0.0
                                while current_frame_time_ms < duration_ms:
                                    frame_times_ms.append(current_frame_time_ms)
                                    current_frame_time_ms += frame_interval_ms
                            else:
                                logging.warning(f"Could not get valid FPS/FrameCount ({fps=}, {total_frames=}) for {video_path_abs} (Category: {task_category})")
                                initial_scan_errors += 1
                                frame_times_ms = [] # Ensure task is not added if metadata fails

                    except Exception as e:
                        logging.warning(f"Error during scan pre-check for {video_path_abs}: {e}", exc_info=False) # Keep log concise
                        initial_scan_errors += 1
                        task_category = None
                        frame_times_ms = [] # Ensure task is not added on exception
                    finally:
                        if temp_video is not None and temp_video.isOpened():
                            temp_video.release()

                    # Add task to the correct list if frames were generated
                    if frame_times_ms and task_category:
                        task_data = (video_path_abs, frame_times_ms, output_dir_base, task_category)
                        if task_category == 'avfoundation':
                            av_tasks.append(task_data)
                            categorized_av += 1
                        elif task_category == 'fallback':
                            fallback_tasks.append(task_data)
                            categorized_fallback += 1
    finally:
        pbar_scan.close() # Ensure tqdm closes

    scan_end_time = time.time()
    # Calculate total frames planned *after* categorization
    total_av_frames = sum(len(times) for _, times, _, _ in av_tasks)
    total_fallback_frames = sum(len(times) for _, times, _, _ in fallback_tasks)
    total_frames_planned = total_av_frames + total_fallback_frames
    total_tasks = len(av_tasks) + len(fallback_tasks)

    logging.info(f"Initial scan complete. Found {found_video_count} videos.")
    logging.info(f"Categorized tasks: AVFoundation={len(av_tasks)} ({total_av_frames} frames), Fallback={len(fallback_tasks)} ({total_fallback_frames} frames).")
    logging.info(f"Total tasks generated: {total_tasks} with {total_frames_planned} total frames planned.")
    logging.info(f"Skipped Dirs: {skipped_excluded_dir_count} excluded, {skipped_permission_count} permission, {skipped_device_count} device.")
    logging.info(f"Initial scan errors (open/metadata): {initial_scan_errors}")
    logging.info(f"Scan and task generation took: {scan_end_time - scan_start_time:.2f} seconds")

    if total_tasks == 0:
        logging.info("No video tasks generated. Exiting.")
        sys.exit(0)

    # ====== Process Video Tasks using Multiprocessing Pools ======
    logging.info(f"Extracting {total_frames_planned} frames using {num_av_workers} AV workers and {num_fallback_workers} Fallback workers...")
    extraction_start_time = time.time()

    all_extracted_frame_info = []
    pool_av = None
    pool_fallback = None
    processed_video_count = 0
    total_extracted_count = 0
    total_failed_count = 0 # Will calculate at the end
    all_async_results = [] # List to store (planned_frames, AsyncResult) tuples

    try:
        # --- Create Pools ---
        if num_av_workers > 0:
            pool_av = mp.Pool(processes=num_av_workers, maxtasksperchild=10)
            logging.info(f"AVFoundation Pool created with {num_av_workers} workers.")
        if num_fallback_workers > 0:
            pool_fallback = mp.Pool(processes=num_fallback_workers, maxtasksperchild=10)
            logging.info(f"Fallback Pool created with {num_fallback_workers} workers.")

        # --- Submit Tasks Asynchronously ---
        logging.info("Submitting tasks to worker pools...")
        if pool_av:
            for task_data in av_tasks:
                video_path, frame_times, _, category = task_data
                planned_frames = len(frame_times)
                res = pool_av.apply_async(process_video_extract_task, (task_data,))
                all_async_results.append((planned_frames, res))
        if pool_fallback:
            for task_data in fallback_tasks:
                 video_path, frame_times, _, category = task_data
                 planned_frames = len(frame_times)
                 res = pool_fallback.apply_async(process_video_extract_task, (task_data,))
                 all_async_results.append((planned_frames, res))

        # --- Close Pools to prevent new tasks ---
        if pool_av: pool_av.close()
        if pool_fallback: pool_fallback.close()
        logging.info("Task submission complete. Waiting for results...")

        # --- Collect Results ---
        pbar_extract = tqdm(total=total_frames_planned, desc="Extr: 0/0 videos", unit="frame", leave=True)
        completed_task_indices = set()

        while len(completed_task_indices) < len(all_async_results):
            check_start_time = time.time()
            found_ready = False
            for i, (planned_frames, result) in enumerate(all_async_results):
                if i in completed_task_indices:
                    continue # Already processed

                if result is not None and result.ready():
                    found_ready = True
                    try:
                        # Worker returns (total_frames_in_task, saved_frame_info_list)
                        _, video_result_info_list = result.get() # Ignore planned frames from result
                        if video_result_info_list:
                            all_extracted_frame_info.extend(video_result_info_list)
                            total_extracted_count += len(video_result_info_list)
                    except Exception as e:
                        logging.error(f"Error getting result for task index {i}: {e}", exc_info=True)
                        # Note: Can't easily get video path here without storing it with AsyncResult
                        # Consider adding video_path to the tuple stored in all_async_results if needed for error logs

                    # Mark as complete regardless of success/failure to prevent reprocessing
                    completed_task_indices.add(i)
                    processed_video_count += 1
                    # Update progress bar for this task
                    pbar_extract.update(planned_frames)
                    pbar_extract.set_description(f"Extr: {processed_video_count}/{total_tasks} videos")

            # Avoid busy-waiting if no tasks were ready in this pass
            if not found_ready:
                time.sleep(0.05) # Short sleep

    except Exception as pool_err:
        logging.error(f"Error during multiprocessing pool execution: {pool_err}", exc_info=True)
        traceback.print_exc()
    finally:
        if 'pbar_extract' in locals() and pbar_extract:
            pbar_extract.close() # Ensure progress bar closes
        # --- Join Pools ---
        if pool_av:
            logging.info("Joining AVFoundation worker pool...")
            pool_av.join()
            logging.info("AVFoundation worker pool joined.")
        if pool_fallback:
            logging.info("Joining Fallback worker pool...")
            pool_fallback.join()
            logging.info("Fallback worker pool joined.")

    extraction_end_time = time.time()
    total_script_time = extraction_end_time - scan_start_time

    # Calculate final failed count based on difference
    total_failed_count = total_frames_planned - total_extracted_count

    # ====== Save Manifest ======
    logging.info(f"Extraction phase complete. Successfully extracted {total_extracted_count} frames.")
    manifest_data = {
        "start_dir": start_dir,
        "frame_interval": frame_interval,
        "scan_time_sec": scan_end_time - scan_start_time,
        "extraction_time_sec": extraction_end_time - extraction_start_time,
        "total_videos_found": found_video_count,
        "total_videos_categorized": total_tasks, # Videos we attempted to process
        "av_tasks": len(av_tasks),
        "fallback_tasks": len(fallback_tasks),
        "total_frames_planned": total_frames_planned,
        "total_frames_extracted": total_extracted_count,
        "total_frames_failed": total_failed_count, # Based on planned vs extracted
        "extracted_frames": all_extracted_frame_info # Use the list of dicts
    }
    manifest_path = os.path.join(output_dir_base, MANIFEST_FILENAME)
    logging.info(f"Saving extraction manifest to {manifest_path}...")
    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving manifest file: {e}", exc_info=True)

    # ====== Print Summary ======
    logging.info(f"===== Frame Extraction Complete =====")
    logging.info(f"Total script execution time: {total_script_time:.2f} seconds")
    logging.info(f"Results saved in: {output_dir_base}")
    logging.info(f"Manifest file: {manifest_path}")
    logging.info("Run frame_indexer.py to process the extracted frames.") 