#!/bin/bash

# clear_index.sh - Completely clear all video indexing data
# This script removes all generated indexes, thumbnails, frame images, 
# transcripts, metadata, and logs to start fresh.

echo "🧹 Clearing Video Index Data..."
echo "================================"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if virtual environment exists
VENV_PATH="$SCRIPT_DIR/videoindexerenv"
if [ -d "$VENV_PATH" ]; then
    echo "🐍 Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
    echo "     ✅ Virtual environment activated"
    echo ""
else
    echo "💡 Note: Virtual environment not found at $VENV_PATH"
    echo "   The launcher will create one automatically when needed"
    echo "   Proceeding with system Python for now"
    echo ""
fi

# Function to safely remove directory contents
clear_directory() {
    local dir_path="$1"
    local dir_name="$2"
    
    if [ -d "$dir_path" ]; then
        echo "  📁 Clearing $dir_name..."
        rm -rf "$dir_path"/*
        echo "     ✅ $dir_name cleared"
    else
        echo "     ⚠️  $dir_name directory not found"
    fi
}

# Function to safely remove specific files
remove_file() {
    local file_path="$1"
    local file_name="$2"
    
    if [ -f "$file_path" ]; then
        echo "  🗑️  Removing $file_name..."
        rm -f "$file_path"
        echo "     ✅ $file_name removed"
    else
        echo "     ⚠️  $file_name not found"
    fi
}

echo ""
echo "🖼️  Clearing Thumbnails..."
clear_directory "video_thumbnails_output" "thumbnails"

echo ""
echo "📊 Clearing Video Index Data..."
clear_directory "video_index_output/frame_images" "frame images"
clear_directory "video_index_output/indexer_logs" "indexer logs"
clear_directory "video_index_output/extractor_logs" "extractor logs"

# Remove FAISS index files
remove_file "video_index_output/video_index.faiss" "video FAISS index"
remove_file "video_index_output/video_frame_metadata.json" "video frame metadata"
remove_file "video_index_output/index_config.json" "index config"
remove_file "video_index_output/extraction_manifest.json" "extraction manifest"

echo ""
echo "🎙️  Clearing Transcript Data..."
clear_directory "video_index_output/transcripts" "transcript files"
clear_directory "video_index_output/transcript_indexer_logs" "transcript logs"

# Remove transcript index files
remove_file "video_index_output/transcript_index.faiss" "transcript FAISS index"
remove_file "video_index_output/transcript_metadata.json" "transcript metadata"
remove_file "video_index_output/transcript_index_config.json" "transcript index config"
remove_file "video_index_output/transcript_manifest.json" "transcript manifest"

echo ""
echo "🧽 Clearing Temporary Files..."
clear_directory "temp_uploads" "temporary uploads"

# Remove any .DS_Store files
find . -name ".DS_Store" -type f -delete 2>/dev/null

echo ""
echo "✨ Index clearing complete!"
echo ""
echo "📋 Summary:"
echo "  • All thumbnails cleared"
echo "  • All frame images and indexes cleared"  
echo "  • All transcript data cleared"
echo "  • All metadata and manifests cleared"
echo "  • All logs cleared"
echo "  • Temporary files cleared"
echo ""
echo "🚀 Ready for fresh indexing!" 