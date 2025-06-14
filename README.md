# Video Indexer with Advanced AI Models

A powerful video search and indexing tool that combines multiple state-of-the-art AI models for comprehensive video content search:

- **Visual Search**: Meta's Perception Model (Core) for understanding visual content and cross-modal search
- **Transcript Search**: Qwen3-Embedding-8B for high-quality semantic search over spoken content (transcribed using whisper and its large v3 model)

## Demo

Watch a quick demo of the Video Indexer in action:

<video src="https://github.com/user-attachments/assets/8efd2c6c-dd3b-415f-bd5e-193adcdc5b7e" controls="controls" style="max-width: 730px;">
</video>

## Features

### Visual Content Search
- Text-to-video search using Meta's Perception Model
- Image-to-video search for finding similar visual content
- Frame extraction and intelligent indexing
- Cross-modal understanding between text and visual content
- Instant visual results with thumbnails

### Transcript Search
- Semantic search using Qwen3-Embedding-8B embeddings
- Exact text matching for precise queries
- Multi-language transcript support
- Automatic video transcription
- Time-aligned transcript results

### User Interface & Playback
- Basic Web UI
- Instant search results with visual previews
- Video playback starting at matched frames/segments
- M3U playlist generation for search results

### Technical Features
- FAISS vector similarity search
- FP16 support for efficient memory usage
- Automatic video transcoding when needed
- Configurable model parameters
- Multi-threaded processing

Quickest way to try it out is to download the VideoIndexer-mac.zip file from the Releases page and run that.  You might need to hop through security hoops to launch it despite the fact that I signed and notarized it, but hopefully that isn't too much of a pain.

You will pick the directory tree of videos you want to index and then click 'Start' and it will run for a while indexing.  If you have less than 64GB or RAM, I would check the 'fp16' box.  The accuracy should be about the same and use less RAM.  When indexing is done, you can hit the local webserver at http://127.0.0.1:8002 and search away!

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/VideoIndexer-project.git
cd VideoIndexer-project
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate video-indexer
```

3. Install additional dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Process videos to extract frames and build the visual search index:
```bash
cd src
# Extract frames (add --fp16 to reduce memory usage)
python frame_extractor.py /path/to/your/videos --output-dir video_index_output

# Build visual search index (add --fp16 for reduced memory)
python frame_indexer.py --input-dir video_index_output
```

2. Generate and index transcripts:
```bash
# Generate transcripts (add --fp16 for reduced memory)
python transcript_extractor_pywhisper.py /path/to/your/videos --output-dir video_index_output

# Build transcript search index (add --fp16 for reduced memory)
python transcript_indexer.py --input-dir video_index_output
```

3. Start the search server:
```bash
cd src
# Add --fp16 flag to reduce memory usage during search
python video_search_server_transcode.py --fp16
```

4. Open http://localhost:8002 in your browser

### Memory Usage Tips

- The `--fp16` flag can be used with most components to reduce memory usage by about 50%
- For large video collections, using FP16 is recommended
- Memory usage is highest during initial indexing and reduces for search operations
- If you encounter memory issues:
  1. Use the `--fp16` flag
  2. Process videos in smaller batches
  3. Close other memory-intensive applications

## Building macOS App

### Prerequisites

1. Copy the credentials template and fill in your Apple Developer details:
```bash
cp store_credentials_template.sh store_credentials.sh
chmod +x store_credentials.sh
# Edit store_credentials.sh with your details
./store_credentials.sh
```

2. Build the app:
```bash
./build_macos.sh
```

The built app will be in `./build/VideoIndexer.app`

## Configuration

### Model Configuration
- `index_config.json`: Configure visual search model and embedding settings
- `transcript_index_config.json`: Configure transcript search settings
- Command line arguments for frame extraction and indexing (see --help)

### Performance Tuning
- Use `--fp16` flag for reduced memory usage
- Adjust frame extraction rate for storage/accuracy tradeoff
- Configure FAISS index type for speed/accuracy tradeoff
- Set maximum result thresholds for faster searches


## License

This project is licensed under the Apache License - see the LICENSE file for details.

## Acknowledgments

- [Meta's Perception Model (Core)](https://github.com/facebookresearch/perception_models) for visual understanding
- [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) for semantic text understanding
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [FFmpeg](https://ffmpeg.org/) for video processing 
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) for multi-lingual transcription
