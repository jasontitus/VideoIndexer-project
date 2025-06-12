# Video Indexer with Qwen3 Embeddings

A video search and indexing tool that uses Qwen3-Embedding-8B for semantic search over video frames and transcripts.

## Features

- Visual search using text or image queries
- Transcript search with both exact and semantic matching
- Automatic video transcription
- Frame extraction and indexing
- Modern web interface with instant results
- Video playback starting at matched frames
- M3U playlist generation for search results

## Requirements

- Python 3.12+
- FFmpeg (for video processing)
- Conda or Miniconda (recommended for environment management)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/VideoIndexer-project-qwen3.git
cd VideoIndexer-project-qwen3
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

1. Process videos to build the search index:
```bash
cd src
python frame_extractor.py /path/to/your/videos --output-dir video_index_output
python frame_indexer.py --input-dir video_index_output
```

2. Start the search server:
```bash
cd src
python video_search_server_transcode.py
```

3. Open http://localhost:8002 in your browser

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

- `index_config.json`: Configure model and embedding settings
- `transcript_index_config.json`: Configure transcript search settings
- Command line arguments for frame extraction and indexing (see --help)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 