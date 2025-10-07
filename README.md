# AI-Powered Meeting Summarizer

Small, Windows-friendly CLI tool to transcribe meeting audio to text and produce a consolidated Markdown meeting summary.

Key features
- CPU-friendly transcription using faster-whisper
- Map→reduce chunked summarization using a local Ollama model (Markdown output only)
- Simple sentence-aware chunking to avoid mid-sentence truncation

Files
- `meeting summarizer.py` - main script (CLI)
- `requirements.txt` - Python package requirements

Prerequisites
- Python 3.10+ (3.11 recommended)
- ffmpeg available on PATH (for audio decoding)
- Ollama CLI installed and at least one local model pulled (see https://ollama.com/)

Installation
1. Create and activate a virtual environment:

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2. Install Python dependencies:

   pip install -r requirements.txt

3. Install platform binaries:
- ffmpeg: https://ffmpeg.org/download.html
- Ollama: https://ollama.com/

Usage
1. Transcribe and summarize a meeting audio file (wav/mp3/m4a/ogg):

   python "meeting summarizer.py" "path/to/meeting.mp3"

2. Common options:
- `--whisper`: faster-whisper model size (tiny|tiny.en|base|small|medium). Default: tiny
- `--whisper-compute-type`: int8, int8_float16, int16, float16, float32. Default: int8
- `--ollama-model`: local Ollama model name (default: `qwen3:4b`)
- `--outdir`: output folder (default: `out`)
- `--max-chars`: chunk character budget (default: 2400)
- `--timeout`: per-chunk Ollama timeout in seconds (default: 600)

Outputs
- A transcript file at `<outdir>/<audio-stem>.transcript.txt`
- A Markdown summary at `<outdir>/<audio-stem>.md`

Notes and troubleshooting
- Ollama is required for summarization. If the script reports "Ollama not found", install Ollama and pull a local model (for example `ollama pull qwen3:4b`).
- If `faster-whisper` fails to load certain compute types on CPU (e.g., float16), try `--whisper-compute-type int8`.
- This project intentionally calls the Ollama CLI; the Ollama binary must be available on PATH and run locally.

License
MIT
