#!/usr/bin/env python3
"""
Windows-friendly meeting summarizer (CPU) with map→reduce consolidation.
- ASR: faster-whisper
- Summarization: Ollama (Markdown only)
- Chunking: sentence-aware
- Final pass: global reduction to produce one coherent Markdown summary
"""
import argparse, re, shutil, subprocess, sys
from pathlib import Path

# ---------- Third-party ----------
from faster_whisper import WhisperModel
#Iam happy

# ---------- Utils ----------
def have_ollama() -> bool:
    return shutil.which("ollama") is not None

def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)

def chunk_text(t: str, max_chars: int = 2400):
    """Sentence-aware chunking to reduce mid-sentence/JSON breakage."""
    sentences = re.split(r'(?<=[.!?])\s+', t.strip())
    buf = ""
    for s in sentences:
        if not s:
            continue
        if len(buf) + len(s) + 1 > max_chars and buf:
            yield buf
            buf = s
        else:
            buf = (buf + " " + s).strip()
    if buf:
        yield buf

PROMPT_MD = """I have provide you with a chunk of a meeting transcript.
Summarize this part of the meeting with:
- TLDR (2–3 lines)
- Key points (5–8 bullets)
- Action items (owner, task, due if mentioned)

Dont show thinking steps.
Text:
{chunk}
"""

PROMPT_REDUCE_MD = """I have provided you with multiple summaries of chunks of a meeting transcript.
Create ONE consolidated meeting summary:
- TLDR (2–3 lines)
- Key points (5–8 bullets, deduped)
- Action items (owner, task, due if any)

Dont show Thinking.
Chunk summaries to merge:
{chunk_summaries}"""

# ---------- ASR ----------
def transcribe(path: str, model_size: str = "tiny", compute_type: str = "int8") -> str:
    try:
        model = WhisperModel(model_size, compute_type=compute_type, device="cpu")
    except Exception as e:
        raise SystemExit(f"Failed to load faster-whisper model '{model_size}' with compute_type='{compute_type}': {e}")
    try:
        segments, _info = model.transcribe(
            path,
            vad_filter=True,
            beam_size=1,  # greedy
        )
    except Exception as e:
        raise SystemExit(f"Transcription failed: {e}")
    return "\n ".join(seg.text.strip() for seg in segments).strip()

# ---------- Ollama ----------
def ollama_run(model: str, prompt: str, timeout: int = 600) -> str:
    try:
        res = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=timeout,
        )
        return res.stdout.decode("utf-8", "ignore").strip()
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode("utf-8", "ignore").strip()
        raise RuntimeError(f"Ollama failed (model='{model}'): {msg or e}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Ollama timed out after {timeout}s (model='{model}').")

def reduce_with_ollama_markdown(summaries: list[str], model: str, timeout: int = 600) -> str:
    text = "\n\n---\n\n".join(summaries)
    prompt = PROMPT_REDUCE_MD.format(chunk_summaries=text)
    out = ollama_run(model, prompt, timeout=timeout)
    return out.strip()

def summarize_with_ollama_markdown(text: str, model: str = "qwen3:4b", max_chars: int = 2400, timeout: int = 600) -> str:
    """Map→reduce summarization with Ollama (Markdown only)."""
    outputs = []
    for ch in chunk_text(text, max_chars=max_chars):
        prompt = PROMPT_MD.format(chunk=ch)
        outputs.append(ollama_run(model, prompt, timeout=timeout))

    try:
        return reduce_with_ollama_markdown(outputs, model=model, timeout=timeout)
    except Exception:
        return "\n\n".join(outputs)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Windows-friendly meeting summarizer (CPU, Markdown only)")
    ap.add_argument("audio", help="Path to meeting audio (wav/mp3/m4a/ogg)")
    ap.add_argument("--whisper", default="tiny", help="faster-whisper size: tiny|tiny.en|base|small|medium")
    ap.add_argument("--whisper-compute-type", default="int8", help="faster-whisper compute type (e.g., int8, int8_float16, int16, float16, float32)")
    ap.add_argument("--ollama-model", default="qwen3:4b", help="Ollama model name (must exist locally)")
    ap.add_argument("--outdir", default="out", help="Output folder")
    ap.add_argument("--max-chars", type=int, default=2400, help="Approximate per-chunk character budget")
    ap.add_argument("--timeout", type=int, default=600, help="Per-chunk Ollama timeout (seconds)")
    ap.add_argument("--plain", action="store_true", help="Plain ASCII console messages (avoid Unicode glyphs)")
    args = ap.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    print("Transcribing..." if args.plain else "Transcribing… (CPU, can be slow)")
    transcript = transcribe(str(audio_path), model_size=args.whisper, compute_type=args.whisper_compute_type)
    (Path(args.outdir) / f"{stem}.transcript.txt").write_text(transcript, encoding="utf-8")
    print("Transcription saved.")

    if not have_ollama():
        raise SystemExit("Ollama not found. Please install Ollama and pull a local model.")

    print("Summarizing with Ollama (Markdown)…")
    md_text = summarize_with_ollama_markdown(
        transcript,
        model=args.ollama_model,
        max_chars=args.max_chars,
        timeout=args.timeout,
    )
    md_final = re.sub(r"(?s)^Thinking\.\.\..*?\.{3}done thinking\.\n*", "", md_text) #Remove Text from Thinking... to ...done thinking

    (Path(args.outdir) / f"{stem}.md").write_text(md_final, encoding="utf-8")
    print("Markdown saved.")

if __name__ == "__main__":
    main()
