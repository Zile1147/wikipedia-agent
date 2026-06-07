# WikiPedia-Agent-Local
|Offline Wikipedia Agent  Q&amp;A from `.zim` archives (RAG) system  using Llama.cpp|
<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/7a950e0b-dafe-4e05-a298-3c8c2d21763a" />


# WikipediaAgent
<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/c340e53b-b8e3-4622-94ac-b5fc2ae5feb5" />
**WikipediaAgent** lets you ask questions about Wikipedia completely offline. It uses a **ZIM** archive (the same format used by Kiwix) and builds a hybrid keyword + semantic index with **TurboRag**. No internet connection is needed after setup – perfect for planes, remote areas, or privacy‑sensitive environments.

---

## Features

- Works with any Wikipedia ZIM file (en, mini, medical, etc.)
- Builds a hybrid index: TF‑IDF keyword search + TurboVec dense vectors
- Configurable max articles for testing (e.g., 5000 articles with `--max-articles`)
- Interactive chat, single‑question mode, or index‑only mode
- Uses quantized Gemma embedding (≈150 MB) and a tiny LLM (Qwen 0.5B, ≈300 MB) – runs on low‑CPU, low‑RAM devices
- Fully offline – no phoning home

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/AHX47/wikipedia-agent.git
cd wikipedia-agent
pip install -r requirements.txt
```

### 2. Install TurboRag (dependency)

```bash
pip install turborag-ahx47   # or your published turborag package
```

### 3. Download a Wikipedia ZIM file

```bash
mkdir -p data
# Mini (~90 MB) – perfect for testing
wget https://download.kiwix.org/zim/wikipedia/wikipedia_en_top_mini_2024-12.zim -O data/wikipedia_en_mini.zim
```

### 4. Download the models

```bash
mkdir -p models
# Embedding model (≈150 MB)
wget -O models/embeddinggemma-300m-q4_k_m.gguf \
  "https://huggingface.co/sabafallah/embeddinggemma-300m-Q4_K_M-GGUF/resolve/main/embeddinggemma-300m-q4_k_m.gguf"

# LLM model (Qwen 0.5B, ≈300 MB)
wget -O models/qwen-0.5b-q4_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen-0.5B-GGUF/resolve/main/qwen-0.5b-q4_k_m.gguf"
```

### 5. Build the index

```bash
python main.py index --zim data/wikipedia_en_mini.zim --max-articles 5000
```

The index will be stored in `data/wikipedia_index/`.

---

## Usage

### Ask a single question

```bash
python main.py ask "Who invented the telephone?"
```

**Output example:**
```
Answer: The telephone was invented by Alexander Graham Bell in 1876.

Sources:
- Alexander Graham Bell (article: A/Alexander_Graham_Bell)
- History of the telephone (article: H/History_of_the_telephone)
```

### Interactive chat

```bash
python main.py chat
```

```
WikipediaAgent> What is the capital of France?
Agent: The capital of France is Paris.
WikipediaAgent> How big is it?
Agent: Paris has an area of about 105 square kilometres.
```

### Re‑index with different settings

```bash
python main.py index --zim data/wikipedia_en_mini.zim --max-articles 10000 --force
```

### Search only (no LLM generation)

```bash
python main.py search "quantum computing" --k 10
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     WikipediaAgent                         │
├────────────────────────────────────────────────────────────┤
│  CLI (ask / chat / index / search)                         │
├────────────────────────────────────────────────────────────┤
│  Core components:                                          │
│  ┌──────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │ ZIM      │  │ Chunker     │  │ SQLite (docstore)   │   │
│  │ Reader   │→│ (overlap    │→│ + FTS5 (keyword)     │   │
│  │ (zimply) │  │  512/50)    │  └─────────────────────┘   │
│  └──────────┘  └─────────────┘  ┌─────────────────────┐   │
│                                 │ TurboVec Q4 Index   │   │
│  ┌──────────┐  ┌─────────────┐  │ (semantic vectors)  │   │
│  │ Gemma    │→│ Embeddings  │→│                     │   │
│  │ 300M Q4  │  │ (2048-dim)  │  └─────────────────────┘   │
│  └──────────┘  └─────────────┘                            │
│                                                           │
│  ┌──────────┐  ┌─────────────┐                           │
│  │ Qwen     │←│ Hybrid      │                           │
│  │ 0.5B LLM │  │ Retriever   │                           │
│  └──────────┘  └─────────────┘                           │
└────────────────────────────────────────────────────────────┘
```

---

## Configuration

Create a `config.yaml` (or edit the defaults in `main.py`):

```yaml
embed_model: "models/embeddinggemma-300m-q4_k_m.gguf"
llm_model: "models/qwen-0.5b-q4_k_m.gguf"
chunk_size: 512
chunk_overlap: 50
max_articles: 5000
index_path: "data/wikipedia_index"
zim_path: "data/wikipedia_en_mini.zim"
```

---

## Requirements

- Python 3.10+
- Rust (only needed if you rebuild TurboVec – not required if using `pip install turborag-ahx47`)
- ~1 GB RAM (2 GB recommended for larger indexes)
- ~2 GB disk space (models + index + ZIM)
- No internet required at runtime

---

## Installation from Source (without PyPI)

If you prefer to build everything from source:

```bash
git clone https://github.com/AHX47/wikipedia-agent.git
cd wikipedia-agent
pip install -r requirements.txt
pip install -e .
```

Then follow steps 3–5 above.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `illegal hardware instruction` | Your CPU lacks AVX2. Reinstall `llama-cpp-python` with `CMAKE_ARGS="-DLLAMA_AVX2=OFF" pip install llama-cpp-python --force-reinstall` |
| `IndexError: list index out of range` | The ZIM may be empty or corrupted. Try another ZIM file. |
| Slow indexing | Reduce `max_articles` or use a smaller ZIM. |

---

## License

MIT

---

## Links

- **GitHub**: [AHX47/wikipedia-agent](https://github.com/AHX47/wikipedia-agent)
- **Related**: [turborag-ahx47](https://pypi.org/project/turborag-ahx47/), [zim-agent](https://github.com/AHX47/zim-agent)
```

