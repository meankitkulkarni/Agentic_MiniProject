# Web Research & Summarization Agent (Ollama)

Local agent: plans queries -> searches the web -> reads pages -> summarizes each source -> reflects on gaps -> writes a cited report.

## 1. Prerequisites
- Python 3.10+
- Ollama: https://ollama.com/download

## 2. Setup
```bash
# Start Ollama (skip if the desktop app is already running)
ollama serve

# Pull a model (new terminal)
ollama pull llama3.1:8b

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run

**Web UI**
```bash
streamlit run app.py
```
Opens at http://localhost:8501. Pick a model, enter a question, click Research.

**CLI**
```bash
python agent.py "What are the latest advances in solid-state batteries?" --out report.md
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` | llama3.1:8b | Any pulled Ollama model |
| `--queries` | 3 | Search queries per round |
| `--results` | 4 | Results per query |
| `--rounds` | 2 | Max research rounds |
| `--out` | - | Save report to a .md file |

## 4. Troubleshooting
- `Connection refused`: Ollama isn't running; run `ollama serve`.
- `model not found`: run `ollama pull <model>`.
- "No usable sources found": DuckDuckGo may be rate-limiting; wait a minute or rephrase.
- Slow runs: use a smaller model or lower `--queries` / `--results`.
- Weak planning or reflection: try `qwen2.5:7b` or a larger model.
