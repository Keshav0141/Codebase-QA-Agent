# Codebase Q&A

CLI + FastAPI tool that ingests a local git repo, lets you ask natural-language
questions about the code, and returns grounded answers with `file:start-end`
citations. Built with LangChain, Chroma (local), sentence-transformers, and
DeepSeek via the OpenCode Zen API.

## Stack

- LangChain chains (`create_retrieval_chain` + `create_stuff_documents_chain`)
- Vector store: Chroma, persisted locally in `chroma/`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, free).
  Set `EMBEDDING_PROVIDER=zen` to use OpenAI-compatible embeddings over the
  Zen API instead (see `.env.example`).
- LLM: DeepSeek `deepseek-v4-flash-free` through OpenCode Zen
  (`https://opencode.ai/zen/v1`), loaded with LangChain's `ChatOpenAI`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your key. `.env` is gitignored.

```
OPENCODE_ZEN_API_KEY=<your key>
EMBEDDING_PROVIDER=local
```

## Ingest a repo

```powershell
python ingest.py --repo C:\path\to\repo
python ingest.py --repo . --clear   # wipe store, then re-ingest
```

Walk skips `.git`, `node_modules`, `__pycache__`, `venv`, `dist`, etc. Supported
languages: python, js/ts, go, rust, java, c/c++, ruby.

Chunking is per function/class where possible: definition lines are detected per
language, each def becomes a segment (with its `def`/`class` header line
prepended so method chunks carry class context), and oversized segments are
split with `RecursiveCharacterTextSplitter` using code-aware separators.
Metadata per chunk: `filepath`, `start_line`, `end_line`, `name`, `lang`.
Example: `gpt_from_scratch/gpt_model.py:34-48` symbol `forward`.

## Ask questions

CLI (single shot):

```powershell
python query.py "how does causal self attention work and where is it"
```

CLI (interactive loop): `python query.py` then type questions; `exit` to quit.

```powershell
python query.py --k 8 "question"    # how many chunks to retrieve
```

API:

```powershell
uvicorn api:app --port 8000
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" `
  -d '{"question": "how is LoRA applied?", "k": 6}'
```

Response:

```json
{
  "answer": "LoRA is injected into the attention projections of every transformer block ... (lora.py:30-45)",
  "sources": ["lora.py:30-45 (symbol: apply_lora)", "..."]
}
```

The prompt instructs the model to answer **only** from retrieved context, cite
`path:start-end` per claim, and say "I could not find an answer to this in the
codebase." rather than invent paths. `chain.ask()` retries transient LLM rate
limits (3 tries, 30s backoff).

## Eval

`eval_questions.json` holds hand-labeled questions mapping to the expected
`file` and `start_line`. The 33-question set covers two repos:
- **gpt-from-scratch** (20 questions): gpt_model.py, tokenizer.py, lora.py, etc.
- **Auto-Assessment-Agent** (13 questions): app/grading/engine.py, app/llm.py,
  app/main.py, app/ingestion/parser.py, app/confidence/scorer.py, etc.

Each expected line is verified against the actual ingested chunks.

```powershell
python eval.py --k 6
```

Metric: retrieval hit rate = fraction of questions where the expected chunk
(file + line) appeared in the top-k retrieved chunks.

Result on gpt-from-scratch (47 chunks): **19/20 = 95%** at k=6.
The single miss is an intentionally ambiguous question ("how does the GPT model
generate new tokens" could point at either `GPT.generate` in `gpt_model.py:109`
or the top-level `generate` wrapper in `generate.py`).

Answer correctness is a manual spot-check on the model output (prompt
enforces citations and honesty about missing context).

To evaluate other repos, add row-shaped questions referencing your own verified
file/line pairs; hit-rate logic lives in `eval.py` (`is_hit`, ±3 lines).

## Notes

- The free OpenCode Zen tier is heavily rate-limited; queries may return
  `429 FreeUsageLimitError` until the window resets or you use a paid key.
- Re-ingesting a repo (no `--clear`) upserts by id, so the store stays deduped.

## License

[MIT](LICENSE)