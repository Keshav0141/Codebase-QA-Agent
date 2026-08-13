import os
import argparse
import re
from langchain_chroma import Chroma
from config import get_embeddings, EMBED_PATH, store_dir
from chunker import chunk_file, SIGNATURES

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "venv", ".venv", "dist",
    "build", ".next", "out", ".idea", ".vscode", ".pytest_cache",
    ".mypy_cache", ".tox", "__MACOSX", ".ruff_cache",
}


def collect_files(repo_root):
    exts = set()
    for spec in SIGNATURES.values():
        for e in spec["exts"]:
            exts.add(e)
    files = []
    for root, dirs, names in os.walk(repo_root):
        pruned = []
        for d in dirs:
            if d not in SKIP_DIRS:
                pruned.append(d)
        dirs[:] = pruned
        for name in names:
            if os.path.splitext(name)[1].lower() in exts:
                files.append(os.path.join(root, name))
    return sorted(files)


def lang_for(filepath):
    for lang, spec in SIGNATURES.items():
        if os.path.splitext(filepath)[1].lower() in spec["exts"]:
            return lang
    return None


def check_env():
    for key in ("OPENCODE_ZEN_API_KEY",):
        if not os.environ.get(key):
            raise SystemExit(f"missing env var {key}; copy .env.example to .env and fill it in")


def main():
    parser = argparse.ArgumentParser(description="ingest a repo into the chroma store")
    parser.add_argument("--repo", default=".", help="path to the repo to ingest")
    parser.add_argument("--store", default=None, help="store name (default: repo folder name)")
    parser.add_argument("--clear", action="store_true", help="wipe the store before ingesting")
    parser.add_argument("--limit", type=int, default=0, help="only ingest first N files (debug)")
    args = parser.parse_args()

    check_env()
    repo_root = os.path.abspath(args.repo)
    store_name = args.store or re.sub(r"[^a-zA-Z0-9_.-]", "_", os.path.basename(repo_root) or "default")
    os.environ["CQA_STORE"] = store_name
    store_path = store_dir()

    if args.clear and os.path.isdir(store_path):
        import shutil

        shutil.rmtree(store_path)

    embed_fn = get_embeddings()
    store = Chroma(embedding_function=embed_fn, persist_directory=store_path)

    files = collect_files(repo_root)[: args.limit] if args.limit else collect_files(repo_root)

    total_chunks = 0
    for filepath in files:
        lang = lang_for(filepath)
        chunks = chunk_file(filepath, repo_root, lang)
        if not chunks:
            continue
        texts = []
        metadatas = []
        for c in chunks:
            texts.append(c["text"])
            metadatas.append(c["meta"])
        rel = os.path.relpath(filepath, repo_root).replace(os.sep, "/")
        ids = []
        for i, m in enumerate(metadatas):
            ids.append(f"{rel}:{m['start_line']}:{m['end_line']}:{i}")
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        total_chunks += len(chunks)
        print(f"{rel}: {len(chunks)} chunks")

    print("")
    print(f"ingested {len(files)} files -> {total_chunks} chunks into {store_path} (store: {store_name})")


if __name__ == "__main__":
    main()