import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_CHAR = 1500
OVERLAP = 150
MIN_CHAR = 30


SIGNATURES = {
    "py": {
        "exts": (".py",),
        "patterns": [
            re.compile(r"^\s*(async\s+)?def\s+([A-Za-z_]\w*)"),
            re.compile(r"^\s*class\s+([A-Za-z_]\w*)"),
        ],
    },
    "js": {
        "exts": (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"),
        "patterns": [
            re.compile(r"^\s*(export\s+)?(async\s+)?function\s*([A-Za-z_$]\w*)"),
            re.compile(r"^\s*(export\s+)?class\s+([A-Za-z_$]\w*)"),
            re.compile(r"^\s*(export\s+)?(const|let|var)\s+([A-Za-z_$]\w*)\s*=\s*(async\s*)?(\(|([A-Za-z_$]\w*\s*,?\s*)+\)|[A-Za-z_$]\w*)\s*=>"),
        ],
    },
    "go": {
        "exts": (".go",),
        "patterns": [
            re.compile(r"^\s*func\s*(\([^)]*\)\s+)?([A-Za-z_]\w*)"),
            re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+(struct|interface)"),
        ],
    },
    "rs": {
        "exts": (".rs",),
        "patterns": [
            re.compile(r"^\s*(pub(\([^)]*\))?\s+)?(async\s+)?fn\s+([A-Za-z_]\w*)"),
            re.compile(r"^\s*(pub(\([^)]*\))?\s+)?(struct|enum|trait|impl)\s+([A-Za-z_]\w*)"),
        ],
    },
    "java": {
        "exts": (".java",),
        "patterns": [
            re.compile(r"^\s*(public|private|protected|abstract|final|static)*\s*(class|interface|enum)\s+([A-Za-z_]\w*)"),
            re.compile(r"^\s*(public|private|protected)\s+(static\s+)?[\w<>\[\],\s]+\s+([A-Za-z_]\w*)\s*\("),
        ],
    },
    "cc": {
        "exts": (".c", ".h", ".cpp", ".cc", ".cxx", ".hpp"),
        "patterns": [
            re.compile(r"^\s*(class|struct)\s+([A-Za-z_]\w*)"),
            re.compile(r"^\s*(static\s+)?[\w:<>\*&,\s]+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{"),
        ],
    },
    "rb": {
        "exts": (".rb",),
        "patterns": [
            re.compile(r"^\s*(def\s+([A-Za-z_]\w*)(\.|\s))"),
            re.compile(r"^\s*class\s+([A-Za-z_]\w*)"),
        ],
    },
}

SEPARATORS = {
    "py": ["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""],
    "js": ["\nasync function ", "\nfunction ", "\nclass ", "\nconst ", "\nlet ", "\nvar ", "\n\n", "\n", " ", ""],
    "go": ["\nfunc ", "\n\n", "\n", " ", ""],
    "rs": ["\nfn ", "\nimpl ", "\n\n", "\n", " ", ""],
    "java": ["\nclass ", "\ninterface ", "\npublic ", "\nprivate ", "\nprotected ", "\n\n", "\n", " ", ""],
    "cc": ["\nclass ", "\nstruct ", "\n\n", "\n", " ", ""],
    "rb": ["\ndef ", "\nclass ", "\n\n", "\n", " ", ""],
}

def split_text(text, lang):
    return RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHAR,
        chunk_overlap=OVERLAP,
        separators=SEPARATORS.get(lang) or SEPARATORS["py"],
    ).split_text(text)


def find_defs(lines, lang):
    defs = []
    for patterns in SIGNATURES.get(lang, {}).get("patterns", []):
        for i, line in enumerate(lines):
            m = patterns.match(line)
            if m:
                name = "unknown"
                for g in reversed(m.groups()):
                    if g:
                        name = g
                        break
                defs.append({"line": i + 1, "name": name, "indent": len(line) - len(line.lstrip()), "line_text": line})
    defs.sort(key=lambda d: d["line"])
    merged = []
    for d in defs:
        if merged and merged[-1]["line"] == d["line"]:
            merged[-1]["name"] = d["name"]
            merged[-1]["indent"] = min(merged[-1]["indent"], d["indent"])
        else:
            merged.append(d)
    return merged


def parent_header(defs, d):
    parent = None
    for prev in defs:
        if prev["line"] >= d["line"]:
            break
        if prev["indent"] < d["indent"]:
            parent = prev
    if parent is None:
        return ""
    return parent["line_text"].strip() + "\n"


def chunk_file(filepath, repo_root, lang):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if not text.strip():
        return []
    rel = os.path.relpath(filepath, repo_root).replace(os.sep, "/")
    lines = text.split("\n")
    defs = find_defs(lines, lang)
    segments = []
    if not defs:
        segments.append({"name": os.path.basename(filepath), "start": 1, "end": len(lines), "def": None})
    else:
        if defs[0]["line"] > 1:
            segments.append({"name": os.path.basename(filepath), "start": 1, "end": defs[0]["line"] - 1, "def": None})
        for k, d in enumerate(defs):
            start = d["line"]
            end = defs[k + 1]["line"] - 1 if k + 1 < len(defs) else len(lines)
            segments.append({"name": d["name"], "start": start, "end": end, "def": d})
    chunks = []
    for seg in segments:
        seg_lines = lines[seg["start"] - 1:seg["end"]]
        seg_text = "\n".join(seg_lines)
        prefix = ""
        if seg["def"] is not None and seg["def"]["indent"] > 0:
            prefix = parent_header(defs, seg["def"])
            seg_text = prefix + seg_text
        if len(seg_text) <= MAX_CHAR:
            chunks.append({"text": seg_text, "meta": meta(rel, seg["start"], seg["end"], seg["name"], lang)})
        else:
            parts = split_text(seg_text, lang)
            for part in parts:
                if len(part) < MIN_CHAR:
                    continue
                extra = part.count("\n")
                chunks.append({
                    "text": part,
                    "meta": meta(rel, seg["start"], min(seg["start"] + extra, seg["end"]), seg["name"], lang),
                })
    filtered = []
    for c in chunks:
        if len(c["text"]) >= MIN_CHAR:
            filtered.append(c)
    return filtered


def meta(filepath, start, end, name, lang):
    return {"filepath": filepath, "start_line": start, "end_line": end, "name": name, "lang": lang}