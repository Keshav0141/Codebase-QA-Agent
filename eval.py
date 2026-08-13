import json
import argparse
from langchain_chroma import Chroma
from config import get_embeddings, store_dir


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_hit(doc, expected_file, expected_line, tol=3):
    m = doc.metadata
    return (
        m.get("filepath") == expected_file
        and abs(m.get("start_line", 0) - expected_line) <= tol
    )


def main():
    parser = argparse.ArgumentParser(description="measure retrieval hit rate on hand-labeled questions")
    parser.add_argument("--k", type=int, default=6, help="number of chunks retrieved per question")
    parser.add_argument("--questions", default="eval_questions.json", help="json file with the eval set")
    args = parser.parse_args()

    store = Chroma(embedding_function=get_embeddings(), persist_directory=store_dir())
    retriever = store.as_retriever(search_kwargs={"k": args.k})
    questions = load_questions(args.questions)

    hits = 0
    for q in questions:
        docs = retriever.invoke(q["question"])
        ok = False
        for d in docs:
            if is_hit(d, q["file"], q["line"]):
                ok = True
                break
        hits += 1 if ok else 0
        top = []
        for d in docs:
            top.append(f"{d.metadata.get('filepath')}:{d.metadata.get('start_line')}")
        print(f"[{'HIT ' if ok else 'MISS'}] {q['question']}")
        if not ok:
            print(f"      expected {q['file']}:{q['line']}   got {top}")

    total = len(questions)
    print("")
    print(f"hit rate at k={args.k}: {hits}/{total} = {hits / total:.2%}")


if __name__ == "__main__":
    main()