import argparse
from chain import make_chain, cite_docs, ask


def main():
    parser = argparse.ArgumentParser(description="ask a question about the ingested codebase")
    parser.add_argument("question", nargs="?", help="the question to ask; if omitted, run an interactive loop")
    parser.add_argument("--k", type=int, default=6, help="number of chunks to retrieve")
    parser.add_argument("--top", type=int, default=5, help="how many sources to print")
    args = parser.parse_args()

    chain = make_chain(k=args.k)

    def run(q):
        result = ask(chain, q)
        print("ANSWER:")
        print(result.get("answer"))
        print("")
        print("SOURCES:")
        for line in cite_docs(result.get("context", []))[: args.top]:
            print("  " + line)
        print("")

    if args.question:
        run(args.question)
    else:
        print("interactive mode. type 'exit' to quit.")
        while True:
            q = input("> ")
            if q.strip().lower() in ("exit", "quit"):
                break
            run(q)


if __name__ == "__main__":
    main()