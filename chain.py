import time
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from config import get_llm, get_embeddings, store_dir

SYSTEM = """You are a code assistant answering questions about a codebase.
Answer using ONLY the context given below. Each context block starts with
file: <path>, lines: <start>-<end>, symbol: <name>.
Use citations of the form (path:start-end) after each claim you make.
If the context does not support an answer, say exactly:
"I could not find an answer to this in the codebase."
Never invent file paths or line numbers. Be concise."""

USER = """Context:
{context}

Question: {input}"""

DOC_PROMPT = ChatPromptTemplate.from_template(
    "file: {filepath}\nlines: {start_line}-{end_line}\nsymbol: {name}\n\n{page_content}"
)


def make_chain(k=6):
    store = Chroma(embedding_function=get_embeddings(), persist_directory=store_dir())
    retriever = store.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", USER)])
    llm = get_llm()
    combine = create_stuff_documents_chain(
        llm,
        prompt,
        document_prompt=DOC_PROMPT,
        document_variable_name="context",
    )
    return create_retrieval_chain(retriever, combine)


def cite_docs(docs):
    seen = set()
    out = []
    for d in docs:
        m = d.metadata
        key = (m.get("filepath"), m.get("start_line"), m.get("end_line"))
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{m.get('filepath')}:{m.get('start_line')}-{m.get('end_line')} (symbol: {m.get('name')})")
    return out


def ask(chain, question, retries=3, backoff=30):
    last_err = None
    for i in range(retries + 1):
        try:
            return chain.invoke({"input": question})
        except Exception as e:
            last_err = e
            if i < retries and "429" in str(e):
                print(f"[*] rate limited, retrying in {backoff}s ({i + 1}/{retries})...")
                time.sleep(backoff)
            else:
                break
    raise last_err