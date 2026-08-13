from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chain import make_chain, cite_docs, ask

app = FastAPI(title="Codebase Q&A")


class QueryBody(BaseModel):
    question: str
    k: int = 6


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(body: QueryBody):
    chain = make_chain(k=body.k)
    try:
        result = ask(chain, body.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {
        "answer": result.get("answer"),
        "sources": cite_docs(result.get("context", [])),
    }