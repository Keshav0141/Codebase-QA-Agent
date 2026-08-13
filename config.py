import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

BASE_URL = "https://opencode.ai/zen/v1"
MODEL = "deepseek-v4-flash-free"
EMBED_PATH = "chroma"
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ZEN_EMBED_MODEL = os.environ.get("ZEN_EMBED_MODEL", "text-embedding-3-classification")


def store_dir():
    name = os.environ.get("CQA_STORE", "default").strip()
    return os.path.join(EMBED_PATH, name)


def get_llm():
    return ChatOpenAI(
        model=MODEL,
        api_key=os.environ["OPENCODE_ZEN_API_KEY"],
        base_url=BASE_URL,
        temperature=0,
    )


_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    provider = os.environ.get("EMBEDDING_PROVIDER", "local").lower()
    if provider == "zen":
        from langchain_openai import OpenAIEmbeddings

        _embeddings = OpenAIEmbeddings(
            model=ZEN_EMBED_MODEL,
            api_key=os.environ["OPENCODE_ZEN_API_KEY"],
            base_url=BASE_URL,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(model_name=LOCAL_EMBED_MODEL)
    return _embeddings