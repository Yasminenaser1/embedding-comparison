import json
import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
]

recipes = json.load(open("recipes.json"))
cases = [c for c in json.load(open("eval_set.json")) if c["expected"]]


def short_form(t):
    return t.split("Directions:")[0].strip() if "Directions:" in t else t


docs = [
    Document(page_content=short_form(r["text"]), metadata={"title": r["title"]})
    for r in recipes
]


def evaluate(model_name):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    t0 = time.time()
    store = Chroma.from_documents(
        docs,
        embeddings,
        collection_name=model_name.split("/")[-1].replace(".", "-"),
    )
    index_time = time.time() - t0

    total_rr = 0.0
    t0 = time.time()
    for c in cases:
        hits = store.similarity_search(c["query"], k=10)
        titles = [h.metadata["title"] for h in hits]
        rank = next((i + 1 for i, t in enumerate(titles) if t in c["expected"]), None)
        total_rr += 1 / rank if rank else 0.0
    query_time = (time.time() - t0) / len(cases)

    return {
        "model": model_name.split("/")[-1],
        "mrr": total_rr / len(cases),
        "index_s": index_time,
        "query_ms": query_time * 1000,
    }


results = []
for m in MODELS:
    print(f"\nevaluating {m}...")
    r = evaluate(m)
    results.append(r)
    print(f"  MRR {r['mrr']:.3f}  index {r['index_s']:.1f}s  "
          f"query {r['query_ms']:.0f}ms")

print("\n| model | MRR | index (s) | query (ms) |")
print("|---|---|---|---|")
for r in results:
    print(f"| {r['model']} | {r['mrr']:.3f} | {r['index_s']:.1f} | "
          f"{r['query_ms']:.0f} |")

json.dump(results, open("results.json", "w"), indent=2)
