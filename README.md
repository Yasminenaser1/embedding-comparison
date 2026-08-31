# embedding-comparison

Does a bigger embedding model actually improve retrieval on a small corpus,
and what does it cost?

Three sentence-transformers models compared on the same 300-document corpus
and the same 15-query eval set, using LangChain's `HuggingFaceEmbeddings` so
swapping models is a one-line change.

## Results

| model | params | MRR | index (s) | query (ms) |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 22M | 0.802 | 0.5 | 8 |
| bge-base-en-v1.5 | 109M | 0.822 | 1.5 | 10 |
| all-mpnet-base-v2 | 110M | 0.867 | 1.9 | 12 |

Bigger helps. MRR climbs monotonically with model size — 6.5 points from
smallest to largest.

The cost split is the more useful finding. Index time nearly 4x from MiniLM
to mpnet, but query latency only moves 8ms to 12ms. The expensive part is
indexing, which happens once; the part that runs on every request barely
changes. On a corpus this size, the bigger model is clearly worth it.

## A measurement I threw away

The first version of this also reported memory usage per model. It showed
MiniLM using 539MB and mpnet using 39MB — the smallest model apparently using
13x the memory of the largest.

The cause was load order. Memory was sampled before and after each model, but
the first model in the loop also loaded all of PyTorch, so MiniLM was charged
for the entire framework while later models were measured against an
already-warm process. It was measuring order, not model size.

Fixing it properly would mean running each model in a separate process. I cut
the column instead — memory scales predictably with parameter count, so a
precise number wouldn't have changed any decision the table supports.

## Caveats

300 documents and 15 queries is a small test. Index time scales linearly, so
the 1.4s gap becomes hours on a corpus of millions. The 4ms query gap is
negligible here and would not be at high request volume.

The eval set comes from a separate project ([recipe-rag](https://github.com/Yasminenaser1/recipe-rag))
where it was hand-built and validated.

## Files

- `compare.py` — loads each model, indexes the corpus, scores MRR, times both
- `recipes.json` / `eval_set.json` — corpus and eval set, copied from recipe-rag
- `chain.py` — LCEL chain: retriever → prompt → LLM → output parser
- `results.json` — raw output
