"""
Verification script: proves the system is working end-to-end.

Checks:
  1. How many records are in Qdrant
  2. Prints a sample of stored Q&A pairs
  3. Runs a live semantic search — embeds a new question and finds
     the closest cached pair from Qdrant
"""
import sys
sys.path.insert(0, "/app")

from openai import OpenAI
from qdrant_client import QdrantClient
from backend.config import settings

openai  = OpenAI(api_key=settings.openai_api_key)
qdrant  = QdrantClient(
    url=settings.qdrant_url,
    port=settings.qdrant_port,
    api_key=settings.qdrant_api_key,
    https=True,
)
col = settings.qdrant_collection_name

# ── 1. Collection stats ────────────────────────────────────────────────────
info  = qdrant.get_collection(col)
count = info.points_count

print("\n" + "=" * 60)
print(f"QDRANT COLLECTION: {col}")
print("=" * 60)
print(f"  Total records : {count}")
print(f"  Vector size   : {info.config.params.vectors.size}")
print(f"  Distance      : {info.config.params.vectors.distance}")

# ── 2. Sample records ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAMPLE STORED Q&A PAIRS (first 3)")
print("=" * 60)

samples = qdrant.scroll(collection_name=col, limit=3, with_payload=True, with_vectors=False)
for point in samples[0]:
    p = point.payload
    print(f"\n  Question   : {p.get('question')}")
    print(f"  Complexity : {p.get('complexity')}")
    print(f"  MDX preview: {str(p.get('mdx', ''))[:80].strip()}...")

# ── 3. Semantic search test ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SEMANTIC SEARCH TEST")
print("=" * 60)

test_questions = [
    "What is the revenue in Turkey?",
    "Show me orders by product type",
    "Premium segment gross margin Germany",
]

for test_q in test_questions:
    # Embed the query
    vec = openai.embeddings.create(
        model=settings.openai_embedding_model,
        input=test_q,
    ).data[0].embedding

    # Search Qdrant
    results = qdrant.search(
        collection_name=col,
        query_vector=vec,
        limit=1,
        with_payload=True,
    )

    if results:
        best   = results[0]
        cached = best.payload.get("question", "")
        score  = round(best.score, 4)
        mdx    = str(best.payload.get("mdx", ""))[:80].strip()
        print(f"\n  Query   : \"{test_q}\"")
        print(f"  Match   : \"{cached}\"")
        print(f"  Score   : {score}  {'✅ Good match' if score > 0.7 else '⚠ Weak match'}")
        print(f"  MDX     : {mdx}...")
    else:
        print(f"\n  Query   : \"{test_q}\"  → No results found")

print("\n" + "=" * 60)
print("Verification complete.")
print("=" * 60)
