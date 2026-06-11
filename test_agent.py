"""
End-to-end pipeline test:
  Question Agent → MDX Agent → Qdrant Uploader

Generates questions for the Sales cube, produces MDX for each,
then uploads everything to Qdrant.
"""
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

import sys
sys.path.insert(0, "/app")

from backend.agents.question_agent  import QuestionGeneratorAgent
from backend.agents.mdx_agent       import MDXGeneratorAgent
from backend.agents.uploader_agent  import QdrantUploaderAgent

CUBE_NAME     = "Sales"
NUM_QUESTIONS = 3   # keep small for a quick smoke test

# ── Step 1: Generate questions ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 — Generating questions")
print("=" * 60)

q_agent   = QuestionGeneratorAgent()
questions = q_agent.generate(cube_name=CUBE_NAME, count=NUM_QUESTIONS, language="en")

for i, q in enumerate(questions, 1):
    print(f"  {i}. {q}")

# ── Step 2: Generate MDX for each question ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Generating MDX queries")
print("=" * 60)

mdx_agent = MDXGeneratorAgent()
pairs     = mdx_agent.generate_batch(questions=questions, cube_name=CUBE_NAME)

for pair in pairs:
    print(f"\n  Question   : {pair.question}")
    print(f"  Complexity : {pair.complexity}")
    print(f"  MDX preview: {pair.mdx[:80].strip()}...")

# ── Step 3: Upload to Qdrant ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Uploading to Qdrant")
print("=" * 60)

uploader = QdrantUploaderAgent()
count    = uploader.upload(pairs)

print(f"\nDone. {count} / {len(pairs)} QA pairs uploaded to Qdrant.")
