"""
End-to-end test: Question Agent → MDX Agent pipeline.

Generates a small number of questions for the Sales cube, then produces
an MDX query for each one and prints the results.
"""
import json
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

import sys
sys.path.insert(0, "/app")

from backend.agents.question_agent import QuestionGeneratorAgent
from backend.agents.mdx_agent import MDXGeneratorAgent

CUBE_NAME  = "Sales"
NUM_QUESTIONS = 3   # Keep small for a quick test

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
    print(f"\nQuestion   : {pair.question}")
    print(f"Complexity : {pair.complexity}")
    print(f"Dimensions : {pair.dimensions_used}")
    print(f"Measures   : {pair.measures_used}")
    print(f"MDX        :\n{pair.mdx}")
    print("-" * 60)

print(f"\nDone. {len(pairs)} QA pairs generated successfully.")
