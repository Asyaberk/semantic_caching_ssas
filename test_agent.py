"""Quick test script to verify the Question Generator Agent works end-to-end."""
import sys
sys.path.insert(0, "/app")

from backend.agents.question_agent import QuestionGeneratorAgent

agent = QuestionGeneratorAgent()
print("Agent initialised successfully.")

questions = agent.generate(cube_name="Sales", count=5, language="en")
print(f"\nGenerated {len(questions)} questions:")
for i, q in enumerate(questions, 1):
    print(f"  {i}. {q}")
