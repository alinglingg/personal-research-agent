from research_agent import run_research_agent


test_cases = [
    "What is the difference between agent state and agent memory?",
    "What does my notes say about SQLite?",
    "What is agent memory?"
]


for test_case in test_cases:
    print("\n==============================")
    print("Testing:")
    print(test_case)
    print("==============================")

    result = run_research_agent(test_case)

    print("\nEvaluation result:")
    print(result["evaluation"]["overall_pass"])