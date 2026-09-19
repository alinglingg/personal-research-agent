from tools import save_summary

summary = "This is a test research summary."

approval = input("Save this summary? (y/n): ")

if approval.lower() == "y":
    result = save_summary(summary)
    print(result)
else:
    print("Save cancelled.")