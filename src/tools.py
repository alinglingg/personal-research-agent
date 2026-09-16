from pathlib import Path


def calculate(a, b, operation):
    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        return a / b

    return None


def search_notes(query):
    notes_folder = Path("data/notes")

    matches = []

    for file_path in notes_folder.glob("*.txt"):
        content = file_path.read_text()

        for line in content.splitlines():
            if query.lower() in line.lower():
                matches.append(
                    f"{file_path.name}: {line}"
                )

    return matches
