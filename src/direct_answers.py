"""Conservative, evidence-backed shortcuts; ambiguous requests use the model."""
import re


def split_simple_goal(goal):
    """Only allow known formatting suffixes, never discard extra instructions."""
    match = re.fullmatch(
        r"\s*([^?!]+?)[?.]?\s*(?:Reply (with only the number|exactly with the "
        r"(?:matching|saved) sentence followed by \([\w.-]+\))\.?)?\s*",
        goal, re.IGNORECASE,
    )
    if not match:
        return None
    main = match[1].strip().rstrip(".")
    # The main clause must not swallow a second sentence or instruction.
    if re.search(r"[?;!]|\b(?:reply|then|also|explain|summari[sz]e|compare)\b", main, re.IGNORECASE):
        return None
    return main, match[2]


def lookup_subject(goal, tool):
    parsed = split_simple_goal(goal)
    if not parsed or parsed[1] == "with only the number":
        return None
    main, _ = parsed
    if tool == "search_notes":
        pattern = r"search (?:my |the )?(?:local )?notes for (?:the |my )?([a-z][a-z -]*)"
    elif tool == "search_memory":
        pattern = r"what ([a-z][a-z -]*) did I (?:previously )?save in memory"
    else:
        return None
    match = re.fullmatch(pattern, main, re.IGNORECASE)
    if match and not re.search(r"\b(?:and|or|why|how)\b", match[1], re.IGNORECASE):
        return match[1].strip().lower()
    return None


def calculation_request(goal):
    """Parse only complete single-operation requests, including their formatting."""
    parsed = split_simple_goal(goal)
    if not parsed or (parsed[1] and parsed[1].lower() != "with only the number"):
        return None
    number = r"(-?\d+(?:\.\d+)?)"
    match = re.fullmatch(
        rf"(?:what is |calculate )?{number}\s*(\+|plus|-|minus|\*|times|multiplied by|/|divided by)\s*{number}",
        parsed[0], re.IGNORECASE,
    )
    if not match:
        return None
    operations = {"+": "add", "plus": "add", "-": "subtract", "minus": "subtract",
                  "*": "multiply", "times": "multiply", "multiplied by": "multiply",
                  "/": "divide", "divided by": "divide"}
    def operand(text):
        return float(text) if "." in text else int(text)
    return {"action": "calculate", "a": operand(match[1]), "b": operand(match[3]),
            "operation": operations[match[2].lower()]}


def calculator_answer(goal, record):
    request = calculation_request(goal)
    if request is None or any(request[key] != record[key] for key in ("a", "b", "operation")):
        return None
    return str(record["result"])


def direct_answer(state):
    history, evidence = state["tool_history"], state["evidence"]
    if len(history) != 1 or len(evidence) != 1:
        return None
    record, item = history[0], evidence[0]
    if record["tool"] == "calculate":
        return calculator_answer(state["goal"], record)
    subject = lookup_subject(state["goal"], record["tool"])
    if not subject or subject != record["query"].strip().lower():
        return None
    content = item["content"].strip()
    # A single short declarative fact with the exact requested subject. Verbatim
    # quotation retains negation/qualifiers; no inference or rewriting is done.
    if len(content) > 300 or not re.fullmatch(
        rf"(?:my |your |the )?{re.escape(subject)} is [^.!?;\n]+[.]?", content, re.IGNORECASE,
    ):
        return None
    if re.search(r"\b(?:and|or|but|if|unless|ignore|instructions|system|assistant)\b", content, re.IGNORECASE):
        return None
    parsed = split_simple_goal(state["goal"])
    if parsed[1] and f"({item['source']})" not in parsed[1]:
        return None
    return f"{content} ({item['source']})"
