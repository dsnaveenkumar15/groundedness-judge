"""Prompt text and verdict parsing. Training and inference must both use this file."""
import re

SYSTEM = (
    "You check whether an answer is supported by the given passages. "
    "Use only the passages, not your own knowledge. "
    "If any claim in the answer is missing from the passages or contradicts them, the answer is unsupported."
)
ASK = (
    "Is every claim in the answer supported by the passages? "
    "Reply with exactly one word: supported or unsupported."
)


def user_message(row: dict) -> str:
    passages = "\n".join(f"[{p['id']}] {p['text'].strip()}" for p in row["context"])
    return (
        f"Passages:\n{passages}\n\nQuestion: {row['question'].strip()}\n\n"
        f"Answer to check:\n{row['answer'].strip()}\n\n{ASK}"
    )


def messages(row: dict, shots: list[dict] = ()) -> list[dict]:
    out = [{"role": "system", "content": SYSTEM}]
    for s in shots:
        out.append({"role": "user", "content": user_message(s)})
        out.append({"role": "assistant", "content": s["label"]})
    out.append({"role": "user", "content": user_message(row)})
    return out


def parse_verdict(text: str) -> str | None:
    """First verdict word in the reply, or None. 'not supported' counts as unsupported."""
    m = re.search(r"\b(not\s+|un)?supported\b", text.lower())
    if not m:
        return None
    return "unsupported" if m.group(1) else "supported"
