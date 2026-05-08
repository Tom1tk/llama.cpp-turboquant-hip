#!/usr/bin/env python3
"""Generate targeted baseline fixtures: each question gets only its answer files."""
import json, os, sys

REPO = "/root/pflash-llama.cpp"

def approx_tokens(text):
    return max(len(text.split()), len(text) // 4)

def read_file(relpath):
    fp = os.path.join(REPO, relpath)
    try:
        return open(fp).read()
    except FileNotFoundError:
        return ""

def build_context(files, budget=30000):
    parts = []
    total = 0
    for path in files:
        content = read_file(path)
        if not content:
            continue
        header = f"\n\n// ===== FILE: {path} =====\n\n"
        chunk = header + content
        ct = approx_tokens(chunk)
        if total + ct > budget:
            remaining = max(budget - total, 0)
            if remaining < 200:
                break
            words = chunk.split()
            chunk = " ".join(words[:int(max(remaining * 0.85, 1))]) + "\n// [truncated]\n"
            ct = approx_tokens(chunk)
        parts.append(chunk)
        total += ct
    return "".join(parts), total

def main():
    try:
        import yaml
    except ImportError:
        print("ERROR: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    outpath = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phase7_targeted_baseline.jsonl"
    questions_file = sys.argv[2] if len(sys.argv) > 2 else f"{REPO}/tools/niah/questions.yaml"
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 30000

    questions = yaml.safe_load(open(questions_file))
    written = 0

    with open(outpath, "w") as out:
        for q in questions:
            answer_files = q.get("answer_files", [])
            context, total_toks = build_context(answer_files, budget)
            if not context.strip():
                print(f"SKIP {q['id']}: no files readable", file=sys.stderr)
                continue

            filler = context + "\n\n---\n\nBased on the source files above, answer the following:\n\n"
            entry = {
                "type": q.get("type", "?"),
                "id": q.get("id", "?"),
                "context_tokens": total_toks,
                "needle": q.get("question", "").strip(),
                "question": q.get("question", "").strip(),
                "filler_text": filler,
                "expected_answer_substrings": q.get("expected_substrings", []),
                "min_recovered": q.get("min_recovered", 1),
                "answer_files": answer_files,
                "difficulty": q.get("difficulty", "medium"),
            }
            out.write(json.dumps(entry) + "\n")
            written += 1
            print(f"  [{q['id']} ({q['type']})] {answer_files} ~{total_toks} tokens", file=sys.stderr)

    print(f"\nWrote {written} targeted fixtures to {outpath}", file=sys.stderr)

if __name__ == "__main__":
    main()
