#!/usr/bin/env python3
"""
Generate code-semantic NIAH fixtures from a question library.

Usage:
  python3 gen_code_fixtures.py \
    --context context_32k.txt \
    --manifest context_32k_manifest.json \
    --questions questions.yaml \
    --out code_fixtures_32k.jsonl \
    [--filter-type A B C]
    [--filter-bucket mid]
"""
import argparse, json, sys

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context",   required=True)
    ap.add_argument("--manifest",  required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out",       required=True)
    ap.add_argument("--filter-type", nargs="*")
    ap.add_argument("--filter-bucket", nargs="*",
                    help="only include questions whose answer files are in these position buckets")
    args = ap.parse_args()

    context  = open(args.context).read()
    manifest = json.load(open(args.manifest))
    questions = yaml.safe_load(open(args.questions))

    if not isinstance(questions, list):
        print("ERROR: questions.yaml must contain a top-level list", file=sys.stderr)
        sys.exit(1)

    file_buckets = {f["path"]: f["position_bucket"] for f in manifest["files"]}

    written = 0
    with open(args.out, "w") as out:
        for q in questions:
            if args.filter_type and q.get("type") not in args.filter_type:
                continue
            answer_files = q.get("answer_files", [])
            buckets = [file_buckets.get(f, "unknown") for f in answer_files]
            if args.filter_bucket and not any(b in args.filter_bucket for b in buckets):
                continue

            filler = context + "\n\n---\n\nBased on the source files above, answer the following:\n\n"
            prompt_question = q.get("question", "").strip()

            entry = {
                "type":                     q.get("type", "?"),
                "id":                       q.get("id", "?"),
                "context_tokens":           manifest["total_tokens_approx"],
                "needle":                   prompt_question,
                "question":                 prompt_question,
                "filler_text":              filler,
                "expected_answer_substring": q.get("expected_substrings", []),
                "expected_answer_substrings": q.get("expected_substrings", []),
                "min_recovered":            q.get("min_recovered", len(q.get("expected_substrings", []))),
                "answer_files":             answer_files,
                "answer_position_buckets":  buckets,
                "difficulty":               q.get("difficulty", "medium"),
            }
            out.write(json.dumps(entry) + "\n")
            written += 1

    print(f"Wrote {written} fixtures to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
