#!/usr/bin/env python3
"""
Per-question context generator for Phase 7 semantic tests.

For each question in questions.yaml:
  - Extracts BSA-relevant sections from large answer files (not full file)
  - Adds filler from non-answer PFlash files
  - Generates 3 context variants: early/mid/late answer position
  - Writes context to tools/niah/contexts/<id>_<pos>.txt
  - Writes JSONL fixture to tools/niah/fixtures/<id>_<pos>.jsonl

Usage:
  python3 gen_question_fixtures.py [--questions-file questions.yaml]
                                   [--filter-id A1,B3,C1]
                                   [--filler-budget 8000]
                                   [--answer-budget 20000]
                                   [--positions early mid late]
"""
import argparse, json, os, random, sys, yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PFlash files suitable as filler (not answer files for any question)
FILLER_CANDIDATES = [
    "src/llama-kv-cache.cpp",
    "src/llama-model.cpp",
    "common/arg.cpp",
    "ggml/src/ggml-cuda/fattn-tile.cu",
    "ggml/src/ggml-backend.cpp",
]

def read_file(relpath):
    fp = os.path.join(REPO, relpath)
    try:
        return open(fp, errors="replace").read()
    except FileNotFoundError:
        return ""

def approx_tokens(text):
    return max(len(text.split()), len(text) // 4)

def extract_section(content, keywords, context_lines=300):
    """Extract lines around first occurrence of any keyword, +/-context_lines."""
    lines = content.splitlines(keepends=True)
    for kw in keywords:
        for i, line in enumerate(lines):
            if kw in line:
                start = max(0, i - context_lines)
                end   = min(len(lines), i + context_lines)
                return "".join(lines[start:end])
    # No keyword found — return first context_lines*2 lines
    return "".join(lines[:context_lines * 2])

def build_answer_block(answer_files, expected_substrings, question_text, answer_budget):
    """Build context string from answer files, extracting relevant sections."""
    # Extract long identifiers from question text — these are more specific, search first
    import re as _re
    q_keywords = []
    for w in _re.findall(r'[a-zA-Z_]\w{5,}', question_text):
        if w.lower() not in {'based','provided','source','which','there','these','their','about','every','location','nature','required','should','would','identify','specific','between','starting','during'}:
            if w not in q_keywords:
                q_keywords.append(w)
    # Question-specific keywords first (more selective), then expected substrings
    all_keywords = q_keywords + list(expected_substrings)
    parts = []
    remaining = answer_budget
    for relpath in answer_files:
        content = read_file(relpath)
        if not content:
            continue
        header = f"\n\n// ===== FILE: {relpath} =====\n\n"
        # Extract relevant section if file is large
        raw_toks = approx_tokens(content)
        if raw_toks > 3000:
            section = extract_section(content, all_keywords)
            section_toks = approx_tokens(section)
            if section_toks > remaining:
                # Truncate section to budget
                words = section.split()
                section = " ".join(words[:max(remaining - 100, 100)]) + "\n// [truncated]\n"
            content = section
        chunk = header + content
        chunk_toks = approx_tokens(chunk)
        if chunk_toks > remaining and remaining < 200:
            break
        if chunk_toks > remaining:
            words = chunk.split()
            chunk = " ".join(words[:max(remaining - 100, 100)]) + "\n// [truncated]\n"
            chunk_toks = approx_tokens(chunk)
        parts.append(chunk)
        remaining -= chunk_toks
    return "".join(parts), answer_budget - remaining

def build_filler(budget, exclude_paths, seed=42):
    """Build filler context from non-answer PFlash files."""
    rng = random.Random(seed)
    candidates = [f for f in FILLER_CANDIDATES
                  if f not in exclude_paths and read_file(f)]
    rng.shuffle(candidates)
    parts = []
    used = 0
    for relpath in candidates:
        if used >= budget:
            break
        content = read_file(relpath)
        # Take a random 200-line window from the file
        lines = content.splitlines(keepends=True)
        if len(lines) > 400:
            start = rng.randint(0, len(lines) - 400)
            lines = lines[start:start + 400]
            content = "".join(lines)
        header = f"\n\n// ===== FILE: {relpath} =====\n\n"
        chunk = header + content
        chunk_toks = approx_tokens(chunk)
        if used + chunk_toks > budget:
            words = chunk.split()
            chunk = " ".join(words[:budget - used]) + "\n// [truncated]\n"
            chunk_toks = approx_tokens(chunk)
        parts.append(chunk)
        used += chunk_toks
    return "".join(parts)

def assemble_context(answer_block, filler, position):
    """
    Assemble context with answer block at early/mid/late position.
    position: 'early' (answer at front), 'mid' (answer in middle), 'late' (answer at end)
    """
    if position == "early":
        return answer_block + "\n" + filler
    elif position == "late":
        return filler + "\n" + answer_block
    else:  # mid
        # Split filler in half, put answer in the middle
        words = filler.split()
        half = len(words) // 2
        filler_a = " ".join(words[:half])
        filler_b = " ".join(words[half:])
        return filler_a + "\n" + answer_block + "\n" + filler_b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-file", default="tools/niah/questions.yaml")
    ap.add_argument("--filter-id",      default=None, help="comma-separated question IDs")
    ap.add_argument("--filler-budget",  type=int, default=8000)
    ap.add_argument("--answer-budget",  type=int, default=20000)
    ap.add_argument("--positions",      nargs="+", default=["early", "mid", "late"])
    ap.add_argument("--ctx-dir",        default="tools/niah/contexts")
    ap.add_argument("--fix-dir",        default="tools/niah/fixtures")
    args = ap.parse_args()

    os.makedirs(os.path.join(REPO, args.ctx_dir), exist_ok=True)
    os.makedirs(os.path.join(REPO, args.fix_dir), exist_ok=True)

    questions = yaml.safe_load(open(os.path.join(REPO, args.questions_file)))

    filter_ids = set(args.filter_id.split(",")) if args.filter_id else None

    for q in questions:
        qid = q["id"]
        if filter_ids and qid not in filter_ids:
            continue
        if q.get("status") == "disabled":
            continue

        answer_files    = q.get("answer_files", [])
        expected        = q.get("expected_substrings", [])
        answer_block, used_toks = build_answer_block(answer_files, expected, q["question"], args.answer_budget)
        filler          = build_filler(args.filler_budget, set(answer_files))

        for pos in args.positions:
            context   = assemble_context(answer_block, filler, pos)
            ctx_fname = f"{qid}_pos-{pos}.txt"
            ctx_path  = os.path.join(REPO, args.ctx_dir, ctx_fname)
            fix_fname = f"{qid}_pos-{pos}.jsonl"
            fix_path  = os.path.join(REPO, args.fix_dir, fix_fname)

            with open(ctx_path, "w") as f:
                f.write(context)

            entry = {
                "id":                       qid,
                "type":                     q.get("type", "?"),
                "difficulty":               q.get("difficulty", "medium"),
                "answer_position":          pos,
                "context_file":             ctx_path,
                "context_tokens":           approx_tokens(context),
                "question":                 q["question"].strip(),
                "filler_text":              "",   # will be loaded from context_file
                "expected_answer_substrings": expected,
                "min_recovered":            q.get("min_recovered", len(expected)),
                "answer_files":             answer_files,
            }
            with open(fix_path, "w") as f:
                f.write(json.dumps(entry) + "\n")

        print(f"  [{qid}] answer={used_toks}tok filler={args.filler_budget}tok "
              f"positions={args.positions}", file=sys.stderr)

    print(f"\nDone. Contexts in {args.ctx_dir}, fixtures in {args.fix_dir}", file=sys.stderr)

if __name__ == "__main__":
    main()
