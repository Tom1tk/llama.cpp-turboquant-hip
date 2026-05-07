#!/usr/bin/env python3
"""
Assemble source files into a code context for NIAH testing.

Usage:
  python3 pack_code_context.py \
    --files src/llama-context.cpp ggml/include/ggml.h ... \
    --budget 30000 \
    --order shuffle|depth-first|interleave \
    --out context_32k.txt \
    --manifest context_32k_manifest.json \
    [--seed 42]

Output manifest JSON:
  {
    "total_tokens_approx": 31402,
    "total_chars": 123456,
    "files": [
      {"path": "src/llama-context.cpp", "start_token_approx": 0, "end_token_approx": 3100,
       "position_frac": 0.0, "position_bucket": "early"},
      ...
    ]
  }
"""
import argparse, json, os, random, sys

def approx_tokens(text):
    return max(len(text.split()), len(text) // 4)

def position_bucket(start_frac):
    if start_frac < 0.33:
        return "early"
    if start_frac < 0.67:
        return "mid"
    return "late"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files",    nargs="+", required=True)
    ap.add_argument("--budget",   type=int,  default=30000)
    ap.add_argument("--order",    choices=["original","shuffle","interleave"], default="original")
    ap.add_argument("--out",      required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--repo",     default=".", help="repo root for relative paths")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = list(args.files)

    if args.order == "shuffle":
        rng.shuffle(files)
    elif args.order == "interleave":
        half = len(files) // 2
        a, b = files[:half], files[half:]
        files = [x for pair in zip(a, b) for x in pair] + files[len(a)+len(b):]

    parts = []
    total_toks = 0
    for path in files:
        full_path = os.path.join(args.repo, path)
        try:
            content = open(full_path).read()
        except FileNotFoundError:
            print(f"WARNING: {full_path} not found, skipping", file=sys.stderr)
            continue
        header = f"\n\n// ===== FILE: {path} =====\n\n"
        chunk = header + content
        chunk_toks = approx_tokens(chunk)
        if total_toks + chunk_toks > args.budget:
            remaining = args.budget - total_toks
            if remaining < 200:
                break
            words = chunk.split()
            chunk = " ".join(words[:int(remaining * 0.85)]) + "\n// [truncated]\n"
            chunk_toks = approx_tokens(chunk)
        parts.append((path, chunk, total_toks, total_toks + chunk_toks))
        total_toks += chunk_toks

    context = "".join(p[1] for p in parts)
    total_chars = len(context)

    manifest = {
        "total_tokens_approx": total_toks,
        "total_chars": total_chars,
        "files": [],
    }
    for path, _chunk, tok_start, tok_end in parts:
        frac = tok_start / max(total_toks, 1)
        manifest["files"].append({
            "path": path,
            "token_start_approx": tok_start,
            "token_end_approx":   tok_end,
            "position_frac": round(frac, 3),
            "position_bucket": position_bucket(frac),
        })

    with open(args.out, "w") as f:
        f.write(context)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Packed {len(parts)} files, ~{total_toks} tokens ({total_chars} chars) -> {args.out}", file=sys.stderr)
    for p in manifest["files"]:
        print(f"  [{p['position_bucket']:5} {p['position_frac']:.2f}] {p['path']}", file=sys.stderr)

if __name__ == "__main__":
    main()
