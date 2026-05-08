#!/usr/bin/env python3
"""Generate multi-needle NIAH fixture files.

Usage:
  python3 gen_fixtures.py --ctx 32768 --n 10 --out niah_32k_multi.jsonl [--seed 42]
"""
import argparse, json, random, sys

WORDS = [
    "gentle", "distant", "synchrony", "painting", "everything", "alive",
    "wildflowers", "rhythm", "breeze", "rose", "quick", "approaches",
    "scent", "pause", "amber", "sunlit", "patch", "warm", "steady",
    "meadow", "horizon", "pine", "forests", "shades", "restful",
    "evening", "fox", "descent", "surrounding", "breathe", "moment",
    "grass", "clouds", "agreement", "nap", "settles", "pulse", "jumps",
    "indigo", "slow", "dog", "brown", "lazy", "long", "toward",
    "small", "begins", "itself", "carries", "world", "across", "while",
]

ADJECTIVES = ["mauve", "crimson", "cerulean", "viridian", "ochre", "sepia",
               "sienna", "umber", "amaranth", "byzantium", "carmine", "cerise"]
NOUNS      = ["velociraptor", "wolverine", "albatross", "salamander",
               "pangolin", "chameleon", "capybara", "axolotl", "narwhal"]

def gen_secret(rng):
    adj  = rng.choice(ADJECTIVES)
    noun = rng.choice(NOUNS)
    num  = rng.randint(1000, 9999)
    return f"{adj}-{noun}-{num}"

def gen_filler(n_tokens_approx, rng):
    """Generate approximately n_tokens_approx of filler text.
    Qwen's tokenizer averages ~1.5 tokens per English word, so use words ÷ 1.5."""
    n_words = int(n_tokens_approx / 1.5)
    return " ".join(rng.choices(WORDS, k=n_words))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx",  type=int, required=True, help="context size in tokens")
    ap.add_argument("--n",    type=int, default=10,    help="number of test cases")
    ap.add_argument("--out",  type=str, required=True, help="output JSONL path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    filler_tokens = args.ctx - 150
    question = "Based on the document above, what is the secret pass code? Answer in one short sentence."

    with open(args.out, "w") as f:
        for i in range(args.n):
            secret = gen_secret(rng)
            needle = f"The secret pass code is {secret}."

            filler = gen_filler(filler_tokens, rng)
            words  = filler.split()
            insert_at = rng.randint(len(words) // 10, 9 * len(words) // 10)
            words.insert(insert_at, needle)
            filler_with_needle = " ".join(words)

            entry = {
                "context_tokens": args.ctx,
                "needle": needle,
                "question": question,
                "expected_answer_substring": secret,
                "filler_text": filler_with_needle,
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {args.n} fixtures to {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
