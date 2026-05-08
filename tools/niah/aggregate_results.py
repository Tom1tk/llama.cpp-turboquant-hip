#!/usr/bin/env python3
"""Aggregate all tools/niah/results/*.json into summary tables.

Usage:
  python3 aggregate_results.py                   # full summary
  python3 aggregate_results.py --mode windowed   # one mode
  python3 aggregate_results.py --kr 0.65         # one KR
  python3 aggregate_results.py --by-position     # show position breakdown
  python3 aggregate_results.py --questions       # show per-question breakdown
"""
import argparse, collections, json, os, glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

def load_results(mode_filter=None, kr_filter=None):
    results = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        try:
            d = json.load(open(path))
            if mode_filter and d.get("mode") != mode_filter: continue
            if kr_filter  and d.get("kr")   != kr_filter:   continue
            results.append(d)
        except Exception:
            pass
    return results

def summary_by_type(results):
    """Pass rate per (mode, kr, type)."""
    cell = collections.defaultdict(lambda: [0, 0])  # noqa
    for r in results:
        key = (r.get("mode","?"), r.get("kr","?"), r.get("type","?"))
        cell[key][1] += 1
        if r.get("pass"): cell[key][0] += 1
    return cell

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",        default=None)
    ap.add_argument("--kr",          default=None)
    ap.add_argument("--by-position", action="store_true")
    ap.add_argument("--questions",   action="store_true", help="show per-question breakdown")
    args = ap.parse_args()

    results = load_results(args.mode, args.kr)
    if not results:
        print("No results found in", RESULTS_DIR)
        return

    # Overall table: mode x kr -> pass%
    print(f"\n{'Mode':>9} {'KR':>5}  {'Total':>10}  {'Pass%':>6}  A  B  C  D  E")
    print("-" * 65)

    cell = summary_by_type(results)
    seen = set()
    for r in sorted(results, key=lambda x: (x.get("mode",""), x.get("kr",""))):
        key = (r.get("mode",""), r.get("kr",""))
        if key in seen: continue
        seen.add(key)
        mode, kr = key
        mode_kr_results = [x for x in results if x.get("mode")==mode and x.get("kr")==kr]
        n_total = len(mode_kr_results)
        n_pass  = sum(1 for x in mode_kr_results if x.get("pass"))
        pct     = 100 * n_pass // n_total if n_total else 0
        types   = []
        for t in "ABCDE":
            p, tot = cell.get((mode, kr, t), [0, 0])
            types.append(f"{p}/{tot}" if tot else "—")
        print(f"{mode:>9} {kr:>5}  {n_pass:>4}/{n_total:<4} {pct:>5}%  {'  '.join(types)}")

    if args.by_position:
        print(f"\n{'Mode':>9} {'KR':>5} {'Pos':>5}  {'Total':>10}  {'Pass%':>6}")
        print("-" * 45)
        seen_pos = set()
        for r in sorted(results, key=lambda x: (x.get("mode",""), x.get("kr",""), x.get("position",""))):
            key = (r.get("mode",""), r.get("kr",""), r.get("position",""))
            if key in seen_pos: continue
            seen_pos.add(key)
            mode, kr, pos = key
            sub = [x for x in results if x.get("mode")==mode and x.get("kr")==kr and x.get("position")==pos]
            n_p, n_t = sum(1 for x in sub if x.get("pass")), len(sub)
            pct = 100 * n_p // n_t if n_t else 0
            print(f"{mode:>9} {kr:>5} {pos:>5}  {n_p:>4}/{n_t:<4} {pct:>5}%")

    if args.questions:
        print(f"\n{'ID':>5} {'Type':>5} {'Mode':>9} {'KR':>5} {'Pos':>5}  {'P/T':>7}  {'Answer preview'}")
        print("-" * 80)
        for r in sorted(results, key=lambda x: (x.get("id",""), x.get("mode",""), x.get("kr",""), x.get("position",""))):
            ans = r.get("answer","")[:50].replace("\n"," ")
            p = "PASS" if r.get("pass") else "FAIL"
            print(f"{r.get('id','?'):>5} {r.get('type','?'):>5} {r.get('mode','?'):>9} {r.get('kr','?'):>5} "
                  f"{r.get('position','?'):>5}  {p:>4}  {ans}")

if __name__ == "__main__":
    main()
