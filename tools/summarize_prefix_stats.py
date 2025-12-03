# tools/summarize_prefix_stats.py

import json
import sys
from pathlib import Path
import math


def load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def describe(name, values):
    vals = [v for v in values if not math.isnan(v)]
    vals.sort()
    n = len(vals)
    print(f"\n=== {name} ===")
    print(f"count = {n}")
    if n == 0:
        return
    mean = sum(vals) / n
    median = vals[n // 2]
    def p(q):
        idx = min(n - 1, int(q * n))
        return vals[idx]
    print(f"mean     = {mean:.4f}")
    print(f"median   = {median:.4f}")
    print(f"p90      = {p(0.90):.4f}")
    print(f"p99      = {p(0.99):.4f}")
    print(f"min/max  = {vals[0]:.4f} / {vals[-1]:.4f}")


def summarize_requests(prefix: str):
    base = Path(prefix)
    req_path = base.with_name(base.name + "_requests.jsonl")
    reuse_single = []
    reuse_multi = []

    for obj in load_jsonl(req_path):
        rf = obj["reuse_fraction"]
        if obj.get("mode") == "single":
            reuse_single.append(rf)
        elif obj.get("mode") == "multi":
            reuse_multi.append(rf)

    describe("Reuse fraction per request (single)", reuse_single)
    describe("Reuse fraction per request (multi)", reuse_multi)

    if reuse_single:
        frac_nonzero = sum(1 for v in reuse_single if v > 0) / len(reuse_single)
        print(f"\nSingle: fraction of requests with reuse_fraction > 0: {frac_nonzero:.4f}")
    if reuse_multi:
        frac_nonzero = sum(1 for v in reuse_multi if v > 0) / len(reuse_multi)
        print(f"Multi:  fraction of requests with reuse_fraction > 0: {frac_nonzero:.4f}")


def summarize_blocks(prefix: str):
    base = Path(prefix)
    blk_path = base.with_name(base.name + "_blocks.jsonl")

    hits = []
    gaps = []

    for obj in load_jsonl(blk_path):
        hits.append(obj["hits"])
        gaps.extend(obj.get("reuse_gaps", []))

    describe("Hits per block", hits)
    describe("Reuse gap between block uses", gaps)


def main():
    if len(sys.argv) != 2:
        print("Usage: python tools/summarize_prefix_stats.py /tmp/prefix_metrics_sharegpt_single")
        sys.exit(1)

    prefix = sys.argv[1]
    print(f"Summarizing prefix metrics for prefix={prefix}\n")

    summarize_requests(prefix)
    summarize_blocks(prefix)


if __name__ == "__main__":
    main()
