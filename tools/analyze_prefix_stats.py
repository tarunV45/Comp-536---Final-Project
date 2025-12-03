# tools/analyze_prefix_stats.py

import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def plot_cdf(data: List[float], title: str, xlabel: str, out_path: Path):
    data = sorted(data)
    n = len(data)
    if n == 0:
        print(f"[WARN] no data for {title}")
        return
    xs = data
    ys = [(i + 1) / n for i in range(n)]

    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] saved {out_path}")


def main(prefix: str):
    base = Path(prefix)

    # --- Request-level reuse ---
    req_path = base.with_name(base.name + "_requests.jsonl")
    reuse_single = []
    reuse_multi = []

    for obj in load_jsonl(req_path):
        rf = obj["reuse_fraction"]
        if obj["mode"] == "single":
            reuse_single.append(rf)
        else:
            reuse_multi.append(rf)

    plot_cdf(
        reuse_single,
        title="Reuse fraction per request (single-turn)",
        xlabel="Reuse fraction",
        out_path=base.with_name("reuse_fraction_single_cdf.png"),
    )
    plot_cdf(
        reuse_multi,
        title="Reuse fraction per request (multi-turn)",
        xlabel="Reuse fraction",
        out_path=base.with_name("reuse_fraction_multi_cdf.png"),
    )

    # --- Block-level hits & reuse gaps ---
    blk_path = base.with_name(base.name + "_blocks.jsonl")
    hits = []
    gaps = []

    for obj in load_jsonl(blk_path):
        hits.append(obj["hits"])
        gaps.extend(obj["reuse_gaps"])

    plot_cdf(
        hits,
        title="Hits per logical block",
        xlabel="#hits",
        out_path=base.with_name("block_hits_cdf.png"),
    )
    plot_cdf(
        gaps,
        title="Reuse gap between block uses",
        xlabel="time gap (s)",
        out_path=base.with_name("block_reuse_gaps_cdf.png"),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/analyze_prefix_stats.py /tmp/prefix_stats_sharegpt")
        sys.exit(1)
    main(sys.argv[1])
