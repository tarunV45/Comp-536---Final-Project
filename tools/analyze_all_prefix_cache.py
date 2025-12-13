# tools/analyze_all_prefix_cache.py
#
# One-shot summary for prefix metrics and cache simulations.
# - Discovers workloads from prefix_*_b*_requests.jsonl / blocks.jsonl
# - Prints Task 2 metrics (reuse fractions, block stats) per workload
# - Prints Task 3 cache hit-rate tables per workload / block size / policy / capacity
# - Also generates plots (CDFs + cache hit curves) per workload.

import json
import sys
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt

# ---- Configuration you can tweak ----
CAPACITIES = [64, 128, 256, 512]
POLICIES = ["LRU", "FIFO"]  # add "LFU" if you want
# -------------------------------------


# -------- Small helpers --------

def load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def quantiles(values: List[float], qs: List[float]) -> List[float]:
    """Return quantiles for sorted list values at probabilities qs (0..1)."""
    if not values:
        return [0.0 for _ in qs]
    vals = sorted(values)
    n = len(vals)
    out = []
    for q in qs:
        if q <= 0:
            out.append(vals[0])
        elif q >= 1:
            out.append(vals[-1])
        else:
            idx = int(q * (n - 1))
            out.append(vals[idx])
    return out


def cdf_xy(values: List[float]):
    """Return (x, y) arrays for a CDF plot from a list of floats."""
    if not values:
        return [], []
    vals = sorted(values)
    n = len(vals)
    xs = vals
    ys = [(i + 1) / n for i in range(n)]
    return xs, ys


def print_markdown_table(headers: List[str], rows: List[List[Any]]):
    # convert everything to str
    rows_str = [[str(x) for x in row] for row in rows]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows_str:
        print("| " + " | ".join(row) + " |")
    print()


# -------- Cache simulators (Task 3) --------

def load_block_events(blocks_jsonl_path: Path):
    """Return a list of (time, block_key) events sorted by time."""
    events = []
    for obj in load_jsonl(blocks_jsonl_path):
        key = tuple(obj["block_key"])
        for t in obj.get("use_times", []):
            events.append((t, key))
    events.sort(key=lambda x: x[0])
    return events


def simulate_lru(events, capacity):
    cache = OrderedDict()
    hits = 0
    misses = 0
    for _, key in events:
        if key in cache:
            hits += 1
            cache.move_to_end(key)
        else:
            misses += 1
            cache[key] = None
            if len(cache) > capacity:
                cache.popitem(last=False)
    return hits, misses


def simulate_fifo(events, capacity):
    from collections import deque

    cache = set()
    queue = deque()
    hits = 0
    misses = 0
    for _, key in events:
        if key in cache:
            hits += 1
        else:
            misses += 1
            cache.add(key)
            queue.append(key)
            if len(cache) > capacity:
                old = queue.popleft()
                cache.remove(old)
    return hits, misses


def simulate_lfu(events, capacity):
    cache = set()
    freq = defaultdict(int)
    last_use = {}
    hits = 0
    misses = 0

    for t, key in events:
        if key in cache:
            hits += 1
        else:
            misses += 1
            if len(cache) >= capacity:
                victim = None
                victim_freq = None
                victim_time = None
                for k in cache:
                    kf = freq[k]
                    kt = last_use.get(k, -1.0)
                    if victim is None:
                        victim = k
                        victim_freq = kf
                        victim_time = kt
                    else:
                        if kf < victim_freq or (kf == victim_freq and kt < victim_time):
                            victim = k
                            victim_freq = kf
                            victim_time = kt
                if victim is not None:
                    cache.remove(victim)
            cache.add(key)

        freq[key] += 1
        last_use[key] = t

    return hits, misses


def simulate(events, capacity, policy):
    policy = policy.upper()
    if policy == "LRU":
        return simulate_lru(events, capacity)
    elif policy == "FIFO":
        return simulate_fifo(events, capacity)
    elif policy == "LFU":
        return simulate_lfu(events, capacity)
    else:
        raise ValueError(f"Unknown policy: {policy}")


# -------- Discovery --------

def discover_files(base_dir: Path):
    """Return mappings for (workload, block_size) -> paths."""
    requests_files: Dict[Tuple[str, int], Path] = {}
    blocks_files: Dict[Tuple[str, int], Path] = {}

    for path in base_dir.glob("prefix_*_b*_requests.jsonl"):
        name = path.name.replace("_requests.jsonl", "")  # prefix_foo_bar_b16
        parts = name.split("_")
        if len(parts) < 3 or parts[0] != "prefix":
            continue
        bpart = parts[-1]
        if not bpart.startswith("b"):
            continue
        block_size = int(bpart[1:])
        workload = "_".join(parts[1:-1])
        requests_files[(workload, block_size)] = path

    for path in base_dir.glob("prefix_*_b*_blocks.jsonl"):
        name = path.name.replace("_blocks.jsonl", "")
        parts = name.split("_")
        if len(parts) < 3 or parts[0] != "prefix":
            continue
        bpart = parts[-1]
        if not bpart.startswith("b"):
            continue
        block_size = int(bpart[1:])
        workload = "_".join(parts[1:-1])
        blocks_files[(workload, block_size)] = path

    workloads = sorted(
        {w for (w, _) in requests_files.keys()} |
        {w for (w, _) in blocks_files.keys()}
    )
    return workloads, requests_files, blocks_files


# -------- Task 2: tables + plots --------

def summarize_task2_tables(base_dir: Path, workloads, requests_files, blocks_files):
    print("\n==================== Task 2: Prefix reuse per workload (tables) ====================\n")

    rows = []
    for workload in workloads:
        # request-level reuse: take smallest block_size that exists
        b_candidates = sorted(b for (w, b) in requests_files.keys() if w == workload)
        if not b_candidates:
            continue
        b_req = b_candidates[0]
        req_path = requests_files[(workload, b_req)]

        reuse_vals = []
        for obj in load_jsonl(req_path):
            rf = obj.get("reuse_fraction")
            if rf is not None:
                reuse_vals.append(float(rf))

        n_req = len(reuse_vals)
        if n_req == 0:
            continue

        reuse_vals_sorted = sorted(reuse_vals)
        mean_reuse = sum(reuse_vals_sorted) / n_req
        median_reuse, p90_reuse, p99_reuse = quantiles(
            reuse_vals_sorted, [0.5, 0.9, 0.99]
        )
        fraction_nonzero = sum(1 for x in reuse_vals_sorted if x > 0.0) / n_req

        # block-level infinite-cache hit rate, using smallest block_size with blocks file
        b_blk_candidates = sorted(b for (w, b) in blocks_files.keys() if w == workload)
        inf_hit_rate = 0.0
        num_blocks = 0
        if b_blk_candidates:
            b_blk = b_blk_candidates[0]
            blk_path = blocks_files[(workload, b_blk)]
            total_hits = 0
            num_blocks = 0
            for obj in load_jsonl(blk_path):
                h = int(obj.get("hits", 0))
                total_hits += h
                num_blocks += 1
            total_events = total_hits + num_blocks
            inf_hit_rate = (total_hits / total_events) if total_events > 0 else 0.0

        rows.append([
            workload,
            n_req,
            f"{mean_reuse:.4f}",
            f"{median_reuse:.4f}",
            f"{p90_reuse:.4f}",
            f"{p99_reuse:.4f}",
            f"{fraction_nonzero:.4f}",
            num_blocks,
            f"{inf_hit_rate:.4f}",
        ])

    headers = [
        "Workload",
        "#requests",
        "mean reuse_frac",
        "median",
        "p90",
        "p99",
        "frac(reuse>0)",
        "#blocks (b=min)",
        "inf-cache hit_rate",
    ]
    print_markdown_table(headers, rows)


def plot_task2_reuse_fraction(base_dir: Path, workloads, requests_files):
    # One CDF per workload
    for workload in workloads:
        b_candidates = sorted(b for (w, b) in requests_files.keys() if w == workload)
        if not b_candidates:
            continue
        b_req = b_candidates[0]
        req_path = requests_files[(workload, b_req)]

        reuse_vals = []
        for obj in load_jsonl(req_path):
            rf = obj.get("reuse_fraction")
            if rf is not None:
                reuse_vals.append(float(rf))
        if not reuse_vals:
            continue

        xs, ys = cdf_xy(reuse_vals)
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel("Reuse fraction")
        plt.ylabel("CDF")
        plt.title(f"Reuse fraction per request – {workload}")
        plt.grid(True)
        out_path = base_dir / f"reuse_fraction_cdf_{workload}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved {out_path}")


def plot_task2_block_hits_and_gaps(base_dir: Path, workloads, blocks_files):
    # Hits per block: overlay block sizes per workload
    for workload in workloads:
        b_sizes = sorted(b for (w, b) in blocks_files.keys() if w == workload)
        if not b_sizes:
            continue

        # Hits CDF
        plt.figure()
        any_data = False
        for b in b_sizes:
            blk_path = blocks_files[(workload, b)]
            hits = []
            for obj in load_jsonl(blk_path):
                hits.append(obj.get("hits", 0))
            if not hits:
                continue
            xs, ys = cdf_xy(hits)
            plt.plot(xs, ys, label=f"block={b}")
            any_data = True
        if any_data:
            plt.xlabel("Hits per block")
            plt.ylabel("CDF")
            plt.title(f"Block hits CDF – {workload}")
            plt.legend()
            plt.grid(True)
            out_path = base_dir / f"block_hits_cdf_{workload}.png"
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print(f"[OK] saved {out_path}")
        else:
            plt.close()

        # Reuse gap CDF
        plt.figure()
        any_data = False
        for b in b_sizes:
            blk_path = blocks_files[(workload, b)]
            gaps = []
            for obj in load_jsonl(blk_path):
                gaps.extend(obj.get("reuse_gaps", []))
            if not gaps:
                continue
            xs, ys = cdf_xy(gaps)
            plt.plot(xs, ys, label=f"block={b}")
            any_data = True
        if any_data:
            plt.xlabel("Reuse gap (seconds)")
            plt.ylabel("CDF")
            plt.title(f"Block reuse gap CDF – {workload}")
            plt.legend()
            plt.grid(True)
            out_path = base_dir / f"block_reuse_gaps_cdf_{workload}.png"
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print(f"[OK] saved {out_path}")
        else:
            plt.close()


# -------- Task 3: tables + plots --------

def summarize_task3_tables(base_dir: Path, workloads, blocks_files):
    print("\n==================== Task 3: Cache hit rates per workload (tables) ====================\n")

    for workload in workloads:
        b_sizes = sorted(b for (w, b) in blocks_files.keys() if w == workload)
        if not b_sizes:
            continue

        print(f"--- Workload: {workload} ---\n")
        for b in b_sizes:
            blk_path = blocks_files[(workload, b)]
            events = load_block_events(blk_path)
            if not events:
                continue

            rows = []
            for cap in CAPACITIES:
                for policy in POLICIES:
                    hits, misses = simulate(events, cap, policy)
                    total = hits + misses
                    hit_rate = hits / total if total > 0 else 0.0
                    rows.append([
                        b,
                        cap,
                        policy,
                        f"{hit_rate:.4f}",
                        hits,
                        misses,
                        total,
                    ])
            headers = [
                "block_size",
                "capacity(blocks)",
                "policy",
                "hit_rate",
                "hits",
                "misses",
                "events",
            ]
            print(f"Block size = {b}")
            print_markdown_table(headers, rows)


def plot_task3_cache_hits(base_dir: Path, workloads, blocks_files):
    # For each workload and policy, plot hit rate vs capacity for each block size.
    for workload in workloads:
        b_sizes = sorted(b for (w, b) in blocks_files.keys() if w == workload)
        if not b_sizes:
            continue

        # precompute results: policy -> block_size -> {capacity: hit_rate}
        results = {policy: {} for policy in POLICIES}

        for b in b_sizes:
            blk_path = blocks_files[(workload, b)]
            events = load_block_events(blk_path)
            if not events:
                continue
            for policy in POLICIES:
                hits_by_cap = {}
                for cap in CAPACITIES:
                    hits, misses = simulate(events, cap, policy)
                    total = hits + misses
                    hit_rate = hits / total if total > 0 else 0.0
                    hits_by_cap[cap] = hit_rate
                    print(
                        f"workload={workload}, policy={policy}, block_size={b}, "
                        f"cap={cap} -> hit_rate={hit_rate:.4f}"
                    )
                results[policy][b] = hits_by_cap

        # Plot per policy
        for policy in POLICIES:
            if not results[policy]:
                continue
            plt.figure()
            for b, hits_by_cap in sorted(results[policy].items()):
                caps = sorted(hits_by_cap.keys())
                rates = [hits_by_cap[c] for c in caps]
                plt.plot(caps, rates, marker="o", label=f"block={b}")
            plt.xlabel("Cache capacity (blocks)")
            plt.ylabel("Hit rate")
            plt.title(f"{workload} – {policy} policy")
            plt.legend()
            plt.grid(True)
            out_path = base_dir / f"cache_hits_{workload}_{policy}.png"
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print(f"[OK] saved {out_path}")


# -------- Main --------

def main():
    if len(sys.argv) == 2:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path("/tmp")

    print(f"[INFO] Using base_dir={base_dir}")
    workloads, requests_files, blocks_files = discover_files(base_dir)

    if not workloads:
        print("[ERROR] No prefix_*_b*_{requests,blocks}.jsonl files found.")
        return

    print(f"[INFO] Found workloads: {', '.join(workloads)}")

    # Task 2: tables + plots
    summarize_task2_tables(base_dir, workloads, requests_files, blocks_files)
    plot_task2_reuse_fraction(base_dir, workloads, requests_files)
    plot_task2_block_hits_and_gaps(base_dir, workloads, blocks_files)

    # Task 3: tables + plots
    summarize_task3_tables(base_dir, workloads, blocks_files)
    plot_task3_cache_hits(base_dir, workloads, blocks_files)


if __name__ == "__main__":
    main()
