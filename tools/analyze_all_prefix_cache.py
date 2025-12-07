import json
import sys
from pathlib import Path
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt

# You can tweak these:
MODES = ["single", "multi"]
BLOCK_SIZES = [16, 32, 64]
CAPACITIES = [64, 128, 256, 512]
POLICIES = ["LRU", "FIFO"]  # add "LFU" if you want


# ---------- Common helpers ----------

def load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_block_events(blocks_jsonl_path: Path):
    """Return a list of (time, block_key) events sorted by time."""
    events = []
    with blocks_jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = tuple(obj["block_key"])
            for t in obj.get("use_times", []):
                events.append((t, key))
    events.sort(key=lambda x: x[0])
    return events


def cdf_xy(values):
    """Return (x, y) arrays for a CDF plot from a list of floats."""
    if not values:
        return [], []
    vals = sorted(values)
    n = len(vals)
    xs = vals
    ys = [(i + 1) / n for i in range(n)]
    return xs, ys


# ---------- Cache simulators (Task 3) ----------

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


# ---------- Task 2 visualizations ----------

def plot_task2_reuse_fraction(base_dir: Path):
    """
    For each mode (single/multi), plot CDF of reuse_fraction.
    We use one block size per mode (the smallest available),
    since reuse_fraction itself does not depend on block size.
    """
    out_dir = base_dir

    for mode in MODES:
        # find first existing prefix for this mode
        prefix = None
        for b in BLOCK_SIZES:
            candidate = base_dir / f"prefix_{mode}_b{b}"
            req_path = candidate.with_name(candidate.name + "_requests.jsonl")
            if req_path.exists():
                prefix = candidate
                break

        if prefix is None:
            print(f"[WARN] no requests.jsonl found for mode={mode}, skipping reuse_fraction CDF")
            continue

        req_path = prefix.with_name(prefix.name + "_requests.jsonl")
        reuse_vals = []
        for obj in load_jsonl(req_path):
            rf = obj.get("reuse_fraction")
            if rf is not None:
                reuse_vals.append(rf)

        if not reuse_vals:
            print(f"[WARN] no reuse_fraction values for mode={mode}")
            continue

        xs, ys = cdf_xy(reuse_vals)
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel("Reuse fraction")
        plt.ylabel("CDF")
        plt.title(f"Reuse fraction per request – {mode} workload")
        plt.grid(True)
        out_path = out_dir / f"reuse_fraction_cdf_{mode}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved {out_path}")


def plot_task2_block_hits_and_gaps(base_dir: Path):
    """
    For each mode, for each block_size, plot:
      - CDF of hits per block
      - CDF of reuse gaps
    Overlaid per block_size.
    """
    out_dir = base_dir

    # Hits per block
    for mode in MODES:
        plt.figure()
        any_data = False

        for b in BLOCK_SIZES:
            prefix = base_dir / f"prefix_{mode}_b{b}"
            blk_path = prefix.with_name(prefix.name + "_blocks.jsonl")
            if not blk_path.exists():
                print(f"[WARN] missing {blk_path}, skipping in hits CDF")
                continue

            hits = []
            for obj in load_jsonl(blk_path):
                hits.append(obj.get("hits", 0))

            if not hits:
                continue

            xs, ys = cdf_xy(hits)
            plt.plot(xs, ys, marker="", label=f"block={b}")
            any_data = True

        if not any_data:
            plt.close()
            continue

        plt.xlabel("Hits per block")
        plt.ylabel("CDF")
        plt.title(f"Block hits CDF – {mode} workload")
        plt.legend()
        plt.grid(True)
        out_path = out_dir / f"block_hits_cdf_{mode}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved {out_path}")

    # Reuse gaps
    for mode in MODES:
        plt.figure()
        any_data = False

        for b in BLOCK_SIZES:
            prefix = base_dir / f"prefix_{mode}_b{b}"
            blk_path = prefix.with_name(prefix.name + "_blocks.jsonl")
            if not blk_path.exists():
                print(f"[WARN] missing {blk_path}, skipping in gaps CDF")
                continue

            gaps = []
            for obj in load_jsonl(blk_path):
                gaps.extend(obj.get("reuse_gaps", []))

            if not gaps:
                continue

            xs, ys = cdf_xy(gaps)
            plt.plot(xs, ys, marker="", label=f"block={b}")
            any_data = True

        if not any_data:
            plt.close()
            continue

        plt.xlabel("Reuse gap (seconds)")
        plt.ylabel("CDF")
        plt.title(f"Block reuse gap CDF – {mode} workload")
        plt.legend()
        plt.grid(True)
        out_path = out_dir / f"block_reuse_gaps_cdf_{mode}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved {out_path}")


# ---------- Task 3 visualizations (cache sweep) ----------

def plot_task3_cache_hits(base_dir: Path):
    """
    For each mode and each policy, plot hit rate vs capacity for all block sizes.
    Uses the block access events and in-memory cache simulation.
    """
    out_dir = base_dir
    results = {}  # mode -> policy -> block_size -> {capacity: hit_rate}

    for mode in MODES:
        results[mode] = {}
        for policy in POLICIES:
            results[mode][policy] = {}

        for b in BLOCK_SIZES:
            prefix = base_dir / f"prefix_{mode}_b{b}"
            blk_path = prefix.with_name(prefix.name + "_blocks.jsonl")
            if not blk_path.exists():
                print(f"[WARN] missing {blk_path}, skipping cache sim for mode={mode}, block={b}")
                continue

            events = load_block_events(blk_path)
            if not events:
                print(f"[WARN] no events in {blk_path}, skipping")
                continue

            for policy in POLICIES:
                hits_by_cap = {}
                for cap in CAPACITIES:
                    hits, misses = simulate(events, cap, policy)
                    total = hits + misses
                    hit_rate = hits / total if total > 0 else 0.0
                    hits_by_cap[cap] = hit_rate
                    print(
                        f"mode={mode}, policy={policy}, block_size={b}, "
                        f"cap={cap} -> hit_rate={hit_rate:.4f}"
                    )
                results[mode][policy][b] = hits_by_cap

    # Plot: for each mode,policy -> capacity vs hit_rate, one line per block_size
    for mode in MODES:
        for policy in POLICIES:
            if not results[mode][policy]:
                continue

            plt.figure()
            for b, hits_by_cap in sorted(results[mode][policy].items()):
                caps = sorted(hits_by_cap.keys())
                rates = [hits_by_cap[c] for c in caps]
                plt.plot(caps, rates, marker="o", label=f"block={b}")

            plt.xlabel("Cache capacity (blocks)")
            plt.ylabel("Hit rate")
            plt.title(f"{mode.capitalize()} workload – {policy} policy")
            plt.legend()
            plt.grid(True)

            out_path = out_dir / f"cache_hits_{mode}_{policy}.png"
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print(f"[OK] saved {out_path}")


# ---------- Main ----------

def main():
    if len(sys.argv) == 2:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path("/tmp")

    print(f"[INFO] Using base_dir={base_dir}")

    # Task 2: per-request reuse + block hits/gaps
    plot_task2_reuse_fraction(base_dir)
    plot_task2_block_hits_and_gaps(base_dir)

    # Task 3: cache hit-rate vs capacity, for each block size
    plot_task3_cache_hits(base_dir)


if __name__ == "__main__":
    main()
