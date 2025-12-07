# tools/plot_cache_sweep.py

import json
from pathlib import Path
import sys
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt

MODES = ["single", "multi"]
BLOCK_SIZES = [16, 32, 64]
CAPACITIES = [64, 128, 256, 512]
POLICIES = ["LRU", "FIFO"]  # add "LFU" if desired


def load_block_events(blocks_jsonl_path: Path):
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


def main():
    # Optional: let user override base directory
    if len(sys.argv) == 2:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path("/tmp")

    results = {}  # mode -> policy -> block_size -> {capacity: hit_rate}

    for mode in MODES:
        results[mode] = {}
        for policy in POLICIES:
            results[mode][policy] = {}

        for b in BLOCK_SIZES:
            prefix = base_dir / f"prefix_{mode}_b{b}"
            blk_path = prefix.with_name(prefix.name + "_blocks.jsonl")
            if not blk_path.exists():
                print(f"[WARN] missing {blk_path}, skipping")
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

    # --- Plotting ---
    # For each mode and policy, plot hit-rate vs capacity for all block sizes.
    out_dir = base_dir
    for mode in MODES:
        for policy in POLICIES:
            if not results[mode][policy]:
                continue

            plt.figure()
            for b, hits_by_cap in sorted(results[mode][policy].items()):
                caps = sorted(hits_by_cap.keys())
                rates = [hits_by_cap[c] for c in caps]
                label = f"block={b}"
                plt.plot(caps, rates, marker="o", label=label)

            plt.xlabel("Cache capacity (blocks)")
            plt.ylabel("Hit rate")
            plt.title(f"{mode.capitalize()} workload – {policy} policy")
            plt.legend()
            plt.grid(True)

            out_path = out_dir / f"cache_hits_{mode}_{policy}.png"
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print(f"[OK] saved {out_path}")


if __name__ == "__main__":
    main()
