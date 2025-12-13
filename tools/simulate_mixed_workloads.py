# tools/simulate_mixed_workloads.py
#
# Build a mixed trace from per-workload block traces and compare:
#  - baseline LRU/FIFO (cache all workloads)
#  - workload-aware LRU (cache only "good" workloads, e.g., chat_multi + qa)

import sys
import json
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Any


WORKLOADS = ["chat_multi", "chat_single", "qa", "summarization"]
BLOCK_SIZE = 16      # we use b16 traces
CAPACITY = 256       # cache capacity in blocks
CACHEABLE = {"chat_multi", "qa"}  # workloads we consider "cache-friendly"


def load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_block_events_with_workload(blocks_path: Path, workload: str):
    """
    Convert blocks.jsonl into a flat list of (time, block_key, workload) events.
    We ignore the original absolute timestamps and only preserve ordering.
    """
    events = []
    for obj in load_jsonl(blocks_path):
        key = tuple(obj["block_key"])
        use_times = obj.get("use_times", [])
        # we only care about order, so time within this workload is enough
        for t in use_times:
            events.append((float(t), key, workload))
    # sort by original time within workload, so we preserve per-workload order
    events.sort(key=lambda x: x[0])
    return events


def build_mixed_trace(events_by_workload: Dict[str, List[Tuple[float, Tuple[int, ...], str]]]):
    """
    Simple round-robin mixing: interleave events from each workload while
    preserving per-workload order. Assign synthetic global times.
    Returns a list of (time, block_key, workload).
    """
    mixed = []
    ptr = {w: 0 for w in events_by_workload}
    global_t = 0.0
    step = 1.0

    workloads = list(events_by_workload.keys())
    while True:
        progress = False
        for w in workloads:
            evs = events_by_workload[w]
            i = ptr[w]
            if i < len(evs):
                _, key, _ = evs[i]
                mixed.append((global_t, key, w))
                ptr[w] = i + 1
                global_t += step
                progress = True
        if not progress:
            break
    return mixed


# ---------- cache simulators on mixed trace ----------

def simulate_lru_mixed(events, capacity):
    cache = OrderedDict()
    hits = misses = 0
    hits_by_w = defaultdict(int)
    misses_by_w = defaultdict(int)

    for _, key, w in events:
        if key in cache:
            hits += 1
            hits_by_w[w] += 1
            cache.move_to_end(key)
        else:
            misses += 1
            misses_by_w[w] += 1
            cache[key] = None
            if len(cache) > capacity:
                cache.popitem(last=False)
    return hits, misses, hits_by_w, misses_by_w


def simulate_fifo_mixed(events, capacity):
    from collections import deque

    cache = set()
    queue = deque()
    hits = misses = 0
    hits_by_w = defaultdict(int)
    misses_by_w = defaultdict(int)

    for _, key, w in events:
        if key in cache:
            hits += 1
            hits_by_w[w] += 1
        else:
            misses += 1
            misses_by_w[w] += 1
            cache.add(key)
            queue.append(key)
            if len(cache) > capacity:
                old = queue.popleft()
                cache.remove(old)
    return hits, misses, hits_by_w, misses_by_w


def simulate_workload_aware_lru(events, capacity, cacheable_workloads):
    """
    Workload-aware policy:
      - Only cache blocks from 'cacheable_workloads' (chat_multi, qa).
      - For other workloads (chat_single, summarization), always treat as miss
        and do not insert into cache.
    """
    cache = OrderedDict()
    hits = misses = 0
    hits_by_w = defaultdict(int)
    misses_by_w = defaultdict(int)

    for _, key, w in events:
        if w not in cacheable_workloads:
            # never cache this workload; always miss
            misses += 1
            misses_by_w[w] += 1
            continue

        if key in cache:
            hits += 1
            hits_by_w[w] += 1
            cache.move_to_end(key)
        else:
            misses += 1
            misses_by_w[w] += 1
            cache[key] = None
            if len(cache) > capacity:
                cache.popitem(last=False)

    return hits, misses, hits_by_w, misses_by_w


def print_results(name, hits, misses, hits_by_w, misses_by_w):
    total = hits + misses
    hit_rate = hits / total if total > 0 else 0.0
    print(f"\n=== {name} ===")
    print(f"Total events = {total}, hits = {hits}, misses = {misses}, hit_rate = {hit_rate:.4f}")
    print("| workload | events | hits | misses | hit_rate |")
    print("|---|---|---|---|---|")
    for w in WORKLOADS:
        h = hits_by_w[w]
        m = misses_by_w[w]
        s = h + m
        r = h / s if s > 0 else 0.0
        print(f"| {w} | {s} | {h} | {m} | {r:.4f} |")


def main():
    if len(sys.argv) < 2:
        base_dir = Path("/tmp")
    else:
        base_dir = Path(sys.argv[1])

    print(f"[INFO] base_dir={base_dir}")
    events_by_workload = {}

    for w in WORKLOADS:
        blocks_path = base_dir / f"prefix_{w}_b{BLOCK_SIZE}_blocks.jsonl"
        if not blocks_path.exists():
            print(f"[WARN] missing {blocks_path}, skipping workload {w}")
            continue
        evs = load_block_events_with_workload(blocks_path, w)
        print(f"[INFO] loaded {len(evs)} events for workload={w}")
        events_by_workload[w] = evs

    if not events_by_workload:
        print("[ERROR] no workloads loaded")
        return

    mixed_events = build_mixed_trace(events_by_workload)
    print(f"[INFO] mixed trace has {len(mixed_events)} events")

    # Baseline LRU
    h, m, hbw, mbw = simulate_lru_mixed(mixed_events, CAPACITY)
    print_results(f"Baseline LRU (cache all workloads, cap={CAPACITY})", h, m, hbw, mbw)

    # Baseline FIFO
    h, m, hbw, mbw = simulate_fifo_mixed(mixed_events, CAPACITY)
    print_results(f"Baseline FIFO (cache all workloads, cap={CAPACITY})", h, m, hbw, mbw)

    # Workload-aware LRU
    h, m, hbw, mbw = simulate_workload_aware_lru(mixed_events, CAPACITY, CACHEABLE)
    ca_str = ", ".join(sorted(CACHEABLE))
    print_results(f"Workload-aware LRU (cache only {{{ca_str}}}, cap={CAPACITY})", h, m, hbw, mbw)


if __name__ == "__main__":
    main()
