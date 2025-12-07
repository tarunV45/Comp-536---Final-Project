import json
import sys
from pathlib import Path
from collections import OrderedDict, defaultdict


def load_block_events(blocks_jsonl_path: Path):
    """
    Return a list of (time, block_key) events sorted by time.
    block_key is a tuple[int,...] so it can be used as a dict key.
    """
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
    """
    LRU cache simulation.
    Return (hits, misses).
    """
    cache = OrderedDict()  # block_key -> None
    hits = 0
    misses = 0

    for _, key in events:
        if key in cache:
            hits += 1
            cache.move_to_end(key)  # mark as most recently used
        else:
            misses += 1
            cache[key] = None
            if len(cache) > capacity:
                cache.popitem(last=False)  # evict least recently used
    return hits, misses


def simulate_fifo(events, capacity):
    """
    FIFO cache simulation.
    Return (hits, misses).
    """
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
    """
    LFU cache simulation.
    Return (hits, misses).
    If multiple blocks tie on min frequency, evict the least recently used among them.
    """
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
                # Evict block with lowest freq, break ties by oldest last_use
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


def run_simulation(prefix: str, capacity: int, policy: str):
    """
    prefix: /tmp/prefix_multi_b16   (without _blocks.jsonl)
    capacity: cache capacity in number of blocks
    policy: one of LRU, FIFO, LFU
    """
    base = Path(prefix)
    blk_path = base.with_name(base.name + "_blocks.jsonl")

    events = load_block_events(blk_path)
    if not events:
        print(f"No events found in {blk_path}")
        return

    policy = policy.upper()
    if policy == "LRU":
        hits, misses = simulate_lru(events, capacity)
    elif policy == "FIFO":
        hits, misses = simulate_fifo(events, capacity)
    elif policy == "LFU":
        hits, misses = simulate_lfu(events, capacity)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    total = hits + misses
    hit_rate = hits / total if total > 0 else 0.0

    print(f"Policy={policy}, capacity={capacity} blocks, prefix={prefix}")
    print(f"  events   = {total}")
    print(f"  hits     = {hits}")
    print(f"  misses   = {misses}")
    print(f"  hit rate = {hit_rate:.4f}")


def main():
    if len(sys.argv) != 4:
        print("Usage: python tools/simulate_cache_policies.py PREFIX CAPACITY POLICY")
        print("  e.g. python tools/simulate_cache_policies.py /tmp/prefix_multi_b16 256 LRU")
        sys.exit(1)

    prefix = sys.argv[1]
    capacity = int(sys.argv[2])
    policy = sys.argv[3]

    run_simulation(prefix, capacity, policy)


if __name__ == "__main__":
    main()