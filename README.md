# A Study of Prefix Sharing in LLM Serving

This repository contains our modified fork of **vLLM** for the course project:

> **A Study of Prefix Sharing in LLM Serving**

We instrument vLLM to:

- Plug in a **token simulator** (Milestone 1),
- Measure **prefix reuse** and **KV-cache behavior** under different workloads (Milestone 2),
- Explore **cache eviction policies** for **heterogeneous workloads** (Milestone 3).

All experiments are designed to run on a **CPU-only machine** using the Torch SDPA attention backend.

---

## 1. Environment Setup

### 1.1 Prerequisites

- Python 3.10 (we developed and tested with Anaconda)
- `git`
- A C/C++ build toolchain (for vLLM dependencies)
- Internet access on first run to download:
  - `gpt2` model weights from Hugging Face,
  - ShareGPT Vicuna dataset: `anon8231489123/ShareGPT_Vicuna_unfiltered`.

### 1.2 Create and activate environment (example)

```bash
conda create -n vllm_prefix python=3.10 -y
conda activate vllm_prefix
```

### 1.3 Install vLLM + project dependencies

From the root of this repo:

```bash
# Editable install with dev extras (recommended)
pip install -e ".[dev]"
```

If that fails, you can also do:

```bash
pip install -r requirements.txt
```

> **Note:** For all our experiments we run with:
>
> - `VLLM_PLATFORM=cpu`
> - `VLLM_ATTENTION_BACKEND=TORCH_SDPA`

These are set per command in the examples below.

---

## 2. Project-specific Repository Structure

Only the most relevant project files are listed here:

```text
vllm/
  engine/
    llm_engine.py              # Simulator integration, prefix metrics, dump_prefix_metrics()
  core/block/
    block_manager.py           # SelfAttnBlockSpaceManager prefix reuse tracking
  simulator.py                 # Simple token simulator (Milestone 1)

scripts/
  demo_sharegpt_client.py          # Demo: ShareGPT client simulator (M2 Task 1)
  demo_prefix_metrics_hello.py     # Tiny "Hello world" run that records prefix metrics
  run_sharegpt_prefix_single.py    # Single-turn chat: runs engine + logs prefix stats
  run_sharegpt_prefix_multi.py     # Multi-turn chat: runs engine + logs prefix stats
  run_all_workloads_prefix.py      # Runs all workloads (chat_single, chat_multi, qa, summarization)

tools/
  sharegpt_client_simulator.py # ShareGPT-based client simulator
  summarize_prefix_stats.py    # Summaries from *_requests/blocks.jsonl
  analyze_prefix_stats.py      # CDF plots for reuse fraction, hits per block, reuse gaps
  simulate_cache_policies.py   # Simulate LRU/FIFO for a single workload trace
  analyze_all_prefix_cache.py  # Run all workloads; tables + plots for cache hit rates
  simulate_mixed_workloads.py  # Mixed trace + workload-aware eviction (M3 Task 2)
```

All metrics and figures are written under `/tmp` by default (configurable inside the scripts).

---

## 3. Milestone 1 – Simulator Integration in vLLM

### 3.1 Token simulator

We add a simple token **simulator** in `vllm/simulator.py` and wire it into `LLMEngine`:

- The simulator keeps a `TraceRequest` per request ID.
- For Milestone 1 it simply **echoes prompt tokens** back one by one as “generated” tokens.
- `LLMEngine.step()` can bypass the real model and construct fake `SequenceOutput` objects using the simulator.

This enables fast, controllable experiments without GPU computation.

### 3.2 Unit test: engine with simulator

To verify simulator integration:

```bash
VLLM_PLATFORM=cpu VLLM_ATTENTION_BACKEND=TORCH_SDPA \
python -m pytest tests/test_m1_engine_with_simulator.py -s
```

Expected behavior:

- vLLM loads a `gpt2` model on CPU.
- The test sends prompt `"Hello world"`.
- Output includes:

  ```text
  Prompt: Hello world
  Generated text: Hello world
  .
  ```

- The final `.` indicates the test passed.

### 3.3 Tiny prefix-metrics sanity check

```bash
VLLM_PLATFORM=cpu VLLM_ATTENTION_BACKEND=TORCH_SDPA \
python scripts/demo_prefix_metrics_hello.py
```

This sends a small prompt through the engine and writes prefix-metrics logs to:

- `/tmp/prefix_metrics_toy_requests.jsonl`
- `/tmp/prefix_metrics_toy_blocks.jsonl`

---

## 4. Milestone 2 – Prefix Reuse in ShareGPT Chat

Milestone 2 focuses on **ShareGPT-style chat** using the ShareGPT Vicuna dataset.

### 4.1 ShareGPT client simulator (Task 1)

We implemented a ShareGPT-based client simulator in `tools/sharegpt_client_simulator.py`. It:

- Loads `anon8231489123/ShareGPT_Vicuna_unfiltered` via Hugging Face `datasets`.
- Converts each conversation to:
  - A **single-turn** request (first human message only),
  - A **multi-turn** chat transcript (`Human:` / `Assistant:` format).
- Emits structured events containing:
  - `time`, `conversation_id`, `turn_index`, `mode`, and `prompt`.

Demo script:

```bash
VLLM_PLATFORM=cpu VLLM_ATTENTION_BACKEND=TORCH_SDPA \
python scripts/demo_sharegpt_client.py
```

Expected:

- Downloads the dataset on first run.
- Prints several **single-turn** events, then **multi-turn** events.

### 4.2 Instrumentation for prefix reuse

We add instrumentation in:

- **`SelfAttnBlockSpaceManager` (`block_manager.py`)**:
  - Track per-block `hits_per_block` and `use_times`.
- **`LLMEngine` (`llm_engine.py`)**:
  - For each request, count tokens served via cached prefixes vs recomputed tokens.
  - Compute per-request **`reuse_fraction = reused_tokens / total_tokens`**.
  - Log per-request and per-block stats into JSONL files.

### 4.3 Collect prefix metrics (single vs multi-turn)

Single-turn chat:

```bash
VLLM_PLATFORM=cpu VLLM_ATTENTION_BACKEND=TORCH_SDPA \
PREFIX_MODE=single \
python scripts/run_sharegpt_prefix_single.py
```

Multi-turn chat:

```bash
VLLM_PLATFORM=cpu VLLM_ATTENTION_BACKEND=TORCH_SDPA \
PREFIX_MODE=multi \
python scripts/run_sharegpt_prefix_multi.py
```

This produces (under `/tmp`):

- `prefix_metrics_sharegpt_single_requests.jsonl`
- `prefix_metrics_sharegpt_single_blocks.jsonl`
- `prefix_metrics_sharegpt_multi_requests.jsonl`
- `prefix_metrics_sharegpt_multi_blocks.jsonl`

### 4.4 Summaries & CDF plots (Task 2)

#### 4.4.1 Text summaries

```bash
python tools/summarize_prefix_stats.py /tmp/prefix_metrics_sharegpt_single
python tools/summarize_prefix_stats.py /tmp/prefix_metrics_sharegpt_multi
```

These print:

- Reuse fraction stats (mean, median, p90, p99, fraction of requests with reuse > 0),
- Hits per block stats,
- Reuse gap stats.

#### 4.4.2 Per-workload CDF plots

```bash
python tools/analyze_prefix_stats.py /tmp/prefix_metrics_sharegpt_single
python tools/analyze_prefix_stats.py /tmp/prefix_metrics_sharegpt_multi
```

Generates plots in `/tmp`, for example:

- `reuse_fraction_cdf_single.png`, `reuse_fraction_cdf_multi.png`
- `block_hits_cdf_single.png`, `block_hits_cdf_multi.png`
- `block_reuse_gaps_cdf_single.png`, `block_reuse_gaps_cdf_multi.png`

These show that:

- **Single-turn chat** has almost no reuse.
- **Multi-turn chat** has high reuse fraction and strong locality.

### 4.5 Cache simulations for single vs multi-turn (Task 3)

We simulate LRU and FIFO caches for varying capacities and block sizes using the block-level traces.

Example commands (block size = 16):

```bash
# Single-turn chat
python tools/simulate_cache_policies.py /tmp/prefix_metrics_sharegpt_single 16 LRU
python tools/simulate_cache_policies.py /tmp/prefix_metrics_sharegpt_single 16 FIFO

# Multi-turn chat
python tools/simulate_cache_policies.py /tmp/prefix_metrics_sharegpt_multi 16 LRU
python tools/simulate_cache_policies.py /tmp/prefix_metrics_sharegpt_multi 16 FIFO
```

We repeat for block sizes 32 and 64 by changing the second argument.  
Outputs include hits, misses, events, and hit rates.

Key findings:

- **Single-turn chat**: hit rates ≈ 0% for all capacities/policies → caching is not useful.
- **Multi-turn chat**: hit rate ≈ 57% with 256 blocks (block size 16), saturating at that point.
- LRU and FIFO behave similarly; block size and workload type matter more than policy choice here.

---

## 5. Milestone 3 – Multiple Workloads & Workload-aware Eviction

Milestone 3 generalizes to four workloads:

- `chat_multi` (multi-turn chat),
- `chat_single` (single-turn chat),
- `qa` (short question–answer style),
- `summarization` (long-document summarization).

### 5.1 Generating traces for all workloads

We use a single script to generate prefix traces (requests + blocks) for all workloads and block sizes {16, 32, 64}:

```bash
VLLM_PLATFORM=cpu VLLM_ATTENTION_BACKEND=TORCH_SDPA \
python scripts/run_all_workloads_prefix.py
```

This writes JSONL files like:

- `/tmp/prefix_chat_multi_b16_requests.jsonl`
- `/tmp/prefix_chat_multi_b16_blocks.jsonl`
- `/tmp/prefix_chat_single_b16_requests.jsonl`
- ...
- (similarly for `qa` and `summarization`, and block sizes 32, 64).

### 5.2 Per-workload reuse & cache simulation (Task 1)

We analyze all workloads together with:

```bash
python tools/analyze_all_prefix_cache.py /tmp
```

This script:

1. Discovers all `prefix_*_b*_requests/blocks.jsonl` under `/tmp`.
2. Produces **Task-2 style** prefix stats for each workload:
   - Mean/median/pXX reuse fraction,
   - Fraction of requests with reuse > 0,
   - Approximate “infinite-cache” hit rate.
3. Produces **Task-3 style** cache hit tables across:
   - Block sizes: 16, 32, 64,
   - Capacities: 64, 128, 256, 512,
   - Policies: LRU, FIFO.
4. Writes plots:
   - `reuse_fraction_cdf_<workload>.png`
   - `block_hits_cdf_<workload>.png`
   - `block_reuse_gaps_cdf_<workload>.png`
   - `cache_hits_<workload>_LRU.png`
   - `cache_hits_<workload>_FIFO.png`

High-level conclusions:

- **Cache-friendly workloads**:
  - `chat_multi` and `qa` have high reuse fraction and high hit rates with modest cache sizes (e.g., ~0.57 and ~0.80 hit rate at block=16, cap=256).
- **Cache-unfriendly workloads**:
  - `chat_single` and `summarization` have hit rates near zero even with large caches → not worth caching.
- LRU vs FIFO again behave similarly; the dominant factor is **which workload you cache** and the chosen **block size**.

### 5.3 Mixed workload & workload-aware eviction (Task 2)

We simulate a **mixed trace** combining all four workloads using:

```bash
python tools/simulate_mixed_workloads.py /tmp
```

The script:

1. Loads the block traces (block size 16) for:
   - `chat_multi`, `chat_single`, `qa`, `summarization`.
2. Interleaves them in a fixed pattern into a single mixed stream.
3. Simulates three policies with capacity 256 blocks:
   - **Baseline LRU** (cache everything),
   - **Baseline FIFO** (cache everything),
   - **Workload-aware LRU (WA-LRU)**:
     - Cache **only** `chat_multi` and `qa`,
     - Treat `chat_single` and `summarization` as uncached (always miss, no insertion).

The script prints per-workload and overall stats.  
Example (what we observed):

- **Baseline LRU**:
  - Overall hit rate ≈ 0.3901.
- **Baseline FIFO**:
  - Overall hit rate ≈ 0.3693.
- **Workload-aware LRU (cache only chat_multi + qa)**:
  - Overall hit rate ≈ 0.4281.
  - `chat_multi` and `qa` hit rates match their single-workload behavior.
  - `chat_single` and `summarization` effectively have zero hit rate, but they had almost no cache benefit even in isolation.

This demonstrates that **even a simple workload-aware policy** (based on offline measurements) can:

- Improve overall hit rate,
- Protect cache-friendly workloads from pollution by cache-unfriendly workloads.

---

## 6. Notes and Caveats

- You may see warnings like:

  ```text
  PermissionError: [Errno 13] Permission denied: '/Users/<user>/.config/vllm'
  ```

  These are from vLLM’s optional usage telemetry and **do not affect** our experiments.

- All experiments are run on **CPU** for simplicity; we use small subsets of the workloads to keep runtimes reasonable.

- Dataset and model downloads happen automatically on first use:
  - `gpt2` from Hugging Face transformers,
  - ShareGPT dataset from `datasets`.

---

## 7. Team

- Tarun Vaseekaran – nv30@rice.edu
- Sai Santosh Venkatesh Charumathi – sv77@rice.edu
- Elakkiyan Pugazhenthi – ep68@rice.edu

Please contact us if you have any questions about the code or experiments.
