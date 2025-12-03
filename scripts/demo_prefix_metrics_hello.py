# scripts/demo_prefix_metrics_hello.py

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams


def main():
    # 1. Create engine
    engine_args = EngineArgs(
        model="gpt2",
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    # 2. Toy prompts
    prompts = [
        "Hello world",
        "Hello world, how are you?",
        "Explain the main ideas of Product Launch Formula.",
        "Explain the main ideas of Product Launch Formula in more detail.",
    ]
    sampling_params = SamplingParams(max_tokens=8)

    # 3. Add all requests (metrics are computed inside add_request)
    for i, prompt in enumerate(prompts):
        rid = f"toy-{i}"
        engine.add_request(
            rid,          # request_id
            prompt,       # inputs (prompt text)
            sampling_params,  # params (SamplingParams), positional
        )

    # 4. We don't actually need to run decoding to compute prefix metrics.
    #    dump_prefix_metrics only uses the data collected at add_request time.
    engine.dump_prefix_metrics("/tmp/prefix_metrics_toy")

    print("Done. Metrics written to:")
    print("  /tmp/prefix_metrics_toy_requests.jsonl")
    print("  /tmp/prefix_metrics_toy_blocks.jsonl")


if __name__ == "__main__":
    main()