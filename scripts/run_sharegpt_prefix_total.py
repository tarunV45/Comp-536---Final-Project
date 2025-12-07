# scripts/run_sharegpt_prefix_total.py

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from tools.sharegpt_client_simulator import ShareGPTClientSimulator


MODES = ["single", "multi"]
BLOCK_SIZES = [16, 32, 64]
MAX_CONVS = 100  # or whatever you used before


def build_engine():
    engine_args = EngineArgs(
        model="gpt2",
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    return LLMEngine.from_engine_args(engine_args)


def run_for_mode_and_block_size(mode: str, block_size: int):
    assert mode in ("single", "multi")
    print(f"=== Running mode={mode}, block_size={block_size} ===")

    os.environ["PREFIX_MODE"] = mode
    os.environ["PREFIX_BLOCK_SIZE"] = str(block_size)

    engine = build_engine()
    sampling_params = SamplingParams(max_tokens=32)

    sim = ShareGPTClientSimulator(
        max_convs=MAX_CONVS,
        rate_lambda=2.0,
    )

    def send_to_engine(event):
        # Same robust handler we used before
        if isinstance(event, dict):
            data = event
        else:
            data = vars(event)

        prompt = (
            data.get("prompt") or data.get("text") or data.get("input")
        )
        conv_id = (
            data.get("conversation_id")
            or data.get("conv_id")
            or data.get("conv_idx")
            or 0
        )
        turn = (
            data.get("turn_index")
            or data.get("turn")
            or data.get("step_index")
            or 0
        )

        if prompt is None:
            raise ValueError(f"No prompt in event keys={list(data.keys())}")

        req_id = f"{mode}-{conv_id}-{turn}"

        engine.add_request(
            req_id,
            prompt,
            sampling_params,
        )

    sim.run(
        mode="single-turn" if mode == "single" else "multi-turn",
        send_fn=send_to_engine,
    )

    prefix = f"/tmp/prefix_{mode}_b{block_size}"
    engine.dump_prefix_metrics(prefix)
    print(f"[prefix-metrics] wrote {prefix}_requests.jsonl and {prefix}_blocks.jsonl")


def main():
    for mode in MODES:
        for b in BLOCK_SIZES:
            run_for_mode_and_block_size(mode, b)


if __name__ == "__main__":
    main()
