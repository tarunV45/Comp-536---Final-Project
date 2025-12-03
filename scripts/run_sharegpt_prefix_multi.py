# scripts/run_sharegpt_prefix_multi.py

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from tools.sharegpt_client_simulator import ShareGPTClientSimulator


def make_engine():
    engine_args = EngineArgs(
        model="gpt2",
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    return engine


def main():
    # Tag run as "multi"
    os.environ["PREFIX_MODE"] = "multi"

    engine = make_engine()
    sampling_params = SamplingParams(max_tokens=32)

    sim = ShareGPTClientSimulator(
        dataset_split="train",
        max_convs=100,
        rate_lambda=2.0,
    )

    def send_to_engine(event):
        # Support both dict-style payloads and RequestEvent objects
        if isinstance(event, dict):
            data = event
        else:
            # Convert attributes to a dict
            data = vars(event)

        # Optional: uncomment once to see what fields you actually have
        # print("DEBUG event keys:", list(data.keys()))

        # Heuristics for field names
        prompt = (
            data.get("prompt")
            or data.get("text")
            or data.get("input")
        )
        conv_id = (
            data.get("conversation_id")
            or data.get("conv_id")
            or data.get("conv_idx")
            or data.get("conversation_index")
            or data.get("conversation_id_int")
        )
        turn = (
            data.get("turn_index")
            or data.get("turn")
            or data.get("step_index")
            or data.get("step")
            or 0
        )

        if prompt is None:
            raise ValueError(f"Could not find prompt field in event. Keys: {list(data.keys())}")

        if conv_id is None:
            # Fallback if no conversation id; just use 0 so we can still run
            conv_id = 0

        req_id = f"multi-{conv_id}-{turn}"

        engine.add_request(
            req_id,
            prompt,
            sampling_params,
        )


    sim.run(mode="multi-turn", send_fn=send_to_engine)

    engine.dump_prefix_metrics("/tmp/prefix_metrics_sharegpt_multi")
    print("Multi-turn metrics dumped to /tmp/prefix_metrics_sharegpt_multi_*")


if __name__ == "__main__":
    main()
