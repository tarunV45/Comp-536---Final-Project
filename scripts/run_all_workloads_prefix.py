import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams

from tools.sharegpt_client_simulator import ShareGPTClientSimulator
from tools.summarization_client_simulator import SummarizationClientSimulator
from tools.qa_client_simulator import QAClientSimulator


BLOCK_SIZES = [16, 32, 64]


def build_engine():
    engine_args = EngineArgs(
        model="gpt2",
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    return LLMEngine.from_engine_args(engine_args)


def run_workload(workload_name: str, mode: str, block_size: int):
    print(f"=== workload={workload_name}, mode={mode}, block_size={block_size} ===")
    os.environ["PREFIX_MODE"] = mode
    os.environ["PREFIX_BLOCK_SIZE"] = str(block_size)
    os.environ["PREFIX_WORKLOAD"] = workload_name

    engine = build_engine()
    sampling_params = SamplingParams(max_tokens=32)

    # Pick simulator
    if workload_name in ("chat_single", "chat_multi"):
        sim = ShareGPTClientSimulator(max_convs=100, rate_lambda=2.0)

        def send_fn(ev):
            data = ev if isinstance(ev, dict) else vars(ev)
            prompt = data["prompt"]
            conv_id = data.get("conversation_id", data.get("idx", 0))
            turn = data.get("turn_index", 0)
            req_id = f"{workload_name}-{conv_id}-{turn}"
            engine.add_request(req_id, prompt, sampling_params)

        sim.run(
            mode="single-turn" if workload_name == "chat_single" else "multi-turn",
            send_fn=send_fn,
        )

    elif workload_name == "summarization":
        sim = SummarizationClientSimulator(max_docs=100)

        def send_fn(ev):
            req_id = f"summarization-{ev.idx}"
            engine.add_request(req_id, ev.prompt, sampling_params)

        sim.run(send_fn)

    elif workload_name == "qa":
        sim = QAClientSimulator(max_examples=100)

        def send_fn(ev):
            req_id = f"qa-{ev.idx}"
            engine.add_request(req_id, ev.prompt, sampling_params)

        sim.run(send_fn)
    else:
        raise ValueError(f"Unknown workload {workload_name}")

    prefix = f"/tmp/prefix_{workload_name}_b{block_size}"
    engine.dump_prefix_metrics(prefix)
    print(f"[prefix-metrics] wrote {prefix}_requests.jsonl and {prefix}_blocks.jsonl")


def main():
    workloads = [
        ("chat_single", "single"),
        ("chat_multi", "multi"),
        ("summarization", "single"),
        ("qa", "single"),
    ]
    for (w, mode) in workloads:
        for b in BLOCK_SIZES:
            run_workload(w, mode, b)


if __name__ == "__main__":
    main()
