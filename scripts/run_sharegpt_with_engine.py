import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams
from tools.sharegpt_client_simulator import ShareGPTClientSimulator

def main():
    # 1) Start your LLM engine (still using the Simulator internally)
    llm = LLM(
        model="gpt2",   # model config still drives tokenizer, etc.
        device="cpu",
    )

    sampling_params = SamplingParams(
        max_tokens=16,  # small, since output is simulated anyway
        temperature=0.0,
    )

    # 2) Build the client simulator
    sim = ShareGPTClientSimulator(
        max_convs=20,
        rate_lambda=2.0,
    )

    def send_to_engine(event):
        prompt = event["prompt"]
        t0 = time.time()
        outputs = llm.generate([prompt], sampling_params)
        t1 = time.time()

        # You *don't* need the text for prefix-cache metrics,
        # but this lets you sanity-check that the engine runs.
        text = outputs[0].outputs[0].text
        print("=== REQUEST DONE ===")
        print(f"mode: {event['mode']}, conv_id: {event['conversation_id']}, turn: {event['turn_index']}")
        print(f"latency: {t1 - t0:.3f}s")
        print(f"prompt len: {len(prompt.split())} words")
        print(f"sample output (truncated): {text[:120]!r}")
        print("----------------------------------------")

    # 3) Run single-turn or multi-turn simulation
    sim.run(mode="single-turn", send_fn=send_to_engine)
    # sim.run(mode="multi-turn", send_fn=send_to_engine)

if __name__ == "__main__":
    main()