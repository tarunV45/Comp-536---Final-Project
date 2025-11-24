# tests/test_m1_engine_with_simulator.py

from vllm import LLM
from vllm.sampling_params import SamplingParams


def test_engine_uses_simulator_echo():
    # Force CPU device; model weights don't matter since we bypass execution
    llm = LLM(
        model="gpt2",
        device="cpu",
    )

    prompts = ["Hello world"]
    sampling_params = SamplingParams(max_tokens=5)

    outputs = llm.generate(prompts, sampling_params)
    out = outputs[0].outputs[0]

    print("Prompt:", outputs[0].prompt)
    print("Generated text:", out.text)

    # With the Milestone 1 simulator that echoes prompt tokens,
    # the generated text should at least start with the original prompt.
    assert out.text.strip().startswith("Hello")
