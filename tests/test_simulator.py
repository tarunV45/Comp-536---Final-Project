# tests/test_simulator.py

from vllm.simulator import Simulator

def test_simulator_echoes_prompt():
    sim = Simulator()
    req_id = "test-1"
    prompt = [10, 20, 30]

    sim.start_request(req_id, prompt)

    tokens = []
    while True:
        t = sim.next_token(req_id)
        if t is None:
            break
        tokens.append(t)

    # For Milestone 1 we just echo the prompt
    assert tokens == prompt