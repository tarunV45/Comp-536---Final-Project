# scripts/demo_simulator_only.py

from vllm.simulator import Simulator

def main():
    sim = Simulator()
    prompt_ids = [101, 202, 303, 404]
    request_id = "demo-req"

    sim.start_request(request_id, prompt_ids)

    print("Prompt token IDs :", prompt_ids)
    print("Simulator outputs:")

    while True:
        t = sim.next_token(request_id)
        if t is None:
            break
        print("  next_token ->", t)

if __name__ == "__main__":
    main()
