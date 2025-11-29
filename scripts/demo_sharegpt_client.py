import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.sharegpt_client_simulator import ShareGPTClientSimulator, RequestEvent


def print_sender(event: RequestEvent) -> None:
    """Simple callback used by the simulator to 'send' requests."""
    print("=== EVENT ===")
    print(f"time: {event.timestamp:.3f}s")
    print(f"conversation id: {event.conv_id}")
    print(f"turn index: {event.turn_index}")
    print(f"mode: {event.meta.get('mode', '?')}")
    print("prompt:")
    print(event.prompt)
    print("-" * 40)


def main() -> None:
    sim = ShareGPTClientSimulator(
        max_convs=3,        # how many conversations to replay
        rate_lambda=2.0,    # Poisson rate λ for inter-arrival times
        dataset_split="train",
    )

    print("---- SINGLE TURN ----")
    sim.run(mode="single-turn", send_fn=print_sender)

    print("---- MULTI TURN ----")
    sim.run(mode="multi-turn", send_fn=print_sender)


if __name__ == "__main__":
    main()