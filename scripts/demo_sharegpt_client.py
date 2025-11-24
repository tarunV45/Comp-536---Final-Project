import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.sharegpt_client_simulator import ShareGPTClientSimulator, RequestEvent


def print_sender(event: RequestEvent) -> None:
    print("=== EVENT ===")
    print("t:", round(event.timestamp, 3),
          "| conv:", event.conv_id,
          "| turn:", event.turn_index,
          "| mode:", event.meta["mode"])
    print("Prompt snippet:", event.prompt_text[:200].replace("\n", " "))
    print()


def main():
    sim = ShareGPTClientSimulator(
        # if huggingface download is slow, lower max_convs
        max_convs=3,
        rate_lambda=2.0,
    )

    print("---- SINGLE TURN ----")
    sim.run(mode="single-turn", send_fn=print_sender)

    print("---- MULTI TURN ----")
    sim.run(mode="multi-turn", send_fn=print_sender)


if __name__ == "__main__":
    main()
