# tools/sharegpt_client_simulator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import math
import random

from datasets import load_dataset


# This is the HF repo & file the project PDF points to.
_HF_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
_DEFAULT_JSON_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"


@dataclass
class RequestEvent:
    """
    A single "client request" event produced by the simulator.

    Fields are chosen to match how demo_sharegpt_client.py uses them:
      - timestamp : simulated time (seconds) when this request arrives
      - conv_id   : which conversation this event belongs to
      - turn_index: which turn within that conversation (0-based)
      - prompt    : text to send to the LLM (user side)
      - meta      : extra metadata (we at least include {"mode": ...})
    """
    timestamp: float
    conv_id: int
    turn_index: int
    prompt: str
    meta: Dict[str, Any]


class ShareGPTClientSimulator:
    """
    Simulator that replays conversations from the ShareGPT Vicuna dataset
    as if they were clients sending requests to our LLM server.

    - We load the dataset JSON:
        https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    - Each entry has a "conversations" list of alternating
      { "from": "human", "value": ... } and { "from": "gpt", "value": ... }.

    Parameters
    ----------
    max_convs : int
        Maximum number of conversations to replay.
    rate_lambda : float
        Poisson rate λ (in requests per second) used to generate
        inter-arrival times: Δt ~ Exp(λ).
    dataset_split : str
        Which split inside the JSON to use ("train" or "test").
        The default "train" matches common usage.
    """

    def __init__(
        self,
        max_convs: int = 3,
        rate_lambda: float = 2.0,
        dataset_split: str = "train",
    ) -> None:
        self.max_convs = max_convs
        self.rate_lambda = rate_lambda
        self.dataset_split = dataset_split

        # Load ShareGPT dataset from HuggingFace using the JSON builder.
        self._dataset = self._load_sharegpt_dataset()

        # For reproducibility in tests.
        random.seed(0)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def _load_sharegpt_dataset(self):
        """
        Load the ShareGPT Vicuna dataset from HuggingFace.

        The HF repo only contains raw JSON files (no dataset script),
        so we MUST use the "json" builder and give an explicit data_files
        argument pointing to the JSON file.
        """
        json_url = (
            f"https://huggingface.co/datasets/{_HF_REPO_ID}"
            f"/resolve/main/{_DEFAULT_JSON_FILE}"
        )

        # This returns a datasets.Dataset object.
        ds = load_dataset(
            "json",
            data_files=json_url,
            split=self.dataset_split,
        )
        return ds

    # ------------------------------------------------------------------
    # Internal helpers to iterate over conversations & turns
    # ------------------------------------------------------------------
    def _iter_conversations(self):
        """
        Yield (conv_index, conversations_list) for up to max_convs examples.

        Each element of conversations_list is a dict like:
          { "from": "human" or "gpt", "value": "<text>" }
        """
        num = min(self.max_convs, len(self._dataset))
        for i in range(num):
            record = self._dataset[i]
            convs = record.get("conversations", [])
            if not isinstance(convs, list) or len(convs) == 0:
                continue
            yield i, convs

    def _iter_single_turn_events(self):
        """
        For single-turn mode:
        - Treat each conversation as exactly one request.
        - Use only the *first* human message as the prompt.
        """
        current_time = 0.0
        for conv_id, conv_msgs in self._iter_conversations():
            # Find first human message
            first_human = None
            for msg in conv_msgs:
                if msg.get("from", "").lower() == "human":
                    first_human = msg.get("value", "")
                    break
            if not first_human:
                continue

            # Generate inter-arrival time using exponential distribution
            if self.rate_lambda > 0:
                dt = random.expovariate(self.rate_lambda)
                current_time += dt

            event = RequestEvent(
                timestamp=current_time,
                conv_id=conv_id,
                turn_index=0,
                prompt=first_human,
                meta={"mode": "single-turn"},
            )
            yield event

    def _iter_multi_turn_events(self):
        """
        For multi-turn mode:
        - Each conversation can generate multiple turn events.
        - We consider each human message as a new "request".
        - The prompt for turn k is the conversation *history* up to that
          human message (so the LLM sees context).
        """
        current_time = 0.0
        for conv_id, conv_msgs in self._iter_conversations():
            history: List[str] = []
            turn_index = 0

            for msg in conv_msgs:
                role = msg.get("from", "").lower()
                text = msg.get("value", "")

                if not text:
                    continue

                if role == "human":
                    # New request: prompt = full history + this human message
                    history_plus = "\n".join(history + [f"Human: {text}"])

                    if self.rate_lambda > 0:
                        dt = random.expovariate(self.rate_lambda)
                        current_time += dt

                    event = RequestEvent(
                        timestamp=current_time,
                        conv_id=conv_id,
                        turn_index=turn_index,
                        prompt=history_plus,
                        meta={"mode": "multi-turn"},
                    )
                    turn_index += 1
                    yield event

                    # Also append to history so future turns see it
                    history.append(f"Human: {text}")

                elif role in ("gpt", "assistant", "bot"):
                    # Assistant reply – part of history, but not a new request.
                    history.append(f"Assistant: {text}")
                else:
                    # Unknown role; just append raw.
                    history.append(text)

    # ------------------------------------------------------------------
    # Public API used by scripts/demo_sharegpt_client.py
    # ------------------------------------------------------------------
    def run(
        self,
        mode: str,
        send_fn: Callable[[RequestEvent], None],
    ) -> None:
        """
        Run the simulator and call `send_fn` for each RequestEvent.

        Parameters
        ----------
        mode : str
            "single-turn" or "multi-turn".
        send_fn : Callable[[RequestEvent], None]
            Callback that will be invoked for each event. In the demo script
            this is just `print_sender`, which prints fields of the event.
        """
        mode = mode.lower()
        if mode not in {"single-turn", "multi-turn"}:
            raise ValueError(f"Unknown mode: {mode}")

        if mode == "single-turn":
            iterator = self._iter_single_turn_events()
        else:
            iterator = self._iter_multi_turn_events()

        for event in iterator:
            send_fn(event)